"""Shared marked-event transformer with pool-conditioned physical support."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .base_model import JointEventModel,interval_log_prob
from .flow import rq_spline,rq_spline_static
from .layers import history_query_mask
from .tick_mechanics import MIN_TICK,MAX_TICK,Q96,sqrt_at_tick,modify_liquidity
from .pool_mechanics import swap_economic_base


def bounds(grid,anchor,unit,width=False):
    # Float64: far from the anchor, asinh edges of one tick differ by less than a
    # float32 ulp, which would zero (or quantize) the interval mass.
    grid,anchor,unit=grid.double(),torch.as_tensor(anchor,dtype=torch.float64,device=grid.device),torch.as_tensor(unit,dtype=torch.float64,device=grid.device)
    if width:return ((grid-.5).clamp(min=.5)*unit).log(),((grid+.5)*unit).log()
    return torch.asinh((grid-anchor-.5)*unit/.001),torch.asinh((grid-anchor+.5)*unit/.001)


def discrete_nll(raw,target,anchor,unit,first,last,candidates=None,mask=None,width=False):
    selected=interval_log_prob(raw,*bounds(target,anchor,unit,width))
    if candidates is not None:
        scores=interval_log_prob(raw[:,None,:],*bounds(candidates,anchor[:,None],unit[:,None],width))
        if not mask.any(-1).all():raise ValueError('Empty burn support')
        normalizer=torch.logsumexp(scores.masked_fill(~mask,-torch.inf),-1)
    else:
        low,_=bounds(first,anchor,unit,width);_,high=bounds(last,anchor,unit,width)
        normalizer=interval_log_prob(raw,low,high)
    return (normalizer-selected).to(raw.dtype)


def discrete_nll_static(raw,target,anchor,unit,candidates,mask,width=False):
    """discrete_nll over a candidate support given as full-row tables, without host checks."""
    selected=interval_log_prob(raw,*bounds(target,anchor,unit,width))
    scores=interval_log_prob(raw[:,None,:],*bounds(candidates,anchor[:,None],unit[:,None],width))
    return (torch.logsumexp(scores.masked_fill(~mask,-torch.inf),-1)-selected).to(raw.dtype)


def native_range(lower,upper,meta):
    s=meta['tick_spacing']
    return (lower*s,upper*s) if meta['base_is_token0'] else (-upper*s,-lower*s)


def capacity_native(state,inventory,meta):
    from .pool_mechanics import InventoryCapacity
    ranges={native_range(lo,hi,meta):L for lo,uppers in inventory.items() for hi,L in uppers.items()}
    return InventoryCapacity(ranges,meta['tick_spacing'],meta['base_is_token0']).query(state)


def log_reserve(state,base0):
    if state.liquidity<=0:raise ValueError('Undefined relative amount at zero active liquidity')
    return math.log(state.liquidity)+(math.log(Q96)-math.log(state.sqrt) if base0 else math.log(state.sqrt)-math.log(Q96))


class PooledEventModel(JointEventModel):
    def __init__(self,scalers,d_model=64,layers=2,pool_count=0,pool_dropout=.1,
                 liquidity_context=False,retain_observed=False,gated_memory=False):
        super().__init__(seq_len=64,d_model=d_model,layers=layers,bins=8)
        self.project=nn.Linear(24,d_model);self.metadata_project=nn.Linear(5,d_model)
        self.pool_dropout=pool_dropout
        self.pool_embedding=nn.Embedding(pool_count+1,d_model,padding_idx=0) if pool_count else None
        if self.pool_embedding is not None:nn.init.zeros_(self.pool_embedding.weight)
        self.register_buffer('meta_scale',torch.tensor([8.,4.,1.,1.,1.]))
        self.mark_mean.copy_(torch.tensor(scalers['targets']['mean'][:3]))
        self.mark_std.copy_(torch.tensor(scalers['targets']['std'][:3]))
        with torch.no_grad():
            self.lower[-1].bias[4:8]=scalers['raw_features']['mean'][4]
            self.width[-1].bias[4:8]=scalers['raw_features']['mean'][5]
        self.liquidity_context=liquidity_context;self.retain_observed=retain_observed
        self.gated_memory=gated_memory
        if gated_memory:
            if not retain_observed:raise ValueError('Gated memory requires retained observations')
            self.memory_gate=nn.Parameter(torch.zeros(()))
        if liquidity_context:
            from .rollout_context import PRICE_OFFSETS
            self.depth_project=nn.Linear(len(PRICE_OFFSETS),d_model,bias=False)
            nn.init.zeros_(self.depth_project.weight)
        if retain_observed:
            self.generated_tag=nn.Parameter(torch.zeros(d_model))
            self.memory_age=nn.Linear(1,d_model,bias=False)
            nn.init.zeros_(self.memory_age.weight)

    def encode(self,x,kinds,mask,metadata,pool_id=None,*,depth=None,observed=None,rollout_steps=None):
        # Shared pretrained encoder, with a zero-initialized residual blend.
        # Base is computed first so its stochastic training path is unchanged.
        if self.gated_memory and observed is not None:
            base=self._encode_context(x,kinds,mask,metadata,pool_id,depth=depth)
            if rollout_steps is not None and bool((torch.as_tensor(rollout_steps)==0).all()):
                return base
            memory=self._encode_context(x,kinds,mask,metadata,pool_id,depth=depth,observed=observed,rollout_steps=rollout_steps)
            return base+self.memory_gate.tanh()*(memory-base)
        return self._encode_context(x,kinds,mask,metadata,pool_id,depth=depth,observed=observed,rollout_steps=rollout_steps)

    def _encode_context(self,x,kinds,mask,metadata,pool_id=None,*,depth=None,observed=None,rollout_steps=None):
        n,s,_=x.shape
        h=self.project(torch.cat([x,mask.float()],-1))+self.kind_emb(kinds)+self.position(torch.arange(s,device=x.device))
        if observed is not None:
            if not self.retain_observed or rollout_steps is None:
                raise ValueError('Retained observations require configured memory and event ages')
            steps=torch.as_tensor(rollout_steps,device=x.device)
            if steps.shape!=(n,) or (steps<0).any() or not bool((steps==steps[0]).all()):
                raise ValueError('Batch must have a common nonnegative generated-event count')
            k=min(int(steps[0]),64)
            if k:
                ox=observed['x'];ok=observed['history_kind'];om=observed['history_mask']
                if ox.shape!=(n,64,12):raise ValueError('Expected original 64 observed tokens')
                old=self.project(torch.cat([ox,om.float()],-1))+self.kind_emb(ok)+self.position(torch.arange(64,device=x.device))
                age=torch.log1p(steps.to(x.dtype))[:,None]/math.log1p(6000)
                old=old+self.memory_age(age)[:,None]
                new=self.project(torch.cat([x[:,-k:],mask[:,-k:].float()],-1))+self.kind_emb(kinds[:,-k:])+self.position(torch.arange(k,device=x.device))+self.generated_tag
                h=torch.cat([old,new],1);s=64+k
        meta=self.metadata_project(metadata/self.meta_scale)
        if depth is not None:
            if not self.liquidity_context:raise ValueError('Depth supplied to unconfigured model')
            meta=meta+self.depth_project(depth)
        if self.pool_embedding is not None:
            ids=torch.zeros(n,dtype=torch.long,device=x.device) if pool_id is None else pool_id
            if self.training:ids=ids.masked_fill(torch.rand(n,device=x.device)<self.pool_dropout,0)
            meta=meta+self.pool_embedding(ids)
        h=torch.cat([h,self.query.expand(n,-1,-1)+meta[:,None]],1)
        return self.norm(self.encoder(h,mask=history_query_mask(s,1,x.device,h.dtype)))[:,-1]

    def range_hidden(self,h,lower,anchor,unit,width=None):
        offset=torch.asinh((lower-anchor)*unit/.001)
        h=h+self.lower_cond(offset[:,None])
        return h if width is None else h+self.range_cond(torch.stack([offset,(width*unit).log()],-1))

    def encode_batch(self,b):
        return self.encode(b['x'],b['history_kind'],b['history_mask'],b['metadata'],b.get('pool_id'),
                           depth=b.get('depth') if self.liquidity_context else None,
                           observed=b.get('observed'),rollout_steps=b.get('rollout_steps'))

    def flow_cdf(self,raw,x,index=0,static=False):
        shift,scale,w,h,d=self.flow.split(raw)
        z=(x-self.mark_mean[index])/self.mark_std[index]
        u=(rq_spline_static if static else rq_spline)((z-shift)*torch.exp(-scale),w,h,d,bound=self.flow.bound)[0]
        return torch.special.log_ndtr(u)

    def amount_nll(self,raw,x,index,cap=None):
        value=-self.flow.log_prob(raw,(x-self.mark_mean[index])/self.mark_std[index])+self.mark_std[index].log()
        if cap is not None:
            if torch.any(x>=cap):raise ValueError('Target outside physical buy support')
            value=value+self.flow_cdf(raw,cap,index)
        return value

    def losses(self,b):
        x=b['x'];h=self.encode_batch(b);dt=b['dt'].float();kind=b['kind']
        y=b['targets'].to(x.dtype);valid=b['target_mask'];raw=self.clock(h)
        rates=(F.softplus(raw[:,4:])+.00001).clamp(max=20.)
        gap=torch.logsumexp(raw[:,1:4].log_softmax(-1)+torch.log(-torch.expm1(-rates))-(dt[:,None]-1)*rates,-1)
        wait=torch.where(dt==0,F.softplus(-raw[:,0]),F.softplus(raw[:,0])-gap)
        types=F.cross_entropy(self.types(self.condition(h,dt)),kind,reduction='none')
        m=self.condition(h,dt,kind);mark=torch.zeros_like(wait);parts={}
        def add(name,selection,values):
            nonlocal mark
            mark=mark.index_add(0,torch.where(selection)[0],values)
            parts[name]=(values.sum(),selection.sum())
        for side in [0,1]:
            selected=(kind==side)&valid[:,0]
            if selected.any():
                raw_size=self.swap_size(m[selected])
                # Near-exhaustion native-unit differences disappear in float32.
                v=self.amount_nll(raw_size.double(),b['targets'][selected,0].double(),0,b['buy_log_cap'][selected].double()).to(x.dtype) if side else self.amount_nll(raw_size,y[selected,0],0)
                add('swap_buy' if side else 'swap_sell',selected,v)
        for code,label in [(2,'mint'),(3,'burn')]:
            selected=kind==code
            if not selected.any():continue
            q=m[selected];anchor=b['anchor'][selected].to(x.dtype);unit=b['unit'][selected].to(x.dtype)
            lower=b['lower'][selected].to(x.dtype);width=b['width'][selected].to(x.dtype)
            minimum=b['grid_min'][selected].to(x.dtype);maximum=b['grid_max'][selected].to(x.dtype)
            support=dict(candidates=b['lower_candidates'],mask=b['lower_mask']) if code==3 else {}
            l=discrete_nll(self.lower(q),lower,anchor,unit,minimum,maximum-1,**support)
            q=self.range_hidden(q,lower,anchor,unit)
            support=dict(candidates=b['width_candidates'],mask=b['width_mask']) if code==3 else {}
            w=discrete_nll(self.width(q),width,anchor,unit,torch.ones_like(width),maximum-lower,width=True,**support)
            q=q+self.range_cond(torch.stack([torch.asinh((lower-anchor)*unit/.001),(width*unit).log()],-1))
            a=torch.zeros_like(l)
            if code==2:
                good=valid[selected,1]
                if good.any():a[good]=self.amount_nll(self.mint_amount(q[good]),y[selected,1][good],1)
            else:
                # Undefined available/L at zero liquidity is explicitly masked.
                available=torch.where(valid[selected,6],y[selected,6],torch.zeros_like(y[selected,6]))
                q=q+self.available(available[:,None]);full=y[selected,3]
                a=F.binary_cross_entropy_with_logits(self.burn_full(q).squeeze(-1),full,reduction='none')
                partial=valid[selected,2]
                if partial.any():
                    z=y[selected,2][partial]
                    a[partial]+=self.amount_nll(self.burn_fraction(q[partial]),z,2)-F.softplus(z)-F.softplus(-z)
            add(label+'_lower',selected,l);add(label+'_width',selected,w);add(label+'_amount',selected,a)
        total=wait+types+mark
        parts.update(wait=(wait.sum(),len(wait)),kind=(types.sum(),len(wait)),joint=(total.sum(),len(wait)))
        return total.mean(),parts

    def amount_nll_static(self,raw,x,index,cap=None):
        """amount_nll without host synchronisation; returns (values, outside-support flag)."""
        value=-self.flow.log_prob(raw,(x-self.mark_mean[index])/self.mark_std[index],static=True)+self.mark_std[index].log()
        if cap is None:return value,None
        return value+self.flow_cdf(raw,cap,index,static=True),x>=cap

    def losses_static(self,b):
        """Same per-row losses and parts as losses(), with no data-dependent shapes.

        Every head is evaluated on every row and masked, so the step never waits on
        the GPU. Inputs of rows a head does not apply to are replaced by safe values
        first, keeping all evaluations finite (masked rows then contribute exactly
        zero, including to gradients). Burn candidates arrive as full-row tables
        (lower_candidates_full / width_candidates_full, see
        dexnet.evaluation.expand_candidates); non-burn rows get a placeholder
        candidate. Physical-support violations, which losses() raises immediately,
        are returned as parts['violations'] for the caller to check periodically.
        """
        x=b['x'];h=self.encode_batch(b);dt=b['dt'].float();kind=b['kind']
        y=b['targets'].to(x.dtype);valid=b['target_mask'];raw=self.clock(h)
        rates=(F.softplus(raw[:,4:])+.00001).clamp(max=20.)
        gap=torch.logsumexp(raw[:,1:4].log_softmax(-1)+torch.log(-torch.expm1(-rates))-(dt[:,None]-1)*rates,-1)
        wait=torch.where(dt==0,F.softplus(-raw[:,0]),F.softplus(raw[:,0])-gap)
        types=F.cross_entropy(self.types(self.condition(h,dt)),kind,reduction='none')
        m=self.condition(h,dt,kind);mark=torch.zeros_like(wait);parts={};violations=torch.zeros((),dtype=torch.long,device=x.device)
        def add(name,selection,values):
            nonlocal mark
            values=torch.where(selection,values,torch.zeros_like(values))
            mark=mark+values;parts[name]=(values.sum(),selection.sum())
        # Swaps: one head evaluation for all rows; sells in float32, buys in float64 with the capacity term.
        raw_size=self.swap_size(m);safe_y0=torch.where(valid[:,0],y[:,0],torch.zeros_like(y[:,0]))
        sell=(kind==0)&valid[:,0];buy=(kind==1)&valid[:,0]
        v,_=self.amount_nll_static(raw_size,safe_y0,0);add('swap_sell',sell,v)
        t0=torch.where(buy,b['targets'][:,0].double(),torch.zeros_like(b['targets'][:,0].double()))
        cap=torch.where(buy,b['buy_log_cap'].double(),t0+1.)
        v,outside=self.amount_nll_static(raw_size.double(),t0,0,cap);violations=violations+(outside&buy).sum();add('swap_buy',buy,v.to(x.dtype))
        # Liquidity events: safe range inputs on rows that are neither mints nor burns.
        lp=kind>=2;anchor=b['anchor'].to(x.dtype);unit=b['unit'].to(x.dtype)
        minimum=b['grid_min'].to(x.dtype);maximum=b['grid_max'].to(x.dtype)
        lower=torch.where(lp,b['lower'].to(x.dtype),anchor.round().clamp(minimum,maximum-2))
        width=torch.where(lp,b['width'].to(x.dtype),torch.ones_like(anchor))
        raw_lower=self.lower(m)
        l_mint=discrete_nll(raw_lower,lower,anchor,unit,minimum,maximum-1)
        burn=kind==3
        lc=torch.where(burn[:,None],b['lower_candidates_full'].to(x.dtype),lower[:,None]);lm=b['lower_mask_full']|~burn[:,None]
        violations=violations+(burn&~b['lower_mask_full'].any(-1)).sum()
        l_burn=discrete_nll_static(raw_lower,lower,anchor,unit,lc,lm)
        q=self.range_hidden(m,lower,anchor,unit);raw_width=self.width(q)
        w_mint=discrete_nll(raw_width,width,anchor,unit,torch.ones_like(width),maximum-lower,width=True)
        wc=torch.where(burn[:,None],b['width_candidates_full'].to(x.dtype),width[:,None]);wm=b['width_mask_full']|~burn[:,None]
        violations=violations+(burn&~b['width_mask_full'].any(-1)).sum()
        w_burn=discrete_nll_static(raw_width,width,anchor,unit,wc,wm,width=True)
        q=q+self.range_cond(torch.stack([torch.asinh((lower-anchor)*unit/.001),(width*unit).log()],-1))
        mint=kind==2;good=mint&valid[:,1]
        a_mint,_=self.amount_nll_static(self.mint_amount(q),torch.where(good,y[:,1],torch.zeros_like(y[:,1])),1)
        a_mint=torch.where(good,a_mint,torch.zeros_like(a_mint))
        available=torch.where(valid[:,6],y[:,6],torch.zeros_like(y[:,6]))
        qb=q+self.available(available[:,None]);full=torch.where(burn,y[:,3],torch.zeros_like(y[:,3]))
        a_burn=F.binary_cross_entropy_with_logits(self.burn_full(qb).squeeze(-1),full,reduction='none')
        partial=burn&valid[:,2];z=torch.where(partial,y[:,2],torch.zeros_like(y[:,2]))
        frac,_=self.amount_nll_static(self.burn_fraction(qb),z,2)
        a_burn=a_burn+torch.where(partial,frac-F.softplus(z)-F.softplus(-z),torch.zeros_like(frac))
        for label,selection,l,w,a in (('mint',mint,l_mint,w_mint,a_mint),('burn',burn,l_burn,w_burn,a_burn)):
            add(label+'_lower',selection,l);add(label+'_width',selection,w);add(label+'_amount',selection,a)
        total=wait+types+mark
        parts.update(wait=(wait.sum(),torch.tensor(len(wait),device=x.device)),kind=(types.sum(),torch.tensor(len(wait),device=x.device)),
                     joint=(total.sum(),torch.tensor(len(wait),device=x.device)),violations=(violations,violations))
        return total.mean(),parts

    @torch.no_grad()
    def sample_grid(self,raw,anchor,unit,first,last,width=False,candidates=None):
        if candidates is not None:
            values=raw.new_tensor(candidates);scores=interval_log_prob(raw,*bounds(values,anchor,unit,width))
            return int(values[torch.multinomial(scores.softmax(-1),1)])
        lo=raw.new_tensor([first]);hi=raw.new_tensor([last]);a,_=bounds(lo,anchor,unit,width);_,b=bounds(hi,anchor,unit,width)
        mass=interval_log_prob(raw,a,b);u=torch.rand_like(mass).clamp(min=1e-7,max=1-1e-7).log()+mass
        left,right=int(first),int(last)
        while left<right:
            mid=(left+right)//2;_,bound=bounds(raw.new_tensor([mid]),anchor,unit,width)
            if bool(interval_log_prob(raw,a,bound)>=u):right=mid
            else:left=mid+1
        return left

    @torch.no_grad()
    def sample_event(self,b,state,inventory,meta):
        self.eval();h=self.encode_batch(b);raw=self.clock(h)
        if bool(torch.rand((),device=h.device)<raw[0,0].sigmoid()):dt=0
        else:
            component=int(torch.multinomial(raw[0,1:4].softmax(0),1));rate=float((F.softplus(raw[0,4+component])+.00001).clamp(max=20.))
            dt=1+int(math.floor(-math.log(max(float(torch.rand(())),1e-12))/rate))
        d=raw.new_tensor([dt]);logits=self.types(self.condition(h,d))[0]
        if not inventory:logits[[0,1,3]]=-torch.inf
        kind=int(torch.multinomial(logits.softmax(0),1));q=self.condition(h,d,torch.tensor([kind],device=h.device));event=dict(kind=kind,dt=dt)
        if state.liquidity<=0:raise ValueError('Undefined relative amount at zero active liquidity; retain as failed rollout')
        if kind<2:
            r=self.swap_size(q);u=torch.randn(1,device=h.device);reserve=log_reserve(state,meta['base_is_token0'])
            if kind==1:
                cap=capacity_native(state,inventory,meta)
                if cap<1:raise ValueError('No physical buy capacity')
                r=r.double();logcap=math.log(cap+1)-reserve;fc=self.flow_cdf(r,r.new_tensor([logcap])).exp()
                prob=torch.minimum(torch.special.ndtr(u.double())*fc,torch.nextafter(fc,torch.zeros_like(fc)))
                if bool(prob<=0):raise ValueError('Supported CDF underflow')
                z=float(self.flow.inverse(r,torch.special.ndtri(prob))*self.mark_std[0]+self.mark_mean[0])
                if z>logcap+1e-9:raise ValueError('Supported sample exceeds capacity')
                from decimal import Decimal
                amount=max(1,min(cap,int(Decimal(cap+1)*Decimal.from_float(min(1.,math.exp(z-logcap))))))
            else:
                z=float(self.flow.inverse(r,u)*self.mark_std[0]+self.mark_mean[0])+reserve
                if not math.isfinite(z) or z>170:raise ValueError('Excessive base size')
                amount=max(1,round(math.exp(z)))
            event['base_amount']=amount if kind==0 else -amount;return event
        spacing=meta['tick_spacing'];sign=1 if meta['base_is_token0'] else -1
        anchor=raw.new_tensor([sign*state.tick/spacing]);unit=raw.new_tensor([spacing*math.log1p(.0001)])
        minimum=-((-MIN_TICK)//spacing);maximum=MAX_TICK//spacing
        lower=self.sample_grid(self.lower(q),anchor,unit,minimum,maximum-1,candidates=sorted(inventory) if kind==3 else None)
        lo=raw.new_tensor([lower]);q=self.range_hidden(q,lo,anchor,unit)
        width=self.sample_grid(self.width(q),anchor,unit,1,maximum-lower,width=True,candidates=sorted(u-lower for u in inventory[lower]) if kind==3 else None)
        q=q+self.range_cond(torch.stack([torch.asinh((lo-anchor)*unit/.001),(raw.new_tensor([width])*unit).log()],-1))
        if kind==2:
            z=float(self.flow.inverse(self.mint_amount(q),torch.randn(1,device=h.device))*self.mark_std[1]+self.mark_mean[1])+math.log(state.liquidity)
            if not math.isfinite(z) or z>=128*math.log(2):raise ValueError('Mint outside uint128')
            amount=max(1,round(math.exp(z)))
        else:
            available=inventory[lower][lower+width];q=q+self.available(raw.new_tensor([[math.log(available/state.liquidity)]]))
            if bool(torch.rand((),device=h.device)<self.burn_full(q).sigmoid().squeeze()):amount=available
            else:
                z=float(self.flow.inverse(self.burn_fraction(q),torch.randn(1,device=h.device))*self.mark_std[2]+self.mark_mean[2])
                amount=max(1,min(available,round(available*float(torch.sigmoid(torch.tensor(z,dtype=torch.float64))))))
        event.update(lower=lower,upper=lower+width,amount=amount);return event


def apply_pooled_event(state,book,inventory,event,meta):
    if event['kind']<2:
        new,detail=swap_economic_base(state,book,event['base_amount'],meta)
        state.sqrt,state.tick,state.liquidity=new.sqrt,new.tick,new.liquidity;return detail
    lo,hi=event['lower'],event['upper'];amount=event['amount'];native_lo,native_hi=native_range(lo,hi,meta)
    if not MIN_TICK<=native_lo<native_hi<=MAX_TICK or amount<1:raise ValueError('Invalid LP proposal')
    if event['kind']==3 and amount>inventory.get(lo,{}).get(hi,0):raise ValueError('Burn outside inventory')
    if event['kind']==2:
        count=2*(MAX_TICK//meta['tick_spacing'])+1;maximum=((1<<128)-1)//count
        if book.tick(native_lo)[0]+amount>maximum or book.tick(native_hi)[0]+amount>maximum:raise ValueError('Mint exceeds tick liquidity limit')
    delta=amount if event['kind']==2 else -amount;modify_liquidity(state,book,native_lo,native_hi,delta)
    level=inventory.setdefault(lo,{});new=level.get(hi,0)+delta
    if new:level[hi]=new
    else:level.pop(hi,None)
    if not level:inventory.pop(lo,None)
    return dict(liquidity_delta=delta if native_lo<=state.tick<native_hi else 0)
