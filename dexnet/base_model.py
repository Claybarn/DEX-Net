"""Joint marked event model with tick-grid mint ranges and inventory-supported burns.

Kinds: 0 WETH sell, 1 WETH buy, 2 mint, 3 nonzero burn. Burn(0) is a fee poke,
not a liquidity change. All conditioning is history, then next gap/type/range.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .layers import history_query_mask, _head
from .flow import ConditionalSizeFlow

MIN_GRID=-88727
MAX_GRID=88727
N_FEATURES=11


def logistic_raw(raw):
    pi,loc,scale=raw.chunk(3,-1)
    return pi.log_softmax(-1),loc,F.softplus(scale)+.05


def interval_log_prob(raw,low,high):
    """Stable logistic-mixture mass in transformed coordinate intervals."""
    pi,loc,scale=logistic_raw(raw)
    a=(low[...,None]-loc)/scale;b=(high[...,None]-loc)/scale
    logmass=F.logsigmoid(b)+F.logsigmoid(-a)+torch.log(-torch.expm1((a-b).clamp(max=-1e-12)))
    return torch.logsumexp(pi+logmass,-1)


def grid_bounds(grid,anchor,width=False):
    if width:
        return (grid-.5).clamp(min=.5).log(),(grid+.5).log()
    return torch.asinh((grid-anchor-.5)/10),torch.asinh((grid-anchor+.5)/10)


def grid_log_prob(raw,grid,anchor,width=False):
    low,high=grid_bounds(grid,anchor,width)
    return interval_log_prob(raw,low,high)


def supported_nll(raw,target,anchor,candidates=None,mask=None,width=False,max_grid=None):
    selected=grid_log_prob(raw,target,anchor,width)
    if candidates is not None:
        scores=grid_log_prob(raw[:,None,:],candidates,anchor[:,None],width).masked_fill(~mask,-torch.inf)
        if not mask.any(-1).all():raise ValueError('Empty range support')
        normalizer=torch.logsumexp(scores,-1)
    else:
        first=torch.ones_like(target) if width else torch.full_like(target,MIN_GRID)
        last=max_grid if width else torch.full_like(target,MAX_GRID-1)
        low,_=grid_bounds(first,anchor,width);_,high=grid_bounds(last,anchor,width)
        normalizer=interval_log_prob(raw,low,high)
    return normalizer-selected


class JointEventModel(nn.Module):
    def __init__(self,seq_len=64,d_model=64,layers=2,bins=8):
        super().__init__();self.seq_len=seq_len;self.d_model=d_model
        self.register_buffer("mark_mean",torch.zeros(3));self.register_buffer("mark_std",torch.ones(3))
        self.project=nn.Linear(N_FEATURES,d_model);self.kind_emb=nn.Embedding(4,d_model)
        self.position=nn.Embedding(seq_len,d_model);self.query=nn.Parameter(torch.zeros(1,1,d_model))
        layer=nn.TransformerEncoderLayer(d_model,4,4*d_model,.1,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False);self.norm=nn.LayerNorm(d_model)
        self.clock=_head(d_model,7)  # continuation, three geometric mixture logits/rates
        self.gap=nn.Linear(2,d_model);self.types=_head(d_model,4);self.target_kind=nn.Embedding(4,d_model)
        self.lower=_head(d_model,12);self.lower_cond=nn.Linear(1,d_model)
        self.width=_head(d_model,12);self.range_cond=nn.Linear(2,d_model)
        self.flow=ConditionalSizeFlow(bins);self.swap_size=_head(d_model,self.flow.n_params)
        self.mint_amount=_head(d_model,self.flow.n_params)
        self.available=nn.Linear(1,d_model);self.burn_full=_head(d_model,1);self.burn_fraction=_head(d_model,self.flow.n_params)
        for head in [self.swap_size,self.mint_amount,self.burn_fraction]:
            nn.init.zeros_(head[-1].weight);head[-1].bias.data.copy_(self.flow.identity_bias())
        for head,loc in [(self.lower,0.),(self.width,3.)]:
            nn.init.zeros_(head[-1].weight);head[-1].bias.data.copy_(torch.tensor([0.]*4+[loc]*4+[1.]*4))

    def encode(self,x,kinds):
        n,s,_=x.shape;h=self.project(x)+self.kind_emb(kinds)+self.position(torch.arange(s,device=x.device))
        h=torch.cat([h,self.query.expand(n,-1,-1)],1)
        return self.norm(self.encoder(h,mask=history_query_mask(s,1,x.device,h.dtype)))[:,-1]

    def condition(self,h,dt,kind=None):
        h=h+self.gap(torch.stack([(dt==0).float(),torch.log1p(dt.float())],-1))
        return h if kind is None else h+self.target_kind(kind)

    def range_hidden(self,h,lower,anchor,width=None):
        offset=torch.asinh((lower-anchor)/10)
        h=h+self.lower_cond(offset[:,None])
        return h if width is None else h+self.range_cond(torch.stack([offset,width.log()],-1))

    def burn_lower_loss(self,raw,target,anchor,candidates,mask):
        """Default burn lower-bound distribution; experiments may override it."""
        return supported_nll(raw,target,anchor,candidates,mask)

    def swap_nll(self,raw,b,selection):
        return -self.flow.log_prob(raw,(b['amount'][selection]-self.mark_mean[0])/self.mark_std[0])+self.mark_std[0].log()

    def sample_swap_amount(self,q,kind,state,inventory):
        value=float(self.flow.inverse(self.swap_size(q),torch.randn(1,device=q.device))*self.mark_std[0]+self.mark_mean[0])
        if not math.isfinite(value) or value>100:raise ValueError('Nonfinite/excessive sampled swap; no silent cap')
        return max(1,round(math.exp(value)*1e18))*(1 if kind==0 else -1)

    def losses(self,b):
        h=self.encode(b['x'],b['history_kind']);dt=b['dt'];kind=b['kind'];raw=self.clock(h)
        p=(F.softplus(raw[:,4:])+.00001).clamp(max=20.)
        gap_log=torch.logsumexp(raw[:,1:4].log_softmax(-1)+torch.log(-torch.expm1(-p))-(dt[:,None]-1)*p,-1)
        wait=torch.where(dt==0,F.softplus(-raw[:,0]),F.softplus(raw[:,0])-gap_log)
        type_loss=F.cross_entropy(self.types(self.condition(h,dt)),kind,reduction='none')
        m=self.condition(h,dt,kind);mark=torch.zeros_like(wait);parts={}
        swap=kind<2;mint=kind==2;burn=kind==3
        if swap.any():
            v=self.swap_nll(self.swap_size(m[swap]),b,swap);mark[swap]=v;parts['swap_size']=(v.sum(),swap.sum())
        lp=kind>=2
        if lp.any():
            for selection,label in [(mint,'mint'),(burn,'burn')]:
                if not selection.any():continue
                q=m[selection];anchor=b['anchor'][selection];lower=b['lower'][selection];width=b['width'][selection]
                if label=='burn':
                    l=self.burn_lower_loss(self.lower(q),lower,anchor,b['lower_candidates'],b['lower_mask'])
                else:l=supported_nll(self.lower(q),lower,anchor)
                q=self.range_hidden(q,lower,anchor)
                ranges=(dict(candidates=b['width_candidates'],mask=b['width_mask']) if label=='burn' else {})
                w=supported_nll(self.width(q),width,anchor,width=True,max_grid=MAX_GRID-lower,**ranges)
                q=q+self.range_cond(torch.stack([torch.asinh((lower-anchor)/10),width.log()],-1))
                if label=='mint':a=-self.flow.log_prob(self.mint_amount(q),(b['amount'][selection]-self.mark_mean[1])/self.mark_std[1])+self.mark_std[1].log()
                else:
                    q=q+self.available(b['available'][selection,None]);full=b['full'][selection]
                    a=F.binary_cross_entropy_with_logits(self.burn_full(q).squeeze(-1),full,reduction='none')
                    partial=full==0
                    if partial.any():
                        z=b['amount'][selection][partial]
                        # Density in fraction coordinates, including logit Jacobian.
                        a=a.clone();a[partial]+=-self.flow.log_prob(self.burn_fraction(q[partial]),(z-self.mark_mean[2])/self.mark_std[2])+self.mark_std[2].log()-F.softplus(z)-F.softplus(-z)
                mark[selection]=l+w+a
                for suffix,v in [('lower',l),('width',w),('amount',a)]:parts[label+'_'+suffix]=(v.sum(),selection.sum())
        total=wait+type_loss+mark
        parts.update(wait=(wait.sum(),len(wait)),kind=(type_loss.sum(),len(wait)),joint=(total.sum(),len(wait)))
        return total.mean(),parts

    @torch.no_grad()
    def sample_range_grid(self,raw,anchor,*,width=False,maximum=None,candidates=None):
        """Exact discrete inverse CDF on bounded grid, or categorical active support."""
        if candidates is not None:
            scores=grid_log_prob(raw.expand(len(candidates),-1),candidates,anchor.expand_as(candidates),width)
            return candidates[torch.multinomial(scores.softmax(0),1)].item()
        first=1 if width else MIN_GRID;last=int(maximum) if width else MAX_GRID-1
        low=raw.new_tensor([first]);high=raw.new_tensor([last]);a,_=grid_bounds(low,anchor,width);_,b=grid_bounds(high,anchor,width)
        mass=interval_log_prob(raw,a,b);u=torch.rand_like(mass).clamp(min=1e-7,max=1-1e-7).log()+mass
        left,right=first,last
        while left<right:
            mid=(left+right)//2;_,bound=grid_bounds(raw.new_tensor([mid]),anchor,width)
            if bool((interval_log_prob(raw,a,bound)>=u).item()):right=mid
            else:left=mid+1
        return left

    @torch.no_grad()
    def sample_event(self,x,kinds,state,inventory):
        """One coherent sampled event, with exact range/amount feasibility.

        Inventory maps compressed lower ticks to {compressed upper: native L}.
        Returns proposal plus wait; application is separate and deterministic.
        """
        self.eval();h=self.encode(x,kinds);raw=self.clock(h)
        if bool((torch.rand((),device=x.device)<raw[0,0].sigmoid()).item()):dt=0
        else:
            component=int(torch.multinomial(raw[0,1:4].softmax(0),1));rate=float((F.softplus(raw[0,4+component])+.00001).clamp(max=20.))
            dt=1+int(math.floor(-math.log(max(float(torch.rand(())),1e-12))/rate))
        d=raw.new_tensor([dt]);logits=self.types(self.condition(h,d))[0]
        if not inventory:logits[[0,1,3]]=-torch.inf
        kind=int(torch.multinomial(logits.softmax(0),1));q=self.condition(h,d,torch.tensor([kind],device=x.device))
        def flow_value(head,index):
            return float(self.flow.inverse(head(q),torch.randn(1,device=x.device))*self.mark_std[index]+self.mark_mean[index])
        event=dict(kind=kind,dt=dt)
        if kind<2:
            event['amount1']=self.sample_swap_amount(q,kind,state,inventory)
            return event
        anchor=raw.new_tensor([state.tick//10]);candidate=(raw.new_tensor(sorted(inventory)) if kind==3 else None)
        lower=int(self.sample_range_grid(self.lower(q),anchor,candidates=candidate))
        lo=raw.new_tensor([lower]);q=self.range_hidden(q,lo,anchor)
        widths=(raw.new_tensor(sorted(u-lower for u in inventory[lower])) if kind==3 else None)
        width=int(self.sample_range_grid(self.width(q),anchor,width=True,maximum=MAX_GRID-lower,candidates=widths))
        q=q+self.range_cond(torch.stack([torch.asinh((lo-anchor)/10),raw.new_tensor([width]).log()],-1))
        if kind==2:
            value=flow_value(self.mint_amount,1)
            if not math.isfinite(value) or value>100:raise ValueError('Nonfinite/excessive sampled mint; no silent cap')
            amount=max(1,round(math.exp(value)*max(state.liquidity,1)))
            if amount>=(1<<128):raise ValueError('Sampled mint exceeds uint128')
        else:
            available=inventory[lower][lower+width];q=q+self.available(raw.new_tensor([[math.log(available/max(state.liquidity,1))]]))
            full=bool((torch.rand((),device=x.device)<self.burn_full(q).sigmoid().squeeze()).item())
            if full:amount=available
            else:
                z=flow_value(self.burn_fraction,2);fraction=float(torch.sigmoid(torch.tensor(z,dtype=torch.float64)))
                amount=max(1,min(available,round(available*fraction)))
        event.update(lower=lower*10,upper=(lower+width)*10,amount=amount)
        return event


def apply_event(state,book,inventory,event):
    """Apply sampled marks through validated tick mechanics; atomic LP validation."""
    from .tick_mechanics import swap_base,modify_liquidity
    from .utils import range_update
    if event['kind']<2:
        result,detail=swap_base(state,book,event['amount1'])
        state.sqrt,state.tick,state.liquidity=result.sqrt,result.tick,result.liquidity
        return detail
    lower,upper,amount=event['lower'],event['upper'],event['amount']
    if amount<1 or lower%10 or upper%10 or not MIN_GRID*10<=lower<upper<=MAX_GRID*10:raise ValueError('Invalid LP marks')
    if event['kind']==3 and amount>inventory.get(lower//10,{}).get(upper//10,0):raise ValueError('Burn exceeds existing range')
    if event['kind']==2:
        maximum=((1<<128)-1)//(MAX_GRID-MIN_GRID+1)
        if book.tick(lower)[0]+amount>maximum or book.tick(upper)[0]+amount>maximum:raise ValueError('Mint exceeds per-tick liquidity capacity')
    delta=amount*(1 if event['kind']==2 else -1)
    modify_liquidity(state,book,lower,upper,delta);range_update(inventory,lower//10,upper//10,delta)
    return dict(liquidity_delta=delta if lower<=state.tick<upper else 0)
