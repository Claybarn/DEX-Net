"""Mechanics ablation: the event model with learned state transitions instead of the pool replica.

DEX-Net applies each sampled event to an exact Uniswap v3 replica, which computes the
event's effect on the price and on active liquidity. This model learns those effects
instead. After the event's type and marks are sampled (as in the no-profile model), two
extra heads give its log price return and its change in log active liquidity:

  return     swaps only (liquidity events never move the price): a zero indicator, then a
             spline flow on log|return|; the sign follows the side (a sell lowers the base price)
  liquidity  every event: a zero indicator, then a spline flow on asinh(change / 0.01)

There is no tick book and no position inventory. Buys are not truncated at the pool's
capacity, burn ranges are not restricted to existing positions, and burn size is drawn
like mint size, relative to active liquidity. The simulated state is the base log price,
log active liquidity and the virtual base reserve, all advanced by the learned changes.
Transition NLLs are in the transformed coordinates (log|return|, asinh change).
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .layers import _head
from .model import PooledEventModel,discrete_nll
from .tick_mechanics import MIN_TICK,MAX_TICK

RET,LP_RATIO,LIQ=1,3,9          # history feature columns: price return, LP amount / L, change in log L
RET_LOC,RET_SCALE=-9.,2.5       # rough centring of log|return| before the conditional flow
LIQ_UNIT=.01                    # asinh unit of the liquidity change
ZERO=1e-10                      # |value| below this counts as exactly zero (scaling round trip)
LOG_TICK=math.log1p(.0001)


class LearnedTransitionModel(PooledEventModel):
    learned=True

    def __init__(self,scalers,learned_transitions=True,**config):
        super().__init__(scalers,**config)
        if self.liquidity_context:raise ValueError('The learned-transition ablation has no book, so no liquidity profile')
        d=self.d_model;n=self.flow.n_params
        self.register_buffer('feature_mean',torch.tensor(scalers['raw_features']['mean'],dtype=torch.float64))
        self.register_buffer('feature_std',torch.tensor(scalers['raw_features']['std'],dtype=torch.float64))
        self.burn_amount=_head(d,n);self.ret_zero=_head(d,1);self.ret_size=_head(d,n)
        self.liq_zero=_head(d,1);self.liq_size=_head(d,n)
        self.transition_cond=nn.Linear(3,d);self.return_cond=nn.Linear(2,d)
        for head in [self.burn_amount,self.ret_size,self.liq_size]:
            nn.init.zeros_(head[-1].weight);head[-1].bias.data.copy_(self.flow.identity_bias())
        for layer in [self.transition_cond,self.return_cond]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def raw_next(self,b):
        """The target event's own (unscaled) history features and mask."""
        return b['next_x'].double()*self.feature_std+self.feature_mean,b['next_mask']

    def transition_context(self,size,amount,swap):
        """size: standardized log swap size; amount: standardized log LP amount / L."""
        return self.transition_cond(torch.stack([size,amount,swap],-1))

    def losses(self,b):return self.losses_static(b)

    def losses_static(self,b):
        x=b['x'];h=self.encode_batch(b);dt=b['dt'].float();kind=b['kind']
        y=b['targets'].to(x.dtype);valid=b['target_mask'];raw=self.clock(h);device=x.device
        rates=(F.softplus(raw[:,4:])+.00001).clamp(max=20.)
        gap=torch.logsumexp(raw[:,1:4].log_softmax(-1)+torch.log(-torch.expm1(-rates))-(dt[:,None]-1)*rates,-1)
        wait=torch.where(dt==0,F.softplus(-raw[:,0]),F.softplus(raw[:,0])-gap)
        types=F.cross_entropy(self.types(self.condition(h,dt)),kind,reduction='none')
        m=self.condition(h,dt,kind);mark=torch.zeros_like(wait);parts={}
        def add(name,selection,values):
            nonlocal mark
            values=torch.where(selection,values,torch.zeros_like(values))
            mark=mark+values;parts[name]=(values.sum(),selection.sum())
        # Swap size, uncapped on both sides (no book, so no buy capacity).
        size=torch.where(valid[:,0],y[:,0],torch.zeros_like(y[:,0]))
        v,_=self.amount_nll_static(self.swap_size(m),size,0)
        add('swap_sell',(kind==0)&valid[:,0],v);add('swap_buy',(kind==1)&valid[:,0],v)
        # Ranges over the whole grid for mints and burns alike (no inventory).
        lp=kind>=2;anchor=b['anchor'].to(x.dtype);unit=b['unit'].to(x.dtype)
        minimum=b['grid_min'].to(x.dtype);maximum=b['grid_max'].to(x.dtype)
        lower=torch.where(lp,b['lower'].to(x.dtype),anchor.round().clamp(minimum,maximum-2))
        width=torch.where(lp,b['width'].to(x.dtype),torch.ones_like(anchor))
        l=discrete_nll(self.lower(m),lower,anchor,unit,minimum,maximum-1)
        q=self.range_hidden(m,lower,anchor,unit)
        w=discrete_nll(self.width(q),width,anchor,unit,torch.ones_like(width),maximum-lower,width=True)
        q=q+self.range_cond(torch.stack([torch.asinh((lower-anchor)*unit/.001),(width*unit).log()],-1))
        # LP amount as log(amount / active L) for mints and burns.
        nxt,nmask=self.raw_next(b);ratio=nxt[:,LP_RATIO].to(x.dtype);has_ratio=lp&nmask[:,LP_RATIO]
        z=torch.where(has_ratio,ratio,torch.zeros_like(ratio))
        a_mint,_=self.amount_nll_static(self.mint_amount(q),z,1);a_burn,_=self.amount_nll_static(self.burn_amount(q),z,1)
        mint,burn=kind==2,kind==3
        for label,selection,a in (('mint',mint,a_mint),('burn',burn,a_burn)):
            add(label+'_lower',selection,l);add(label+'_width',selection,w)
            add(label+'_amount',selection,torch.where(has_ratio,a,torch.zeros_like(a)))
        # Learned transitions.
        swap=kind<2
        size_std=torch.where(swap&valid[:,0],(size-self.mark_mean[0])/self.mark_std[0],torch.zeros_like(size))
        amount_std=torch.where(has_ratio,(z-self.mark_mean[1])/self.mark_std[1],torch.zeros_like(z))
        t=torch.where(lp[:,None],q,m)+self.transition_context(size_std,amount_std,swap.to(x.dtype))
        ret=nxt[:,RET];ret_zero=ret.abs()<ZERO;has_ret=swap&nmask[:,RET]
        log_ret=torch.where(ret_zero,torch.zeros_like(ret),ret.abs().clamp(min=1e-300).log())
        z_ret=((log_ret-RET_LOC)/RET_SCALE).to(x.dtype)
        flow_ret=-self.flow.log_prob(self.ret_size(t),z_ret,static=True)+math.log(RET_SCALE)
        nll_ret=F.binary_cross_entropy_with_logits(self.ret_zero(t).squeeze(-1),ret_zero.to(x.dtype),reduction='none')
        nll_ret=nll_ret+torch.where(ret_zero,torch.zeros_like(flow_ret),flow_ret)
        add('return',has_ret,nll_ret)
        given=torch.where(has_ret[:,None],torch.stack([ret_zero.to(x.dtype),torch.where(ret_zero,torch.zeros_like(z_ret),z_ret)],-1),torch.zeros((len(t),2),device=device,dtype=x.dtype))
        t=t+self.return_cond(given)
        liq=nxt[:,LIQ];liq_zero=liq.abs()<ZERO;has_liq=nmask[:,LIQ]
        z_liq=torch.asinh(liq/LIQ_UNIT).to(x.dtype)
        flow_liq=-self.flow.log_prob(self.liq_size(t),z_liq,static=True)
        nll_liq=F.binary_cross_entropy_with_logits(self.liq_zero(t).squeeze(-1),liq_zero.to(x.dtype),reduction='none')
        nll_liq=nll_liq+torch.where(liq_zero,torch.zeros_like(flow_liq),flow_liq)
        add('liquidity',has_liq,nll_liq)
        total=wait+types+mark;count=torch.tensor(len(wait),device=device);none=torch.zeros((),dtype=torch.long,device=device)
        parts.update(wait=(wait.sum(),count),kind=(types.sum(),count),joint=(total.sum(),count),violations=(none,none))
        return total.mean(),parts

    @torch.no_grad()
    def _flow_sample(self,head,q,loc,scale):
        return float(self.flow.inverse(head(q),torch.randn(1,device=q.device)))*scale+loc

    @torch.no_grad()
    def sample_learned(self,b,state,meta):
        """One event and its learned effect. state: dict(log_liquidity, tick, log_reserve)."""
        self.eval();h=self.encode_batch(b);raw=self.clock(h);device=h.device
        if bool(torch.rand((),device=device)<raw[0,0].sigmoid()):dt=0
        else:
            component=int(torch.multinomial(raw[0,1:4].softmax(0),1));rate=float((F.softplus(raw[0,4+component])+.00001).clamp(max=20.))
            dt=1+int(math.floor(-math.log(max(float(torch.rand(())),1e-12))/rate))
        d=raw.new_tensor([dt]);kind=int(torch.multinomial(self.types(self.condition(h,d))[0].softmax(0),1))
        q=self.condition(h,d,torch.tensor([kind],device=device));event=dict(kind=kind,dt=dt)
        one=lambda *v:raw.new_tensor([list(v)])
        if kind<2:
            z=self._flow_sample(self.swap_size,q,float(self.mark_mean[0]),float(self.mark_std[0]))
            value=z+state['log_reserve']
            if not math.isfinite(value) or value>170:return dict(event,error='Excessive base size')
            t=q+self.transition_cond(one((z-float(self.mark_mean[0]))/float(self.mark_std[0]),0.,1.))
            zero=bool(torch.rand((),device=device)<self.ret_zero(t).sigmoid().squeeze())
            z_ret=0. if zero else (self._flow_sample(self.ret_size,t,0.,1.))
            ret=0. if zero else (-1 if kind==0 else 1)*math.exp(z_ret*RET_SCALE+RET_LOC)
            t=t+self.return_cond(one(float(zero),z_ret))
            event.update(z=z,base_amount=math.exp(value),ret=ret)
        else:
            spacing=meta['tick_spacing'];sign=1 if meta['base_is_token0'] else -1
            anchor=raw.new_tensor([sign*math.floor(state['tick'])/spacing]);unit=raw.new_tensor([spacing*LOG_TICK])
            minimum=-((-MIN_TICK)//spacing);maximum=MAX_TICK//spacing
            lower=self.sample_grid(self.lower(q),anchor,unit,minimum,maximum-1)
            lo=raw.new_tensor([lower]);q=self.range_hidden(q,lo,anchor,unit)
            width=self.sample_grid(self.width(q),anchor,unit,1,maximum-lower,width=True)
            q=q+self.range_cond(torch.stack([torch.asinh((lo-anchor)*unit/.001),(raw.new_tensor([width])*unit).log()],-1))
            z=self._flow_sample(self.mint_amount if kind==2 else self.burn_amount,q,float(self.mark_mean[1]),float(self.mark_std[1]))
            if not math.isfinite(z) or z+state['log_liquidity']>=128*math.log(2):return dict(event,error='LP amount outside uint128')
            t=q+self.transition_cond(one(0.,(z-float(self.mark_mean[1]))/float(self.mark_std[1]),0.))+self.return_cond(one(0.,0.))
            event.update(z=z,lower=lower,width=width,anchor=float(anchor),ret=0.)
        zero=bool(torch.rand((),device=device)<self.liq_zero(t).sigmoid().squeeze())
        event['liquidity_change']=0. if zero else LIQ_UNIT*math.sinh(self._flow_sample(self.liq_size,t,0.,1.))
        return event


def learned_features(event,meta,mean,std,seconds_per_block=12.):
    """History features of a learned-transition event, built as events.event_features builds real ones."""
    f=np.zeros(12);m=np.ones(12,bool);m[[2,3,4,5,8,9]]=False
    f[0]=math.log1p(round(seconds_per_block*event['dt']));f[1]=event['ret'];f[6]=math.log1p(event['dt']);f[7]=event['dt']==0
    f[9]=event['liquidity_change'];f[10]=f[11]=1.;m[9]=True
    if event['kind']<2:f[2]=event['z'];m[2]=True
    else:
        spacing=meta['tick_spacing'];distance=(event['lower']-event['anchor'])*spacing;span=event['width']*spacing
        f[3]=event['z'];f[4]=math.asinh(distance*LOG_TICK/.001);f[5]=math.log(span*LOG_TICK)
        # Same boundary convention as the native-tick test lower <= tick < upper.
        f[8]=(distance<=0<distance+span) if meta['base_is_token0'] else (distance<0<=distance+span)
        m[[3,4,5,8]]=True
    scaled=np.where(m,(f-mean)/std,0).astype(np.float32)
    if not np.isfinite(scaled).all():raise ValueError('Nonfinite feedback features')
    return scaled,m


def advance(state,event,meta):
    """Apply a learned event to the tracked state (base log price, log L, tick, log reserve)."""
    sign=1 if meta['base_is_token0'] else -1
    state['log_price']+=event['ret'];state['log_liquidity']+=event['liquidity_change']
    state['tick']+=sign*event['ret']/LOG_TICK
    # Virtual base reserve L / sqrt(P_base) in either orientation.
    state['log_reserve']+=event['liquidity_change']-.5*event['ret']
    if state['log_liquidity']<0:raise ValueError('Zero active liquidity')
    if not MIN_TICK<=state['tick']<MAX_TICK:raise ValueError('Price outside the tick range')
