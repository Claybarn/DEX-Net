"""Likelihood-ratio gradients through exact, nondifferentiable pooled trajectories.

Every sampled latent is detached before evaluating its conditional log probability.
Summed conditional scores credit early choices through all downstream mechanics.
No pathwise or straight-through gradients are used. Dropout must be disabled.
"""
import math
from copy import deepcopy
from decimal import Decimal
import numpy as np
import torch
import torch.nn.functional as F
from .model import discrete_nll,capacity_native,log_reserve,apply_pooled_event
from .tick_mechanics import MIN_TICK,MAX_TICK,PoolState
from .diagnostics import feedback,endpoint
from .scores import energy_surrogate


def clock_log_prob(raw,dt):
    rates=(F.softplus(raw[:,4:])+.00001).clamp(max=20.)
    gap=torch.logsumexp(raw[:,1:4].log_softmax(-1)+torch.log(-torch.expm1(-rates))-(dt[:,None]-1)*rates,-1)
    return torch.where(dt==0,-F.softplus(-raw[:,0]),-F.softplus(raw[:,0])+gap)


def draw_flow(model,raw,index,cap=None):
    """Score pre-rounding continuous latent, including buy truncation normalizer."""
    if cap is not None:raw=raw.double()
    with torch.no_grad():
        u=torch.randn(raw.shape[:-1],device=raw.device,dtype=raw.dtype)
        if cap is not None:
            fc=model.flow_cdf(raw,cap,index).exp()
            prob=torch.minimum(torch.special.ndtr(u)*fc,torch.nextafter(fc,torch.zeros_like(fc)))
            if bool((prob<=0).any()):raise ValueError('Supported CDF underflow')
            u=torch.special.ndtri(prob)
        z=model.flow.inverse(raw,u)*model.mark_std[index]+model.mark_mean[index]
    score=-model.amount_nll(raw,z,index,cap)
    return z,score.sum()


def sampled_event(model,h,state,inventory,meta,forced_dt=None,capacity_fn=None,trace=None):
    if forced_dt is not None and torch.is_grad_enabled():raise ValueError('Forced waits are inference-only')
    raw=model.clock(h)
    with torch.no_grad():
        if forced_dt is not None:dt=int(forced_dt)
        elif bool(torch.rand((),device=h.device)<raw[0,0].sigmoid()):dt=0
        else:
            component=int(torch.multinomial(raw[0,1:4].softmax(0),1))
            rate=float((F.softplus(raw[0,4+component])+.00001).clamp(max=20.))
            dt=1+int(math.floor(-math.log(max(float(torch.rand((),device=h.device)),1e-12))/rate))
    d=raw.new_tensor([dt]);score=clock_log_prob(raw,d).sum()
    logits=model.types(model.condition(h,d))
    if not inventory:logits=logits.masked_fill(torch.tensor([[True,True,False,True]],device=h.device),-torch.inf)
    kind=int(torch.multinomial(logits.detach().softmax(-1)[0],1))
    score=score+logits.log_softmax(-1)[0,kind]
    q=model.condition(h,d,torch.tensor([kind],device=h.device));event=dict(kind=kind,dt=dt)
    if kind<2:
        reserve=log_reserve(state,meta['base_is_token0']);cap=(capacity_fn(state) if capacity_fn is not None else capacity_native(state,inventory,meta)) if kind==1 else None
        if cap is not None and cap<1:return dict(event,error='No physical buy capacity'),score
        logcap=math.log(cap+1)-reserve if cap is not None else None
        z,lp=draw_flow(model,model.swap_size(q),0,None if cap is None else q.new_tensor([logcap],dtype=torch.float64));score=score+lp
        if trace is not None:trace.update(z=float(z),logcap=logcap)
        value=float(z)
        if kind==1:
            amount=max(1,min(cap,int(Decimal(cap+1)*Decimal.from_float(min(1.,math.exp(value-logcap))))))
        else:
            value+=reserve
            if not math.isfinite(value) or value>170:return dict(event,error='Excessive base size'),score
            amount=max(1,round(math.exp(value)))
        return dict(event,base_amount=amount if kind==0 else -amount),score
    spacing=meta['tick_spacing'];sign=1 if meta['base_is_token0'] else -1
    anchor=raw.new_tensor([sign*state.tick/spacing]);unit=raw.new_tensor([spacing*math.log1p(.0001)])
    minimum=-((-MIN_TICK)//spacing);maximum=MAX_TICK//spacing
    def grid(r,first,last,width=False,candidates=None):
        value=model.sample_grid(r,anchor,unit,first,last,width=width,candidates=candidates)
        support={} if candidates is None else dict(candidates=r.new_tensor([candidates]),mask=torch.ones((1,len(candidates)),device=r.device,dtype=torch.bool))
        lp=-discrete_nll(r,r.new_tensor([value]),anchor,unit,r.new_tensor([first]),r.new_tensor([last]),width=width,**support)
        return value,lp.sum()
    lower,lp=grid(model.lower(q),minimum,maximum-1,candidates=sorted(inventory) if kind==3 else None);score=score+lp
    lo=raw.new_tensor([lower]);q=model.range_hidden(q,lo,anchor,unit)
    width,lp=grid(model.width(q),1,maximum-lower,width=True,candidates=sorted(u-lower for u in inventory[lower]) if kind==3 else None);score=score+lp
    if trace is not None:trace.update(lower=lower,width=width)
    q=q+model.range_cond(torch.stack([torch.asinh((lo-anchor)*unit/.001),(raw.new_tensor([width])*unit).log()],-1))
    if kind==2:
        z,lp=draw_flow(model,model.mint_amount(q),1);score=score+lp;value=float(z)+math.log(state.liquidity)
        if trace is not None:trace['z']=float(z)
        if not math.isfinite(value) or value>=128*math.log(2):return dict(event,error='Mint outside uint128'),score
        amount=max(1,round(math.exp(value)))
    else:
        available=inventory[lower][lower+width];q=q+model.available(raw.new_tensor([[math.log(available/state.liquidity)]]))
        logits=model.burn_full(q).squeeze();full=torch.rand((),device=h.device)<logits.detach().sigmoid()
        if trace is not None:trace['full']=bool(full)
        score=score-F.binary_cross_entropy_with_logits(logits,full.to(logits.dtype))
        if bool(full):amount=available
        else:
            z,lp=draw_flow(model,model.burn_fraction(q),2);score=score+lp
            if trace is not None:trace['z']=float(z)
            amount=max(1,min(available,round(available*float(z.double().sigmoid()))))
    return dict(event,lower=lower,upper=lower+width,amount=amount),score


def rollout_loss(model,b,record,book,mean,std,scales,paths=4,steps=8):
    if model.training:raise ValueError('Rollout probability model requires dropout disabled')
    device=b['x'].device;initial=PoolState(**record['state']);meta=record['meta']
    states=[deepcopy(initial) for _ in range(paths)]
    inv0={int(lo):{int(hi):v for hi,v in ups.items()} for lo,ups in record['inventory'].items()}
    inventories=[deepcopy(inv0) for _ in range(paths)];books=[book.fork() for _ in range(paths)]
    histories={k:v.repeat_interleave(paths,0) for k,v in b.items()}
    logps=[b['x'].new_zeros(()) for _ in range(paths)];values=[None]*paths;failures=[None]*paths
    blocks=[0]*paths;volumes=[0]*paths;checkpoints={}
    for step in range(steps):
        h=model.encode(histories['x'],histories['history_kind'],histories['history_mask'],histories['metadata'])
        features=[];masks=[];kinds=[]
        for i in range(paths):
            if failures[i] is not None:
                features.append(histories['x'][i,-1].cpu().numpy());masks.append(histories['history_mask'][i,-1].cpu().numpy());kinds.append(int(histories['history_kind'][i,-1]));continue
            state=states[i];previous=deepcopy(state)
            event,lp=sampled_event(model,h[i:i+1],state,inventories[i],meta);logps[i]=logps[i]+lp
            try:
                if 'error' in event:raise ValueError(event['error'])
                available=inventories[i].get(event.get('lower'),{}).get(event.get('upper'),0)
                apply_pooled_event(state,books[i],inventories[i],event,meta)
                if state.liquidity<=0:raise ValueError('Zero active liquidity')
                f,mask=feedback(previous,state,event,meta,mean,std,available,seconds_per_block=record.get('seconds_per_block',12.))
                blocks[i]+=event['dt'];volumes[i]+=abs(event['base_amount']) if event['kind']<2 else 0
                values[i]=endpoint(state,initial,blocks[i],volumes[i],meta)
                if not np.isfinite(values[i]).all():raise ValueError('Nonfinite endpoint')
            except (ValueError,OverflowError) as exc:
                failures[i]=dict(step=step+1,error=str(exc));states[i]=previous
                f=histories['x'][i,-1].cpu().numpy();mask=histories['history_mask'][i,-1].cpu().numpy()
                if values[i] is None:values[i]=endpoint(previous,initial,blocks[i],volumes[i],meta)
            features.append(f);masks.append(mask);kinds.append(event['kind'])
        histories['x']=torch.cat([histories['x'][:,1:],torch.as_tensor(np.array(features),device=device)[:,None]],1)
        histories['history_mask']=torch.cat([histories['history_mask'][:,1:],torch.as_tensor(np.array(masks),device=device)[:,None]],1)
        histories['history_kind']=torch.cat([histories['history_kind'][:,1:],torch.tensor(kinds,device=device)[:,None]],1)
        if step+1 in [1,8,32]:checkpoints[str(step+1)]=[list(v)+[10. if failure else 0.] for v,failure in zip(values,failures)]
    # Failed paths remain an absorbing outcome with an explicit failure coordinate.
    x=np.array([list(v)+[10. if failure else 0.] for v,failure in zip(values,failures)])
    scale=np.r_[scales,1.];y=np.r_[record['actual'][str(steps)],0.]
    loss,score=energy_surrogate(torch.tensor(x/scale,device=device,dtype=torch.float64)[None],torch.tensor(y/scale,device=device,dtype=torch.float64)[None],torch.stack(logps)[None])
    return loss,dict(energy=float(score),failures=failures,endpoints=x.tolist(),checkpoints=checkpoints)
