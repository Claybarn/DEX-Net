"""Shared per-event representation for observed and simulated trades/states."""
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn
from .tick_mechanics import PoolState
from .model import apply_pooled_event,native_range
from .rollout import sampled_event
from .diagnostics import feedback,endpoint

NAMES=['sell','buy','mint','burn','log_gap_blocks','swap_amount0','swap_amount1','liquidity_delta',
       'range_lower','range_upper','pre_price','post_price','pre_liquidity','post_liquidity',
       'pre_tick','post_tick','pre_liquidity_present','post_liquidity_present']


def event_token(row,initial,meta):
    """Invertible numeric transforms of individual fields, no temporal averaging.

    Amounts normalized to origin virtual reserves; LP amounts to origin active L.
    LP token settlement amounts are omitted identically in both classes because
    the generator models liquidity delta/range, not LP fee-collection transfers.
    """
    initial=initial if isinstance(initial,dict) else vars(initial)
    kind=int(row['kind']);lp=kind>=2;sign=1 if meta['base_is_token0'] else -1
    logL=math.log(initial['liquidity']);logs=math.log(initial['sqrt']);logq=math.log(2**96)
    def amount(value,logscale):return math.asinh(int(value)*math.exp(-logscale))
    onehot=[float(kind==k) for k in range(4)]
    fields=[math.log1p(row['dt']),amount(row.get('amount0',0),logL+logq-logs) if not lp else 0.,
            amount(row.get('amount1',0),logL+logs-logq) if not lp else 0.,
            amount(row.get('amount',0)*(1 if kind==2 else -1),logL) if lp else 0.]
    fields += [math.asinh((int(row.get(k,0))-initial['tick'])*math.log1p(.0001)/.001) if lp else 0. for k in ['lower','upper']]
    fields += [sign*2*(math.log(int(row[k]))-logs) for k in ['sqrt_before','sqrt_after']]
    fields += [math.log1p(int(row[k]))-math.log1p(initial['liquidity']) for k in ['liquidity_before','liquidity_after']]
    fields += [math.asinh((int(row[k])-initial['tick'])/meta['tick_spacing']) for k in ['tick_before','tick_after']]
    fields += [float(int(row[k])>0) for k in ['liquidity_before','liquidity_after']]
    result=np.array(onehot+fields,dtype=np.float32)
    if row['dt']<0 or not np.isfinite(result).all():raise ValueError('Invalid raw event token')
    return result


def transition(event,previous,state,detail,meta):
    row=dict(kind=event['kind'],dt=event['dt'],sqrt_before=previous.sqrt,sqrt_after=state.sqrt,
        liquidity_before=previous.liquidity,liquidity_after=state.liquidity,tick_before=previous.tick,tick_after=state.tick,
        amount0=detail.get('amount0',0),amount1=detail.get('amount1',0),amount=event.get('amount',0))
    if event['kind']>=2:row['lower'],row['upper']=native_range(event['lower'],event['upper'],meta)
    return row


@torch.no_grad()
def sample_traces(model,b,record,book,mean,std,paths=16,steps=32):
    if model.training:raise ValueError('Generator must have dropout disabled')
    initial=PoolState(**record['state']);meta=record['meta'];states=[deepcopy(initial) for _ in range(paths)]
    inventories=[{int(lo):{int(hi):v for hi,v in ups.items()} for lo,ups in record['inventory'].items()} for _ in range(paths)]
    books=[book.fork() for _ in range(paths)];histories={k:v.repeat_interleave(paths,0) for k,v in b.items()}
    records=[[] for _ in range(paths)];failures=[None]*paths;blocks=[0]*paths;volumes=[0]*paths
    for step in range(steps):
        h=model.encode(histories['x'],histories['history_kind'],histories['history_mask'],histories['metadata'])
        features=[];masks=[];kinds=[]
        for i in range(paths):
            previous=deepcopy(states[i]);row=None
            if failures[i] is None:
                event,_=sampled_event(model,h[i:i+1],states[i],inventories[i],meta)
                try:
                    if 'error' in event:raise ValueError(event['error'])
                    available=inventories[i].get(event.get('lower'),{}).get(event.get('upper'),0)
                    detail=apply_pooled_event(states[i],books[i],inventories[i],event,meta)
                    if states[i].liquidity<=0:raise ValueError('Zero active liquidity')
                    f,mask=feedback(previous,states[i],event,meta,mean,std,available,seconds_per_block=record.get('seconds_per_block',12.))
                    row=transition(event,previous,states[i],detail,meta)
                    event_token(row,initial,meta)
                    blocks[i]+=event['dt'];volumes[i]+=abs(event['base_amount']) if event['kind']<2 else 0
                    kind=event['kind']
                except (ValueError,OverflowError) as exc:
                    failures[i]=dict(step=step+1,error=str(exc));states[i]=previous
            if row is None:
                # Fixed-length absorbing continuation, with failure recorded in
                # the audit only. No synthetic-only mask is fed to the classifier.
                event=dict(kind=int(histories['history_kind'][i,-1]),dt=0,amount=0,lower=0,upper=1)
                row=transition(event,states[i],states[i],{},meta)
                f=histories['x'][i,-1].cpu().numpy();mask=histories['history_mask'][i,-1].cpu().numpy();kind=event['kind']
            records[i].append(row);features.append(f);masks.append(mask);kinds.append(kind)
        histories['x']=torch.cat([histories['x'][:,1:],torch.as_tensor(np.array(features),device=h.device)[:,None]],1)
        histories['history_mask']=torch.cat([histories['history_mask'][:,1:],torch.as_tensor(np.array(masks),device=h.device)[:,None]],1)
        histories['history_kind']=torch.cat([histories['history_kind'][:,1:],torch.tensor(kinds,device=h.device)[:,None]],1)
    tokens=np.array([[event_token(row,initial,meta) for row in rows] for rows in records],dtype=np.float32)
    endpoints=[endpoint(state,initial,dt,vol,meta) for state,dt,vol in zip(states,blocks,volumes)]
    return tokens,dict(events=records,failures=failures,endpoints=endpoints)


class SequenceDiscriminator(nn.Module):
    def __init__(self,features=len(NAMES),metadata=7,hidden=48):
        super().__init__();self.project=nn.Linear(features+1,hidden)
        self.encoder=nn.GRU(hidden,hidden,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.Linear(2*hidden+metadata,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,history,future,meta,mode='conditional'):
        if mode=='history':future=future[:,:0]
        if mode=='future':history=history[:,:0]
        past=torch.cat([history,torch.zeros_like(history[:,:,:1])],-1)
        next_=torch.cat([future,torch.ones_like(future[:,:,:1])],-1)
        x=self.project(torch.cat([past,next_],1));_,last=self.encoder(x)
        return self.head(torch.cat([last[-2],last[-1],meta],-1)).squeeze(-1)
