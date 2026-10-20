"""Dataset loading, batching and next-event evaluation shared by training and analysis."""
import json,math
from pathlib import Path
import numpy as np
import torch
from .training_data import PooledTrainingDataset,pooled_collate


def load_sets(plan,split):
    """Pools of a split. A run limited to some pools (plan['pools']) trains and selects on them only;
    its test splits keep every pool, in manifest order, so evaluation rows match the pooled runs."""
    manifest=json.loads((Path(plan['data'])/'dataset.json').read_text())
    keep=set(plan['pools']) if plan.get('pools') and split in ('train','val') else None
    return {name:PooledTrainingDataset(Path(plan['data']),Path(plan['context']),name,split,depth=plan.get('depth'))
            for name,m in manifest['pools'].items() if m['held_out']==split.startswith('heldout') and (keep is None or name in keep)}


def host_batch(ds,indices):
    """One batch from one pool, built on the CPU (safe to run in loader worker processes)."""
    b=pooled_collate([ds[int(i)] for i in indices])
    if hasattr(ds,'pool_id'):b['pool_id']=torch.full((len(indices),),ds.pool_id,dtype=torch.long)
    visible=getattr(ds,'visible_history',None)
    if visible is not None:
        # History ablation: hide all but the last `visible` events (features, masks and kinds).
        hidden=b['x'].shape[1]-int(visible)
        if hidden>0:
            b['x'][:,:hidden]=0;b['history_mask'][:,:hidden]=0;b['history_kind'][:,:hidden]=0
    return b


def device_batch(ds,indices,device):
    return {k:v.to(device) for k,v in host_batch(ds,indices).items()}


CANDIDATES=(('lower_candidates','lower_mask'),('width_candidates','width_mask'))


def merge_batches(batches):
    """Concatenate batches row-wise (rows keep their order, so burn rows keep their candidates).

    Burn candidate tables are re-padded to a common width with the same filler (1)
    and a False mask, exactly as pooled_collate pads within a batch.
    """
    out={}
    for key in batches[0]:
        if key in {c for pair in CANDIDATES for c in pair}:continue
        out[key]=torch.cat([b[key] for b in batches])
    for values,mask in CANDIDATES:
        width=max(b[values].shape[1] for b in batches)
        pad=lambda t,fill:torch.nn.functional.pad(t,(0,width-t.shape[1]),value=fill)
        out[values]=torch.cat([pad(b[values],1.) for b in batches]);out[mask]=torch.cat([pad(b[mask].to(torch.uint8),0) for b in batches]).bool()
    return out


@torch.no_grad()
def evaluate(model,sets,limit,device):
    model.eval();result={}
    for no,(name,ds) in enumerate(sets.items()):
        rows=np.sort(np.random.default_rng(74401+no).choice(len(ds),min(limit,len(ds)),replace=False))
        totals={};counts={};correct=0;tail_n=0;tail_pred=np.zeros(5);tail_actual=np.zeros(5)
        for start in range(0,len(rows),256):
            b=device_batch(ds,rows[start:start+256],device);loss,parts=model.losses(b)
            if not torch.isfinite(loss):raise ValueError('Nonfinite evaluation loss')
            for key,(v,n) in parts.items():totals[key]=totals.get(key,0)+float(v);counts[key]=counts.get(key,0)+int(n)
            h=model.encode_batch(b);q=model.condition(h,b['dt'].float())
            correct+=int((model.types(q).argmax(-1)==b['kind']).sum())
            swap=(b['kind']<2)&b['target_mask'][:,0]
            if swap.any():
                q=model.condition(h[swap],b['dt'][swap].float(),b['kind'][swap]);raw=model.swap_size(q).double()
                actual=b['targets'][swap,0];buy=b['kind'][swap]==1;cap=b['buy_log_cap'][swap]
                for j,threshold in enumerate([1e-4,1e-3,.01,.1,1.]):
                    logcdf=model.flow_cdf(raw,raw.new_full((len(raw),),math.log(threshold)))
                    if buy.any():logcdf[buy]-=model.flow_cdf(raw[buy],cap[buy])
                    probability=-torch.expm1(logcdf.clamp(max=0));tail_pred[j]+=float(probability.sum());tail_actual[j]+=int((actual>math.log(threshold)).sum())
                tail_n+=len(raw)
        result[name]=dict(examples=len(rows),loss={k:totals[k]/max(counts[k],1) for k in totals},counts=counts,
                          type_accuracy=correct/max(len(rows),1),swap_tail_examples=tail_n,
                          virtual_base_ratio_thresholds=[1e-4,1e-3,.01,.1,1.],predicted_exceedance_counts=tail_pred.tolist(),observed_exceedance_counts=tail_actual.tolist())
    return result


def expand_candidates(b,lower_width=1280,width_width=128):
    """Burn candidate tables as fixed-width full-row tables (for losses_static).

    pooled_collate stores one candidate row per burn, in row order. This places
    each burn's candidates on its own batch row, padded to a fixed width; other
    rows are empty (losses_static gives them a placeholder). Raises if a burn has
    more candidates than the fixed width, so callers can fall back to losses().
    """
    n=len(b['kind']);burns=torch.nonzero(b['kind']==3).flatten()
    for compact,mask,full,size in (('lower_candidates','lower_mask','lower_candidates_full',lower_width),
                                   ('width_candidates','width_mask','width_candidates_full',width_width)):
        used=int(b[mask].sum(-1).max()) if len(burns) else 0
        if used>size:raise ValueError(f'{compact}: {used} candidates exceed fixed width {size}')
        values=torch.ones((n,size),dtype=b[compact].dtype);m=torch.zeros((n,size),dtype=torch.bool)
        k=min(b[compact].shape[1],size)
        values[burns,:k]=b[compact][:,:k];m[burns,:k]=b[mask][:,:k]
        b[full]=values;b[full.replace('candidates','mask')]=m
    return b


class _EvalChunks(torch.utils.data.Dataset):
    """Fixed evaluation rows of one pool in chunks, built on the CPU (worker-safe)."""
    def __init__(self,ds,rows,size):
        self.ds,self.rows,self.size=ds,rows,size
    def __len__(self):return (len(self.rows)+self.size-1)//self.size
    def __getitem__(self,i):
        return expand_candidates(host_batch(self.ds,self.rows[i*self.size:(i+1)*self.size]))


@torch.no_grad()
def evaluate_fast(model,sets,limit,device,batch=1024,workers=4):
    """evaluate() with the same rows, outputs and formulas, without per-batch synchronisation.

    Batches are built by loader workers and scored with losses_static; type accuracy
    and swap-size tail counts are masked sums kept on the device and read once per pool.
    """
    model.eval();result={}
    thresholds=[1e-4,1e-3,.01,.1,1.]
    for no,(name,ds) in enumerate(sets.items()):
        rows=np.sort(np.random.default_rng(74401+no).choice(len(ds),min(limit,len(ds)),replace=False))
        loader=torch.utils.data.DataLoader(_EvalChunks(ds,rows,batch),batch_size=None,num_workers=workers,pin_memory=True)
        totals={};counts={};correct=0;tail_n=0;tail_pred=torch.zeros(5,dtype=torch.float64,device=device);tail_actual=torch.zeros(5,dtype=torch.float64,device=device)
        bad=torch.zeros((),dtype=torch.bool,device=device)
        for cpu in loader:
            b={k:v.to(device,non_blocking=True) for k,v in cpu.items()};loss,parts=model.losses_static(b)
            bad=bad|~torch.isfinite(loss)|(parts['violations'][0]>0)
            for key,(v,n) in parts.items():
                if key=='violations':continue
                totals[key]=totals.get(key,0)+v;counts[key]=counts.get(key,0)+n
            h=model.encode_batch(b);q=model.condition(h,b['dt'].float())
            correct=correct+(model.types(q).argmax(-1)==b['kind']).sum()
            swap=(b['kind']<2)&b['target_mask'][:,0];buy=swap&(b['kind']==1)
            q=model.condition(h,b['dt'].float(),b['kind']);raw=model.swap_size(q).double()
            actual=b['targets'][:,0];cap=torch.where(buy,b['buy_log_cap'].double(),torch.zeros_like(b['buy_log_cap'].double()))
            capcdf=model.flow_cdf(raw,cap,static=True)
            for j,threshold in enumerate(thresholds):
                logcdf=model.flow_cdf(raw,raw.new_full((len(raw),),math.log(threshold)),static=True)
                logcdf=torch.where(buy,logcdf-capcdf,logcdf)
                probability=-torch.expm1(logcdf.clamp(max=0))
                tail_pred[j]+=torch.where(swap,probability,torch.zeros_like(probability)).sum()
                tail_actual[j]+=(swap&(actual>math.log(threshold))).sum()
            tail_n=tail_n+swap.sum()
        if bool(bad):raise ValueError('Nonfinite evaluation loss or physical-support violation')
        counts={k:int(v) for k,v in counts.items()}
        result[name]=dict(examples=len(rows),loss={k:float(totals[k])/max(counts[k],1) for k in totals},counts=counts,
                          type_accuracy=int(correct)/max(len(rows),1),swap_tail_examples=int(tail_n),
                          virtual_base_ratio_thresholds=thresholds,predicted_exceedance_counts=tail_pred.tolist(),observed_exceedance_counts=tail_actual.tolist())
    return result
