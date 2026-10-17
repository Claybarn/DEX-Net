"""Dataset loading, batching and next-event evaluation shared by training and analysis."""
import json,math
from pathlib import Path
import numpy as np
import torch
from .training_data import PooledTrainingDataset,pooled_collate


def load_sets(plan,split):
    manifest=json.loads((Path(plan['data'])/'dataset.json').read_text())
    return {name:PooledTrainingDataset(Path(plan['data']),Path(plan['context']),name,split,depth=plan.get('depth'))
            for name,m in manifest['pools'].items() if m['held_out']==split.startswith('heldout')}


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
            h=model.encode(b['x'],b['history_kind'],b['history_mask'],b['metadata'],b.get('pool_id'));q=model.condition(h,b['dt'].float())
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
