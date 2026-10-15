"""Sample event paths from frozen models at prepared anchors and score them.

For every anchor (scripts.prepare_anchors) and model, draws --paths independent
paths through exact pool mechanics. Each path stores its endpoint at every horizon
and a compact per-event trace (kind, block gap, cumulative return in bp, base volume)
for stylized-fact analysis. Invalid sampled events end the path and are recorded,
never repaired. Scores: unbiased ensemble energy and CRPS, bias, 90% coverage.

  python -m scripts.rollout_eval --anchors eval/anchors-test --out eval/rollouts-test \
      --model ref=runs/ref/training/best.pt --paths 32 --workers 8
"""
import argparse,json,os,time,zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from dexnet.model import PooledEventModel,apply_pooled_event,native_range
from dexnet.diagnostics import ENDPOINT_NAMES,ENDPOINT_SCALES,proper_scores,feedback,endpoint
from dexnet.tick_mechanics import PoolState,TickBook
from dexnet.utils import dump


def load_model(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    m=PooledEventModel(ck['scalers'],**ck['model_config']).to(device).eval();m.load_state_dict(ck['model'])
    return m


def make_book(inventory,meta):
    b=TickBook(meta['tick_spacing'],lambda word:0,lambda tick:(0,0))
    for lo,ups in inventory.items():
        for hi,L in ups.items():
            a,z=native_range(lo,hi,meta);b.change(a,L,L);b.change(z,L,-L)
    return b


@torch.no_grad()
def trajectory(model,b,initial,inventory,book,meta,mean,std,horizons,seed,device):
    torch.manual_seed(seed);state=deepcopy(initial);inv=deepcopy(inventory);book=book.fork();b={k:v.clone() for k,v in b.items()}
    values={};blocks=0;volume=0;trace=dict(kind=[],dt=[],return_bp=[],base_volume=[])
    for step in range(1,max(horizons)+1):
        try:
            e=model.sample_event(b,state,inv,meta);previous=deepcopy(state)
            available=inv.get(e.get('lower'),{}).get(e.get('upper'),0)
            apply_pooled_event(state,book,inv,e,meta);blocks+=e['dt'];size=abs(e['base_amount']) if e['kind']<2 else 0;volume+=size
            f,mask=feedback(previous,state,e,meta,mean,std,available)
            b['x']=torch.cat([b['x'][:,1:],torch.as_tensor(f,device=device)[None,None]],1)
            b['history_mask']=torch.cat([b['history_mask'][:,1:],torch.as_tensor(mask,device=device)[None,None]],1)
            b['history_kind']=torch.cat([b['history_kind'][:,1:],torch.tensor([[e['kind']]],device=device)],1)
            v=endpoint(state,initial,blocks,volume,meta)
            if not np.isfinite(v).all():raise ValueError('Nonfinite outcome')
            trace['kind'].append(int(e['kind']));trace['dt'].append(int(e['dt']));trace['return_bp'].append(float(v[0]));trace['base_volume'].append(float(size))
            if step in horizons:values[str(step)]=v
        except (ValueError,OverflowError,RuntimeError) as exc:return dict(values=values,trace=trace,failure=dict(step=step,error=str(exc)))
    return dict(values=values,trace=trace,failure=None)


def run_anchor(task):
    """One anchor, all models; runs in a worker process."""
    torch.set_num_threads(1)
    path,models,paths,horizons,data,out,device,traces=task
    r=json.loads(Path(path).read_text());pool=r['pool'];row=r['row'];dest=Path(out)/'anchors'/f'{pool}-{row}.json'
    if dest.exists():return str(dest)
    meanstd=json.loads((Path(data)/'scalers.json').read_text())['scalers']['raw_features'];mean=np.array(meanstd['mean']);std=np.array(meanstd['std'])
    folder=Path(data)/'pools'/pool;arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['features','feature_mask','kind']}
    meta=r['meta'];initial=PoolState(**r['state']);inv={int(lo):{int(hi):L for hi,L in ups.items()} for lo,ups in r['inventory'].items()};book=make_book(inv,meta)
    b=dict(x=torch.tensor(np.asarray(arrays['features'][row-64:row]),device=device)[None],
           history_mask=torch.tensor(np.asarray(arrays['feature_mask'][row-64:row]),device=device)[None],
           history_kind=torch.tensor(np.asarray(arrays['kind'][row-64:row]),device=device,dtype=torch.long)[None],
           metadata=torch.tensor(meta['metadata_features'],device=device,dtype=torch.float32)[None])
    record=dict(pool=pool,row=row,split=r['split'],actual=r['actual'],paths={})
    for no,(label,checkpoint) in enumerate(models.items()):
        model=load_model(checkpoint,device)
        sampled=[trajectory(model,b,initial,inv,book,meta,mean,std,horizons,
                            int(np.random.SeedSequence([row,j,no,zlib.crc32(pool.encode())]).generate_state(1)[0]),device) for j in range(paths)]
        if not traces:
            for s in sampled:s.pop('trace')
        record['paths'][label]=sampled
    dump(dest,record);return str(dest)


def summarize(records,labels,horizons):
    result={}
    for split in sorted({r['split'] for r in records}):
        result[split]={}
        for h in horizons:
            population=[r for r in records if r['split']==split]
            common=[r for r in population if all(all(str(h) in p['values'] for p in r['paths'][l]) for l in labels)]
            cell=dict(anchors=len(population),common_complete_anchors=len(common),models={})
            for label in labels:
                failures=Counter(p['failure']['error'] for r in population for p in r['paths'][label] if str(h) not in p['values'] and p['failure'])
                completed=sum(str(h) in p['values'] for r in population for p in r['paths'][label]);scores=[];crps=[];errors=[];coverage=[]
                for r in common:
                    x=np.array([p['values'][str(h)] for p in r['paths'][label]]);y=np.array(r['actual'][str(h)])
                    es,c=proper_scores(x,y);scores.append(es);crps.append(c);errors.append(np.median(x,0)-y)
                    lo,hi=np.quantile(x,[.05,.95],axis=0);coverage.append((y>=lo)&(y<=hi))
                v=dict(completed_paths=completed,total_paths=sum(len(r['paths'][label]) for r in population),failures=dict(failures))
                if scores:v.update(mean_energy=float(np.mean(scores)),mean_crps=np.mean(crps,0).tolist(),median_bias=np.mean(errors,0).tolist(),
                                   coverage_90=np.mean(coverage,0).tolist())
                cell['models'][label]=v
            result[split][str(h)]=cell
    return dict(endpoint_names=list(ENDPOINT_NAMES),energy_scales=ENDPOINT_SCALES.tolist(),results=result)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--anchors',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,help='prepared data folder; default: read from the first checkpoint run protocol')
    p.add_argument('--model',action='append',required=True,help='label=path/to/best.pt (repeatable)')
    p.add_argument('--paths',type=int,default=16);p.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 2)//2))
    p.add_argument('--device',default='cpu');p.add_argument('--no-traces',action='store_true')
    a=p.parse_args()
    models=dict(m.split('=',1) for m in a.model)
    anchor_protocol=json.loads((a.anchors/'protocol.json').read_text());horizons=tuple(anchor_protocol['horizons'])
    data=a.data or Path(json.loads((Path(next(iter(models.values()))).parent/'protocol.json').read_text())['data'])
    (a.out/'anchors').mkdir(parents=True,exist_ok=True)
    dump(a.out/'protocol.json',dict(models=models,paths=a.paths,horizons=list(horizons),anchors=str(a.anchors),
        scores='Unbiased ensemble energy and CRPS; median bias; central 90% interval coverage. Failed paths retained and counted.',
        clock='Sampled block gaps; feature timestamps use 12 s per block.'))
    files=sorted(f for d in a.anchors.iterdir() if d.is_dir() for f in d.glob('*.json') if f.stem.isdigit())
    tasks=[(str(f),models,a.paths,horizons,str(data),str(a.out),a.device,not a.no_traces) for f in files]
    began=time.monotonic();done=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(run_anchor,t) for t in tasks]):
            f.result();done+=1
            if done%10==0 or done==len(tasks):
                el=time.monotonic()-began;print(json.dumps(dict(anchors_complete=done,anchors_total=len(tasks),seconds=round(el),eta_seconds=round(el/done*(len(tasks)-done)))),flush=True)
    records=[json.loads(f.read_text()) for f in sorted((a.out/'anchors').glob('*.json'))]
    dump(a.out/'summary.json',summarize(records,list(models),horizons))
    dump(a.out/'COMPLETE.json',dict(anchors=len(records),models=list(models)))


if __name__=='__main__':main()
