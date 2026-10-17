"""Where do oversized simulated moves come from? Swap sizes and impacts, simulated vs real.

For anchors from scripts.prepare_anchors, samples paths with a frozen model and, for
every swap, records its size relative to the pre-swap virtual base reserve
(log ratio, the model's own size coordinate), its price move in bp, the active
liquidity before it, and the ticks it crossed. The real swaps after the same anchors
give the reference: sizes from the prepared targets, moves from the anchor's
realised trace.

Two questions:
  drift   does the simulated size distribution move away from the real one as the
          path gets longer? (quantiles by event index: 1-8, 9-64, 65-256, 257-512)
  impact  for swaps of the same relative size, are simulated price moves larger?
          (|move| quantiles within size bins)

  python -m scripts.swap_diagnostics --anchors eval/anchors-test --data data/prepared \
      --model runs/ref/training/best.pt --out results/swap-diagnostics.json
"""
import argparse,json,math,os,zlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from dexnet.model import apply_pooled_event,log_reserve
from dexnet.diagnostics import feedback
from dexnet.rollout_context import liquidity_profile
from dexnet.tick_mechanics import PoolState
from scripts.rollout_eval import load_model,make_book

BUCKETS=((1,8),(9,64),(65,256),(257,512))
SIZE_BINS=(-np.inf,-11,-9,-7,-5,np.inf)
QUANTILES=(.5,.9,.99,.999)


@torch.no_grad()
def diag_path(model,b,initial,inventory,book,meta,mean,std,horizon,seed,device):
    torch.manual_seed(seed);state=deepcopy(initial);inv=deepcopy(inventory);book=book.fork();b={k:v.clone() for k,v in b.items()}
    rows=[];base0=meta['base_is_token0'];sign=1 if base0 else -1
    for step in range(1,horizon+1):
        try:
            if model.liquidity_context:b['depth']=torch.as_tensor(liquidity_profile(state,book,meta),device=device)[None]
            e=model.sample_event(b,state,inv,meta);previous=deepcopy(state)
            available=inv.get(e.get('lower'),{}).get(e.get('upper'),0)
            ratio=math.log(abs(e['base_amount']))-log_reserve(previous,base0) if e['kind']<2 else None
            apply_pooled_event(state,book,inv,e,meta)
            f,mask=feedback(previous,state,e,meta,mean,std,available)
            b['x']=torch.cat([b['x'][:,1:],torch.as_tensor(f,device=device)[None,None]],1)
            b['history_mask']=torch.cat([b['history_mask'][:,1:],torch.as_tensor(mask,device=device)[None,None]],1)
            b['history_kind']=torch.cat([b['history_kind'][:,1:],torch.tensor([[e['kind']]],device=device)],1)
            if ratio is not None:
                rows.append((step,ratio,sign*2*math.log(state.sqrt/previous.sqrt)*1e4,math.log(previous.liquidity),abs(state.tick-previous.tick)//meta['tick_spacing']))
        except (ValueError,OverflowError,RuntimeError):break
    return rows


def run_anchor(task):
    torch.set_num_threads(1)
    path,checkpoint,paths,horizon,data=task;r=json.loads(Path(path).read_text());pool=r['pool'];row=r['row']
    meanstd=json.loads((Path(data)/'scalers.json').read_text())['scalers']['raw_features'];mean=np.array(meanstd['mean']);std=np.array(meanstd['std'])
    folder=Path(data)/'pools'/pool;arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['features','feature_mask','kind','targets','target_mask']}
    meta=r['meta'];initial=PoolState(**r['state']);inv={int(lo):{int(hi):L for hi,L in ups.items()} for lo,ups in r['inventory'].items()};book=make_book(inv,meta)
    b=dict(x=torch.tensor(np.asarray(arrays['features'][row-64:row]))[None],history_mask=torch.tensor(np.asarray(arrays['feature_mask'][row-64:row]))[None],
           history_kind=torch.tensor(np.asarray(arrays['kind'][row-64:row]),dtype=torch.long)[None],metadata=torch.tensor(meta['metadata_features'],dtype=torch.float32)[None])
    model=load_model(checkpoint,'cpu');sim=[]
    for j in range(paths):
        sim.extend(diag_path(model,b,initial,inv,book,meta,mean,std,horizon,zlib.crc32(f'{pool}-{row}-{j}'.encode()),'cpu'))
    t=r['actual_trace'];kinds=np.asarray(t['kind'][:horizon]);ret=np.diff(np.r_[0.,np.asarray(t['return_bp'][:horizon])])
    tg=np.asarray(arrays['targets'][row:row+horizon,0]);tm=np.asarray(arrays['target_mask'][row:row+horizon,0])
    real=[(i+1,float(tg[i]),float(ret[i]),None,None) for i in range(len(kinds)) if kinds[i]<2 and tm[i]]
    return pool,r['split'],sim,real


def summarize(rows):
    rows=np.array([(s,z,abs(m)) for s,z,m,*_ in rows],float)
    if not len(rows):return {}
    out=dict(swaps=len(rows),drift={},impact={})
    for lo,hi in BUCKETS:
        z=rows[(rows[:,0]>=lo)&(rows[:,0]<=hi),1]
        if len(z)>20:out['drift'][f'{lo}-{hi}']=dict(n=len(z),**{f'q{q}':float(np.quantile(z,q)) for q in QUANTILES})
    for a,b in zip(SIZE_BINS[:-1],SIZE_BINS[1:]):
        m=rows[(rows[:,1]>a)&(rows[:,1]<=b),2]
        if len(m)>20:out['impact'][f'({a},{b}]']=dict(n=len(m),share=len(m)/len(rows),**{f'abs_move_q{q}':float(np.quantile(m,q)) for q in QUANTILES})
    for x in (50,100,500):out[f'share_abs_move_gt_{x}bp']=float(np.mean(rows[:,2]>x))
    big=rows[rows[:,2]>100]
    out['moves_gt_100bp']=dict(n=len(big),median_size_log_ratio=float(np.median(big[:,1])) if len(big) else None)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--anchors',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--per-pool',type=int,default=12);p.add_argument('--paths',type=int,default=32)
    p.add_argument('--horizon',type=int,default=512);p.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 2)//2))
    a=p.parse_args()
    files=[]
    for d in sorted(x for x in a.anchors.iterdir() if x.is_dir()):
        fs=sorted((f for f in d.glob('*.json') if f.stem.isdigit()),key=lambda f:int(f.stem));step=max(1,len(fs)//a.per_pool)
        files+=fs[::step][:a.per_pool]
    groups={}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(run_anchor,(str(x),a.model,a.paths,a.horizon,str(a.data))) for x in files]):
            pool,split,sim,real=f.result();g='heldout' if split.startswith('heldout') else 'seen'
            groups.setdefault(g,dict(sim=[],real=[]));groups[g]['sim']+=sim;groups[g]['real']+=real
    result=dict(method=__doc__,anchors=len(files),model=a.model,groups={g:dict(sim=summarize(v['sim']),real=summarize(v['real'])) for g,v in groups.items()})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for g,v in result['groups'].items():
        print(f'\n== {g} pools ({len(files)} anchors total)')
        print('size log-ratio quantiles by event index      p50     p90     p99   p99.9')
        for k in v['real'].get('drift',{}):
            for who in ('real','sim'):
                q=v[who]['drift'].get(k)
                if q:print(f"  {k:8s} {who:5s} n={q['n']:7d}  "+'  '.join(f"{q['q'+str(x)]:6.2f}" for x in QUANTILES))
        print('|move| bp quantiles within size bins           share     p50     p90     p99   p99.9')
        for k in v['real'].get('impact',{}):
            for who in ('real','sim'):
                q=v[who]['impact'].get(k)
                if q:print(f"  {k:12s} {who:5s} n={q['n']:7d}  {q['share']:.3f} "+'  '.join(f"{q['abs_move_q'+str(x)]:7.2f}" for x in QUANTILES))
        for who in ('real','sim'):
            print(f"  {who}: share |move|>50bp {v[who]['share_abs_move_gt_50bp']:.4f}  >100bp {v[who]['share_abs_move_gt_100bp']:.5f}  >500bp {v[who]['share_abs_move_gt_500bp']:.6f}  median size of >100bp moves {v[who]['moves_gt_100bp']['median_size_log_ratio']}")


if __name__=='__main__':main()
