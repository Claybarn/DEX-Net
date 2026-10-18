"""Just-in-time (JIT) liquidity: simulated versus real, with identical definitions.

Simulated sequences start at test anchors from scripts.prepare_anchors (per-pool subset,
evenly spaced) and run for --horizon events with a frozen model; real sequences are
the full test-period event streams of the same pools. Both are reduced to the same
event records (raw token units) and measured identically:

  large mint      minted liquidity > 10x the active liquidity before it
  strict JIT pair consecutive Mint, Swap, Burn in one block with the same range and
                  the same liquidity amount
  swap/deposit    quote notional of the paired swap / quote value of the minted
                  position at the mint price
  pair swap size  the paired swap's quote notional relative to the median swap of the
                  same pool and source
  removal         for large mints with 64 later events observed: share whose
                  same-range burns return >= 99% of the minted liquidity within
                  64 events (not owner attribution)

These follow the definitions of the original JIT study in the research code.

  python -m scripts.jit_analysis --anchors eval/anchors-test --data data/prepared \
      --model depth=runs/depth/training/best.pt --model reference=runs/ref/training/best.pt \
      --out results/jit.json
"""
import argparse,json,math,os,zlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from dexnet.model import apply_pooled_event,native_range
from dexnet.diagnostics import feedback
from dexnet.rollout_context import liquidity_profile
from dexnet.tick_mechanics import PoolState,Q96,sqrt_at_tick,delta0,delta1
from scripts.rollout_eval import load_model,make_book

FOLLOW=64
NAMES=('Swap','Swap','Mint','Burn')


# ---------------------------------------------------------------- common measures
def mint_amounts(L,sqrt,lo,hi):
    a,b=sqrt_at_tick(lo),sqrt_at_tick(hi)
    if sqrt<=a:return delta0(a,b,L,True),0
    if sqrt>=b:return 0,delta1(a,b,L,True)
    return delta0(sqrt,b,L,True),delta1(a,sqrt,L,True)


def quote_value(a0,a1,sqrt,base0):
    """Value in raw quote units at the given price (base token converted at the pool price)."""
    p=(sqrt/Q96)**2  # token1 per token0, raw units
    return a0*p+a1 if base0 else a1/p+a0


def measure(rows,base0):
    """Per-sequence counts and pair/removal records from event records."""
    out=dict(events=len(rows),mints=0,large=0,swaps=[],pairs=[],removal=[])
    for i,r in enumerate(rows):
        if r['event']=='Swap':out['swaps'].append(abs(r['amount1'] if base0 else r['amount0']))
        if r['event']!='Mint':continue
        out['mints']+=1;large=r['amount']>10*r['liquidity_before']>0;out['large']+=int(large)
        if i+2<len(rows):
            s,b=rows[i+1],rows[i+2]
            if s['event']=='Swap' and b['event']=='Burn' and r['block']==s['block']==b['block'] and (r['lower'],r['upper'],r['amount'])==(b['lower'],b['upper'],b['amount']):
                a0,a1=mint_amounts(r['amount'],r['sqrt_before'],r['lower'],r['upper']);deposit=quote_value(a0,a1,r['sqrt_before'],base0)
                swap=abs(s['amount1'] if base0 else s['amount0'])
                out['pairs'].append(dict(swap=swap,swap_deposit=swap/deposit if deposit else None,large=bool(large)))
        if large and i+FOLLOW<len(rows):
            burned=0;recovered=None
            for step,x in enumerate(rows[i+1:i+1+FOLLOW],1):
                if x['event']=='Burn' and (x['lower'],x['upper'])==(r['lower'],r['upper']):
                    burned+=x['amount']
                    if burned>=.99*r['amount']:recovered=step;break
            out['removal'].append(recovered)
    return out


def summarize(by_pool):
    """Aggregate {pool: [sequence measures]}; swap sizes are scaled by each pool's own median swap (token units differ)."""
    parts=[x for ps in by_pool.values() for x in ps]
    events=sum(p['events'] for p in parts);mints=sum(p['mints'] for p in parts);large=sum(p['large'] for p in parts)
    rel=[]
    for ps in by_pool.values():
        swaps=[v for p in ps for v in p['swaps']]
        if not swaps:continue
        med=float(np.median(swaps)) or 1.;rel+=[q['swap']/med for p in ps for q in p['pairs']]
    pairs=[q for p in parts for q in p['pairs']];removal=[x for p in parts for x in p['removal']]
    sd=np.array([q['swap_deposit'] for q in pairs if q['swap_deposit']]);rel=np.array(rel)
    rec=[x for x in removal if x is not None]
    return dict(events=events,mints=mints,large_mints=large,large_share_of_mints=large/mints if mints else None,
                pairs=len(pairs),pairs_per_1000_events=1e3*len(pairs)/events if events else None,
                large_mints_paired=sum(q['large'] for q in pairs),
                swap_deposit=dict(n=len(sd),**{f'q{q}':float(np.quantile(sd,q)) for q in (.1,.5,.9)}) if len(sd) else None,
                pair_swap_over_pool_median=dict(n=len(rel),**{f'q{q}':float(np.quantile(rel,q)) for q in (.1,.5,.9)}) if len(rel) else None,
                removal=dict(large_mints_followed=len(removal),recovered_within_64=len(rec)/len(removal) if removal else None,
                             unremoved_after_64=1-len(rec)/len(removal) if removal else None,median_events_to_recovery=float(np.median(rec)) if rec else None))


# ---------------------------------------------------------------- simulation
@torch.no_grad()
def sim_path(model,b,initial,inventory,book,meta,mean,std,horizon,seed):
    torch.manual_seed(seed);state=deepcopy(initial);inv=deepcopy(inventory);book=book.fork();b={k:v.clone() for k,v in b.items()}
    rows=[];block=0;base0=meta['base_is_token0']
    for _ in range(horizon):
        try:
            if model.liquidity_context:b['depth']=torch.as_tensor(liquidity_profile(state,book,meta))[None]
            e=model.sample_event(b,state,inv,meta);previous=deepcopy(state);block+=int(e['dt'])
            available=inv.get(e.get('lower'),{}).get(e.get('upper'),0)
            detail=apply_pooled_event(state,book,inv,e,meta)
            r=dict(event=NAMES[e['kind']],block=block,liquidity_before=previous.liquidity,sqrt_before=previous.sqrt,amount=0,amount0=0,amount1=0,lower=0,upper=0)
            if e['kind']<2:r.update(amount0=detail['amount0'],amount1=detail['amount1'])
            else:
                lo,hi=native_range(e['lower'],e['upper'],meta);r.update(lower=lo,upper=hi,amount=e['amount'])
            rows.append(r)
            f,mask=feedback(previous,state,e,meta,mean,std,available)
            b['x']=torch.cat([b['x'][:,1:],torch.as_tensor(f)[None,None]],1)
            b['history_mask']=torch.cat([b['history_mask'][:,1:],torch.as_tensor(mask)[None,None]],1)
            b['history_kind']=torch.cat([b['history_kind'][:,1:],torch.tensor([[e['kind']]])],1)
        except (ValueError,OverflowError,RuntimeError):break
    return measure(rows,base0)


def sim_anchor(task):
    torch.set_num_threads(1)
    path,label,checkpoint,paths,horizon,data=task;r=json.loads(Path(path).read_text());pool=r['pool'];row=r['row']
    ms=json.loads((Path(data)/'scalers.json').read_text())['scalers']['raw_features'];mean=np.array(ms['mean']);std=np.array(ms['std'])
    folder=Path(data)/'pools'/pool;arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['features','feature_mask','kind']}
    meta=r['meta'];initial=PoolState(**r['state']);inv={int(lo):{int(hi):L for hi,L in ups.items()} for lo,ups in r['inventory'].items()};book=make_book(inv,meta)
    b=dict(x=torch.tensor(np.asarray(arrays['features'][row-64:row]))[None],history_mask=torch.tensor(np.asarray(arrays['feature_mask'][row-64:row]))[None],
           history_kind=torch.tensor(np.asarray(arrays['kind'][row-64:row]),dtype=torch.long)[None],metadata=torch.tensor(meta['metadata_features'],dtype=torch.float32)[None])
    model=load_model(checkpoint,'cpu')
    parts=[sim_path(model,b,initial,inv,book,meta,mean,std,horizon,zlib.crc32(f'{pool}-{row}-{j}'.encode())) for j in range(paths)]
    return label,pool,r['split'],parts


# ---------------------------------------------------------------- real
def real_pool(task):
    data,pool=task;folder=Path(data)/'pools'/pool;meta=json.loads((folder/'metadata.json').read_text())
    split='heldout_test' if json.loads((Path(data)/'dataset.json').read_text())['pools'][pool]['held_out'] else 'test'
    # The test period is one block range for every pool (exposure.json names it 'test' for held-out pools too).
    exposure=next(x for x in json.loads((folder/'exposure.json').read_text()) if x['split']=='test');first,last=exposure['first_block'],exposure['last_block']
    rows=[]
    for f in sorted((folder/'decoded').glob('*.parquet')):
        pf=pq.ParquetFile(f);col=pf.schema_arrow.names.index('block')
        groups=[g for g in range(pf.num_row_groups) if pf.metadata.row_group(g).column(col).statistics.max>=first and pf.metadata.row_group(g).column(col).statistics.min<=last]
        for g in groups:
            for r in pf.read_row_group(g,columns=['event','block','lower','upper','amount','amount0','amount1','sqrt_before','liquidity_before']).to_pylist():
                if not first<=r['block']<=last or r['event']=='Initialize':continue
                for k in ('amount','amount0','amount1','sqrt_before','liquidity_before'):r[k]=int(r[k] or 0)
                if r['event'] in ('Mint','Burn') and not r['amount']:continue
                rows.append(r)
    base0=meta['base_is_token0']
    return pool,split,measure(rows,base0)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--anchors',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--model',action='append',required=True,help='label=path/to/best.pt (repeatable)')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--per-pool',type=int,default=12);p.add_argument('--paths',type=int,default=32)
    p.add_argument('--horizon',type=int,default=96);p.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 2)//2))
    a=p.parse_args();models=dict(m.split('=',1) for m in a.model)
    files=[]
    for d in sorted(x for x in a.anchors.iterdir() if x.is_dir()):
        fs=sorted((f for f in d.glob('*.json') if f.stem.isdigit()),key=lambda f:int(f.stem));step=max(1,len(fs)//a.per_pool);files+=fs[::step][:a.per_pool]
    pools=sorted({f.parent.name for f in files});collected={}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures=[ex.submit(real_pool,(str(a.data),pool)) for pool in pools]
        futures+=[ex.submit(sim_anchor,(str(f),label,path,a.paths,a.horizon,str(a.data))) for label,path in models.items() for f in files]
        for i,fut in enumerate(as_completed(futures),1):
            res=fut.result()
            if len(res)==3:pool,split,part=res;label='real';parts=[part]
            else:label,pool,split,parts=res
            group='heldout' if split.startswith('heldout') else 'seen'
            collected.setdefault(label,{}).setdefault(group,{}).setdefault(pool,[]).extend(parts)
            if i%20==0:print(f'{i}/{len(futures)}',flush=True)
    result=dict(method=__doc__,anchors=len(files),paths=a.paths,horizon=a.horizon,models=models,groups={})
    for label,groups in collected.items():
        for group,by_pool in groups.items():
            result['groups'].setdefault(group,{})[label]=dict(all=summarize(by_pool),by_pool={pool:summarize({pool:parts}) for pool,parts in by_pool.items()})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for group,v in result['groups'].items():
        print(f'\n== {group}')
        for label,s in v.items():
            s=s['all'];sd=s['swap_deposit'] or {};rel=s['pair_swap_over_pool_median'] or {};rm=s['removal']
            print(f"{label:10s} events {s['events']:>9,}  large/mints {s['large_share_of_mints'] or 0:.3f}  pairs/1k {s['pairs_per_1000_events'] or 0:.2f}  "
                  f"swap/deposit p50 {sd.get('q0.5',float('nan')):.4f}  pair swap/pool median p50 {rel.get('q0.5',float('nan')):.1f}  "
                  f"large unremoved@64 {rm['unremoved_after_64'] if rm['unremoved_after_64'] is not None else float('nan'):.3f} (n={rm['large_mints_followed']})")


if __name__=='__main__':main()
