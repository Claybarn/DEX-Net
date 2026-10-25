"""What follows a large mint, simulated against real: the diagnostic behind JIT stickiness.

Uses the event records and settings of scripts.jit_analysis (test anchors, 12 per pool,
32 paths of 96 events per model; real: the full test-period streams). For every large mint
(minted liquidity > 10x the active liquidity before it) with at least --follow later
events it records:

  next          type of the next event and whether it is in the same block
  removal step  events until burns of the same range return >= 99% of the minted
                liquidity (None if not within --follow)
  first burn    for the first same-range burn: events after the mint and the share of the
                minted liquidity it removes

  python -m scripts.jit_followup --anchors eval/anchors-test --data data/prepared \
      --model depth=runs/depth/training/best.pt --out results/jit-followup.json
"""
import argparse,json,os,zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from dexnet.model import apply_pooled_event,native_range
from dexnet.diagnostics import feedback
from dexnet.rollout_context import liquidity_profile
from dexnet.tick_mechanics import PoolState
from scripts.rollout_eval import load_model,make_book
from scripts.jit_analysis import NAMES

FOLLOW=16


def follow(rows,steps=FOLLOW,base0=True):
    out=[]
    for i,r in enumerate(rows):
        if r['event']!='Mint' or not r['amount']>10*r['liquidity_before']>0 or i+steps>=len(rows):continue
        nxt=rows[i+1];burned=0;removed=None;first=None
        for k,x in enumerate(rows[i+1:i+1+steps],1):
            if x['event']=='Burn' and (x['lower'],x['upper'])==(r['lower'],r['upper']):
                if first is None:first=dict(step=k,share=x['amount']/r['amount'],same_block=x['block']==r['block'])
                burned+=x['amount']
                if burned>=.99*r['amount']:removed=k;break
        two=rows[i+2];same=two['event']=='Burn' and (two['lower'],two['upper'])==(r['lower'],r['upper'])
        step2=f"{two['event']}{' same range' if same else (' other range' if two['event']=='Burn' else '')}, {'same' if two['block']==r['block'] else 'later'} block"
        size=abs(nxt.get('amount1' if base0 else 'amount0',0)) if nxt['event']=='Swap' else None
        out.append(dict(next=nxt['event'],next_same_block=nxt['block']==r['block'],removed=removed,first_burn=first,step2=step2,
                        swap_over_liquidity=size/r['liquidity_before'] if size and r['liquidity_before'] else None))
    return out


@torch.no_grad()
def sim_rows(model,b,initial,inventory,book,meta,mean,std,horizon,seed):
    torch.manual_seed(seed);state=deepcopy(initial);inv=deepcopy(inventory);book=book.fork();b={k:v.clone() for k,v in b.items()};rows=[];block=0
    for _ in range(horizon):
        try:
            if model.liquidity_context:b['depth']=torch.as_tensor(liquidity_profile(state,book,meta))[None]
            e=model.sample_event(b,state,inv,meta);previous=deepcopy(state);block+=int(e['dt'])
            available=inv.get(e.get('lower'),{}).get(e.get('upper'),0);detail=apply_pooled_event(state,book,inv,e,meta)
            r=dict(event=NAMES[e['kind']],block=block,liquidity_before=previous.liquidity,amount=0,lower=0,upper=0,amount0=0,amount1=0)
            if e['kind']<2:r.update(amount0=detail['amount0'],amount1=detail['amount1'])
            if e['kind']>=2:
                lo,hi=native_range(e['lower'],e['upper'],meta);r.update(lower=lo,upper=hi,amount=e['amount'])
            rows.append(r);f,mask=feedback(previous,state,e,meta,mean,std,available)
            b['x']=torch.cat([b['x'][:,1:],torch.as_tensor(f)[None,None]],1)
            b['history_mask']=torch.cat([b['history_mask'][:,1:],torch.as_tensor(mask)[None,None]],1)
            b['history_kind']=torch.cat([b['history_kind'][:,1:],torch.tensor([[e['kind']]])],1)
        except (ValueError,OverflowError,RuntimeError):break
    return rows


def sim_anchor(task):
    torch.set_num_threads(1)
    path,label,checkpoint,paths,horizon,data=task;r=json.loads(Path(path).read_text());pool=r['pool'];row=r['row']
    ms=json.loads((Path(data)/'scalers.json').read_text())['scalers']['raw_features'];mean=np.array(ms['mean']);std=np.array(ms['std'])
    folder=Path(data)/'pools'/pool;arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['features','feature_mask','kind']}
    meta=r['meta'];initial=PoolState(**r['state']);inv={int(lo):{int(hi):L for hi,L in ups.items()} for lo,ups in r['inventory'].items()};book=make_book(inv,meta)
    b=dict(x=torch.tensor(np.asarray(arrays['features'][row-64:row]))[None],history_mask=torch.tensor(np.asarray(arrays['feature_mask'][row-64:row]))[None],
           history_kind=torch.tensor(np.asarray(arrays['kind'][row-64:row]),dtype=torch.long)[None],metadata=torch.tensor(meta['metadata_features'],dtype=torch.float32)[None])
    model=load_model(checkpoint,'cpu');records=[]
    for j in range(paths):records+=follow(sim_rows(model,b,initial,inv,book,meta,mean,std,horizon,zlib.crc32(f'{pool}-{row}-{j}'.encode())),base0=meta['base_is_token0'])
    return label,'heldout' if r['split'].startswith('heldout') else 'seen',records


def real_pool(task):
    data,pool=task;folder=Path(data)/'pools'/pool
    held=json.loads((Path(data)/'dataset.json').read_text())['pools'][pool]['held_out']
    exposure=next(x for x in json.loads((folder/'exposure.json').read_text()) if x['split']=='test');first,last=exposure['first_block'],exposure['last_block']
    rows=[]
    for f in sorted((folder/'decoded').glob('*.parquet')):
        pf=pq.ParquetFile(f);col=pf.schema_arrow.names.index('block')
        for g in range(pf.num_row_groups):
            st=pf.metadata.row_group(g).column(col).statistics
            if st.max<first or st.min>last:continue
            for r in pf.read_row_group(g,columns=['event','block','lower','upper','amount','amount0','amount1','liquidity_before']).to_pylist():
                if not first<=r['block']<=last or r['event']=='Initialize':continue
                for k in ('amount','amount0','amount1','liquidity_before'):r[k]=int(r[k] or 0)
                if r['event'] in ('Mint','Burn') and not r['amount']:continue
                rows.append(r)
    meta=json.loads((folder/'metadata.json').read_text())
    return 'real','heldout' if held else 'seen',follow(rows,base0=meta['base_is_token0'])


def summarize(records):
    n=len(records)
    if not n:return dict(large_mints=0)
    nxt=Counter((x['next'],x['next_same_block']) for x in records);fb=[x['first_burn'] for x in records if x['first_burn']]
    removed=[x['removed'] for x in records]
    return dict(large_mints=n,next={f'{k} {"same block" if s else "later block"}':v/n for (k,s),v in sorted(nxt.items())},
                removed_within={str(k):sum(r is not None and r<=k for r in removed)/n for k in (1,2,4,8,FOLLOW)},
                first_burn_share=dict(n=len(fb),**{f'q{q}':float(np.quantile([x['share'] for x in fb],q)) for q in (.1,.5,.9)}) if fb else None,
                first_burn_full=sum(x['share']>=.99 for x in fb)/len(fb) if fb else None,
                first_burn_same_block=sum(x['same_block'] for x in fb)/len(fb) if fb else None,
                no_same_range_burn=sum(x['first_burn'] is None for x in records)/n,
                step2_when_not_removed_by_2={k:v/max(1,sum(1 for x in records if not (x['removed'] and x['removed']<=2))) for k,v in Counter(x['step2'] for x in records if not (x['removed'] and x['removed']<=2)).most_common(6)},
                swap_over_liquidity={lab:dict(**{f'q{q}':float(np.quantile(v,q)) for q in (.1,.5,.9)}) for lab,v in (('removed_by_2',[x['swap_over_liquidity'] for x in records if x['removed'] and x['removed']<=2 and x['swap_over_liquidity']]),('not_removed_by_2',[x['swap_over_liquidity'] for x in records if not (x['removed'] and x['removed']<=2) and x['swap_over_liquidity']])) if len(v)>2})


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--anchors',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--model',action='append',required=True,help='label=path (repeatable; labels ending in digits are seeds of one family)')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--per-pool',type=int,default=12);p.add_argument('--paths',type=int,default=32)
    p.add_argument('--horizon',type=int,default=96);p.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 2)//2))
    a=p.parse_args();models=dict(m.split('=',1) for m in a.model)
    files=[]
    for d in sorted(x for x in a.anchors.iterdir() if x.is_dir()):
        # Same anchors as scripts.jit_analysis.
        fs=sorted((f for f in d.glob('*.json') if f.stem.isdigit()),key=lambda f:int(f.stem));step=max(1,len(fs)//a.per_pool);files+=fs[::step][:a.per_pool]
    pools=sorted({f.parent.name for f in files});collected={}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures=[ex.submit(real_pool,(str(a.data),pool)) for pool in pools]
        futures+=[ex.submit(sim_anchor,(str(f),label,path,a.paths,a.horizon,str(a.data))) for label,path in models.items() for f in files]
        for fu in as_completed(futures):
            label,group,records=fu.result();family=label.rstrip('0123456789') or label
            collected.setdefault(group,{}).setdefault(family,[]).extend(records)
    result=dict(method=__doc__,groups={g:{fam:summarize(r) for fam,r in v.items()} for g,v in collected.items()})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for g,v in result['groups'].items():
        for fam,s in v.items():print(g,fam,json.dumps(s))


if __name__=='__main__':main()
