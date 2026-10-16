"""Reconstruct exact pool state and liquidity inventory at evaluation start points ("anchors").

Anchors are spaced uniformly in event index within one split, never selected by
outcome, and their outcome windows (up to the longest horizon) do not overlap and
stay inside the split. For each anchor the full raw log is replayed to recover the
exact pool state and per-range inventory; every real feature row inside the windows
is re-derived and checked against the prepared dataset; and the realised endpoint
(price return, liquidity change, blocks, volume) is recorded at each horizon, with
the realised per-event sequence for stylized-fact comparisons.

Two history-only baselines are stored with every anchor, as ensembles of endpoints
of past windows of the same length (computed with the same endpoint definition):
  recent     256 windows from the 32 horizons of events just before the anchor
  expanding  256 windows drawn uniformly from all of the pool's history before it

  python -m scripts.prepare_anchors --data data/prepared --out eval/anchors-test --split test \
      --anchors 64 --horizons 32,128,512
"""
import argparse,json,math,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pyarrow.parquet as pq
from dexnet.events import LiquidityLedger,event_features,eligible_indices
from dexnet.pool_mechanics import canonical_inventory
from dexnet.diagnostics import endpoint
from dexnet.tick_mechanics import PoolState
from dexnet.utils import dump


def choose(blocks,exposure,split,anchors,horizons):
    """Uniform event-index quantiles among rows whose longest window fits in the split."""
    span=[s for s in exposure if s['split']==split]
    if len(span)!=1:raise ValueError(f'No unique {split} span in exposure.json')
    rows=eligible_indices(blocks,span[0]['first_block'],span[0]['last_block']+1,horizon=max(horizons))
    if not len(rows):return rows
    n=min(anchors,len(rows)//max(horizons))          # non-overlapping windows
    if n<1:return rows[:0]
    picks=np.unique(rows[np.linspace(0,len(rows)-1,n,dtype=int)])
    assert np.all(np.diff(picks)>=max(horizons)),'Overlapping anchor windows'
    return picks


BASELINE_SAMPLES=256
RECENT_SPAN=32


def window_endpoints(post,starts,h,base_is_token0):
    """Endpoints of windows of h events starting at model rows `starts`, from post-event states.

    post[i] holds the state after model row i; a window starting at row s begins from
    post[s-1]. Mirrors dexnet.diagnostics.endpoint in floating point.
    """
    s0=starts-1;s1=starts+h-1;sign=1 if base_is_token0 else -1
    ret=sign*2*(post['logsqrt'][s1]-post['logsqrt'][s0])*1e4
    liq=post['log1pL'][s1]-post['log1pL'][s0]
    blocks=np.log1p(post['block'][s1]-post['block'][s0])
    reserve=post['logL'][s0]+((math.log(1<<96)-post['logsqrt'][s0]) if base_is_token0 else (post['logsqrt'][s0]-math.log(1<<96)))
    volume=np.log1p((post['cumvol'][s1]-post['cumvol'][s0])*np.exp(-reserve))
    out=np.stack([ret,liq,blocks,volume],-1)
    return out[np.isfinite(out).all(-1)]


def prepare(data,out,name,anchors,split,horizons):
    folder=Path(data)/'pools'/name;dest=Path(out)/name;dest.mkdir(parents=True,exist_ok=True)
    meta=json.loads((folder/'metadata.json').read_text())
    if meta['held_out']:
        if split=='train':return dict(pool=name,anchors=0,skipped='held-out pool has no training split')
        split='heldout_'+split
    arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['block','log_index','timestamp','features','feature_mask']}
    exposure=json.loads((folder/'exposure.json').read_text())
    base_split=split.replace('heldout_','')
    picks=choose(np.asarray(arrays['block']),exposure,base_split,anchors,horizons)
    if not len(picks):return dict(pool=name,anchors=0,skipped='no eligible windows')
    starts={int(v):i for i,v in enumerate(picks)};ends={int(v+h-1):(int(v),h) for v in picks for h in horizons}
    ledger=LiquidityLedger(meta['tick_spacing']);modelrow=0;records={};active={};began=time.monotonic()
    n=int(max(ends))+1;post=dict(logsqrt=np.full(n,np.nan),logL=np.full(n,np.nan),log1pL=np.full(n,np.nan),block=np.zeros(n),cumvol=np.zeros(n))
    rng=np.random.default_rng(int.from_bytes(name.encode(),'little')%2**32);kinds=np.load(folder/'kind.npy',mmap_mode='r');baseline_check=0.
    meanstd=json.loads((Path(data)/'scalers.json').read_text())['scalers']['raw_features'];mean=np.array(meanstd['mean']);std=np.array(meanstd['std']);checked=0
    for f in sorted((folder/'decoded').glob('*.parquet')):
        for r in pq.read_table(f).to_pylist():
            for k in ['amount','amount0','amount1','sqrt','liquidity']:r[k]=int(r[k])
            matched=modelrow<len(arrays['block']) and (r['block'],r['log_index'])==(int(arrays['block'][modelrow]),int(arrays['log_index'][modelrow]))
            before=(ledger.L,ledger.sqrt,ledger.tick);available=ledger.ranges.get((r['lower'],r['upper']),0)
            if matched and modelrow in starts:
                state=PoolState(ledger.sqrt,ledger.tick,ledger.L)
                if state.liquidity<=0:raise ValueError('Undefined initial active liquidity')
                records[modelrow]=dict(pool=name,row=modelrow,split=split,meta=meta,state=vars(state),inventory=canonical_inventory(ledger,meta['base_is_token0']),
                                       actual={},initial_block=int(arrays['block'][modelrow-1]),initial_timestamp=int(arrays['timestamp'][modelrow-1]))
                elapsed=int(arrays['timestamp'][modelrow-1])-int(arrays['timestamp'][modelrow-64]);blocks=int(arrays['block'][modelrow-1])-int(arrays['block'][modelrow-64])
                records[modelrow]['seconds_per_block']=elapsed/blocks if blocks else 12.
                base={'recent':{},'expanding':{}}
                for h in horizons:
                    hi=modelrow-h                                   # last start whose window ends before the anchor
                    recent=np.unique(np.linspace(max(1,modelrow-RECENT_SPAN*h),hi,BASELINE_SAMPLES).astype(int)) if hi>=1 else np.array([],int)
                    expanding=rng.integers(1,hi+1,BASELINE_SAMPLES) if hi>=1 else np.array([],int)
                    base['recent'][h]=window_endpoints(post,recent,h,meta['base_is_token0']).tolist()
                    base['expanding'][h]=window_endpoints(post,expanding,h,meta['base_is_token0']).tolist()
                records[modelrow]['baselines']=base
                records[modelrow]['actual_trace']=dict(kind=[],dt=[],return_bp=[],base_volume=[])
                active[modelrow]=dict(initial=state,volume=0,last_block=int(arrays['block'][modelrow-1]))
            ledger.apply(r)
            if matched:
                amount=abs(r['amount0'] if meta['base_is_token0'] else r['amount1']) if r['event']=='Swap' else 0
                post['logsqrt'][modelrow]=math.log(ledger.sqrt);post['logL'][modelrow]=math.log(ledger.L) if ledger.L>0 else np.nan
                post['log1pL'][modelrow]=math.log1p(ledger.L);post['block'][modelrow]=r['block']
                post['cumvol'][modelrow]=(post['cumvol'][modelrow-1] if modelrow else 0.)+amount
                for row,v in active.items():
                    v['volume']+=amount;t=records[row]['actual_trace']
                    if len(t['kind'])<max(horizons):
                        sign=1 if meta['base_is_token0'] else -1
                        t['kind'].append(int(kinds[modelrow]));t['dt'].append(int(r['block'])-v['last_block']);v['last_block']=int(r['block'])
                        t['return_bp'].append(float(sign*2*(math.log(ledger.sqrt)-math.log(v['initial'].sqrt))*1e4));t['base_volume'].append(float(amount))
                if active:
                    previous=(int(arrays['block'][modelrow-1]),int(arrays['timestamp'][modelrow-1]));raw,mask,*_=event_features(r,before,ledger,meta,previous,available)
                    expected=np.where(mask,(raw-mean)/std,0).astype(np.float32)
                    if not np.allclose(expected,arrays['features'][modelrow],rtol=2e-5,atol=2e-5):raise ValueError('Real feature feedback mismatch')
                    if not np.array_equal(mask,arrays['feature_mask'][modelrow]):raise ValueError('Real feature mask mismatch')
                    checked+=1
                if modelrow in ends:
                    row,h=ends[modelrow];v=active[row];record=records[row]
                    record['actual'][h]=endpoint(PoolState(ledger.sqrt,ledger.tick,ledger.L),v['initial'],r['block']-record['initial_block'],v['volume'],meta)
                    check=window_endpoints(post,np.array([row]),h,meta['base_is_token0'])
                    if len(check):baseline_check=max(baseline_check,float(np.max(np.abs(check[0]-np.array(record['actual'][h]))/np.array([1.,1e-3,1e-3,1e-6]))))
                    if h==max(horizons):
                        record['final_timestamp']=int(arrays['timestamp'][modelrow])
                        dump(dest/(str(row)+'.json'),record);active.pop(row)
                modelrow+=1
                if modelrow>max(ends):break
        dump(dest/'progress.json',dict(rows=modelrow,target=max(ends),seconds=time.monotonic()-began))
        if modelrow>max(ends):break
    if len(records)!=len(picks) or any(len(r['actual'])!=len(horizons) for r in records.values()):raise ValueError('Incomplete anchor outcomes')
    result=dict(pool=name,split=split,anchors=len(records),teacher_features_checked=checked,seconds=time.monotonic()-began,
                baseline_endpoint_max_scaled_error=baseline_check)
    dump(dest/'COMPLETE.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['train','val','test'],default='val',help='held-out pools use heldout_val/heldout_test')
    p.add_argument('--anchors',type=int,default=8,help='maximum anchors per pool')
    p.add_argument('--horizons',default='1,8,32',help='event horizons, comma-separated')
    p.add_argument('--pools',default='',help='comma-separated subset; default all')
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    horizons=tuple(sorted(int(h) for h in a.horizons.split(',')))
    manifest=json.loads((a.data/'dataset.json').read_text())['pools']
    names=a.pools.split(',') if a.pools else list(manifest)
    dump(a.out/'protocol.json',dict(split=a.split,anchors_per_pool=a.anchors,horizons=list(horizons),pools=names,
        selection='Uniform event-index quantiles among rows whose longest outcome window lies inside the split; non-overlapping; never outcome-selected.'))
    results={}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs={ex.submit(prepare,str(a.data),str(a.out),n,a.anchors,a.split,horizons):n for n in names}
        for f in as_completed(fs):results[fs[f]]=f.result();dump(a.out/'progress.json',results)
    dump(a.out/'COMPLETE.json',results)


if __name__=='__main__':main()
