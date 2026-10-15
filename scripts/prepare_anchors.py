"""Reconstruct exact pool state and liquidity inventory at evaluation start points ("anchors").

Anchors are spaced uniformly in event index within one split, never selected by
outcome, and their outcome windows (up to the longest horizon) do not overlap and
stay inside the split. For each anchor the full raw log is replayed to recover the
exact pool state and per-range inventory; every real feature row inside the windows
is re-derived and checked against the prepared dataset; and the realised endpoint
(price return, liquidity change, blocks, volume) is recorded at each horizon.

  python -m scripts.prepare_anchors --data data/prepared --out eval/anchors-test --split test \
      --anchors 64 --horizons 32,128,512
"""
import argparse,json,time
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
                active[modelrow]=dict(initial=state,volume=0)
            ledger.apply(r)
            if matched:
                amount=abs(r['amount0'] if meta['base_is_token0'] else r['amount1']) if r['event']=='Swap' else 0
                for v in active.values():v['volume']+=amount
                if active:
                    previous=(int(arrays['block'][modelrow-1]),int(arrays['timestamp'][modelrow-1]));raw,mask,*_=event_features(r,before,ledger,meta,previous,available)
                    expected=np.where(mask,(raw-mean)/std,0).astype(np.float32)
                    if not np.allclose(expected,arrays['features'][modelrow],rtol=2e-5,atol=2e-5):raise ValueError('Real feature feedback mismatch')
                    if not np.array_equal(mask,arrays['feature_mask'][modelrow]):raise ValueError('Real feature mask mismatch')
                    checked+=1
                if modelrow in ends:
                    row,h=ends[modelrow];v=active[row];record=records[row]
                    record['actual'][h]=endpoint(PoolState(ledger.sqrt,ledger.tick,ledger.L),v['initial'],r['block']-record['initial_block'],v['volume'],meta)
                    if h==max(horizons):
                        record['final_timestamp']=int(arrays['timestamp'][modelrow])
                        dump(dest/(str(row)+'.json'),record);active.pop(row)
                modelrow+=1
                if modelrow>max(ends):break
        dump(dest/'progress.json',dict(rows=modelrow,target=max(ends),seconds=time.monotonic()-began))
        if modelrow>max(ends):break
    if len(records)!=len(picks) or any(len(r['actual'])!=len(horizons) for r in records.values()):raise ValueError('Incomplete anchor outcomes')
    result=dict(pool=name,split=split,anchors=len(records),teacher_features_checked=checked,seconds=time.monotonic()-began)
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
