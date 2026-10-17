"""Causal liquidity-depth profiles for every model event (input for depth-aware training).

For each model row, the profile describes the pool's liquidity around the price
just before that event: log1p(L at offset / L now) / 4 at the 17 economic
log-price offsets in dexnet.rollout_context.PRICE_OFFSETS (-4% .. +4%). It is the
same quantity dexnet.rollout_context.liquidity_profile computes from a simulated
book during sampling, here read from the exact-integer Fenwick ledger, whose
active(t) is the liquidity of all ranges with lower <= t < upper.

  python -m scripts.prepare_depth --data data/prepared --out data/depth
"""
import argparse,json,math,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pyarrow.parquet as pq
from dexnet.events import LiquidityLedger
from dexnet.rollout_context import PRICE_OFFSETS
from dexnet.tick_mechanics import MIN_TICK,MAX_TICK,Q96
from dexnet.utils import dump

STEP=math.log1p(.0001)


def profile(ledger,base_is_token0):
    """Depth profile from the ledger's current state; None if active liquidity is zero."""
    now=ledger.active(ledger.tick)
    if now<=0:return None
    sign=1 if base_is_token0 else -1
    tick=2*(math.log(ledger.sqrt)-math.log(Q96))/STEP
    targets=np.clip(np.floor(tick+sign*PRICE_OFFSETS/STEP),MIN_TICK,MAX_TICK-1).astype(int)
    targets[len(PRICE_OFFSETS)//2]=ledger.tick
    return np.array([math.log1p(ledger.active(int(t))/now)/4. for t in targets],np.float32)


def prepare(data,out,name):
    folder=Path(data)/'pools'/name;dest=Path(out)/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return json.loads((dest/'COMPLETE.json').read_text())
    meta=json.loads((folder/'metadata.json').read_text())
    blocks=np.load(folder/'block.npy',mmap_mode='r');indices=np.load(folder/'log_index.npy',mmap_mode='r');n=len(blocks)
    depth=np.lib.format.open_memmap(dest/'depth.npy',mode='w+',dtype='float32',shape=(n,len(PRICE_OFFSETS)))
    valid=np.lib.format.open_memmap(dest/'depth_valid.npy',mode='w+',dtype='bool',shape=(n,))
    ledger=LiquidityLedger(meta['tick_spacing']);row=0;missing=0;start=time.time()
    for file in sorted((folder/'decoded').glob('*.parquet')):
        for r in pq.read_table(file,columns=['event','block','log_index','owner','lower','upper','amount','sqrt','tick','liquidity']).to_pylist():
            for k in ['amount','sqrt','liquidity']:r[k]=int(r[k])
            if row<n and (r['block'],r['log_index'])==(int(blocks[row]),int(indices[row])):
                p=profile(ledger,meta['base_is_token0']) if ledger.initialized else None
                if p is None:depth[row]=0;valid[row]=False;missing+=1
                else:depth[row]=p;valid[row]=True
                row+=1
            ledger.apply(r)
    if row!=n:raise ValueError(f'Matched {row} of {n} model rows')
    depth.flush();valid.flush()
    result=dict(pool=name,rows=n,undefined_rows=missing,seconds=time.time()-start)
    dump(dest/'COMPLETE.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=10)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    names=list(json.loads((a.data/'dataset.json').read_text())['pools']);results={}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs={ex.submit(prepare,str(a.data),str(a.out),n):n for n in names}
        for f in as_completed(fs):results[fs[f]]=f.result();print(json.dumps(results[fs[f]]),flush=True);dump(a.out/'progress.json',results)
    dump(a.out/'COMPLETE.json',dict(pools=results,offsets=PRICE_OFFSETS.tolist(),definition='log1p(L(offset)/L(now))/4, state before each event'))


if __name__=='__main__':main()
