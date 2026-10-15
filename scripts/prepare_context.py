"""Causal pre-event buy capacities and active burn support for pooled training."""
import argparse,json,math,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pyarrow.parquet as pq
from dexnet.pool_mechanics import CapacityLedger
from dexnet.utils import dump


def prepare(data,out,name):
    folder=Path(data)/'pools'/name;dest=Path(out)/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return json.loads((dest/'COMPLETE.json').read_text())
    meta=json.loads((folder/'metadata.json').read_text());base0=meta['base_is_token0'];spacing=meta['tick_spacing'];sign=1 if base0 else -1
    blocks=np.load(folder/'block.npy',mmap_mode='r');indices=np.load(folder/'log_index.npy',mmap_mode='r');kinds=np.load(folder/'kind.npy',mmap_mode='r');targets=np.load(folder/'targets.npy',mmap_mode='r');masks=np.load(folder/'target_mask.npy',mmap_mode='r')
    n=len(blocks);anchor=np.lib.format.open_memmap(dest/'anchor.npy',mode='w+',dtype='float64',shape=(n,))
    cap=np.lib.format.open_memmap(dest/'buy_log_cap.npy',mode='w+',dtype='float64',shape=(n,));cap[:]=0
    burn=np.lib.format.open_memmap(dest/'burn_id.npy',mode='w+',dtype='int32',shape=(n,));burn[:]=-1
    inv={};ledger=CapacityLedger(spacing);row=0;versions=[];lo_offsets=[0];width_offsets=[0];previous_lower=None;version=-1
    buys=0;min_margin=math.inf;start=time.time()
    try:
        with (dest/'lower_values.bin').open('wb') as lowerfile,(dest/'width_values.bin').open('wb') as widthfile:
            for file in sorted((folder/'decoded').glob('*.parquet')):
                records=pq.read_table(file,columns=['event','block','log_index','owner','lower','upper','amount','amount0','amount1','sqrt','tick','liquidity']).to_pylist()
                for r in records:
                    for k in ['amount','amount0','amount1','sqrt','liquidity']:r[k]=int(r[k])
                    is_model=row<n and (r['block'],r['log_index'])==(int(blocks[row]),int(indices[row]))
                    if is_model:
                        anchor[row]=sign*ledger.tick/spacing
                        if kinds[row]==1 and masks[row,0]:
                            native=ledger.capacity(base0)
                            if native<=0:raise ValueError('Observed buy has zero causal capacity')
                            cap[row]=np.nextafter(math.log(native+1)-ledger.log_reserve(base0),math.inf)
                            observed=-(r['amount0'] if base0 else r['amount1'])
                            if observed>native:raise ValueError(f'Observed buy exceeds exact native capacity {name} row {row}: {observed}>{native}')
                            margin=cap[row]-targets[row,0];min_margin=min(min_margin,float(margin));buys+=1
                            if margin<=0:raise ValueError(f'Observed buy outside capacity {name} row {row}')
                        if kinds[row]==3:
                            lower,upper=(r['lower']//spacing,r['upper']//spacing) if base0 else (-r['upper']//spacing,-r['lower']//spacing)
                            keys=tuple(sorted(inv))
                            if keys!=previous_lower:
                                np.asarray(keys,np.int32).tofile(lowerfile);lo_offsets.append(lo_offsets[-1]+len(keys));version+=1;previous_lower=keys
                            widths=sorted(u-lower for u in inv[lower]);np.asarray(widths,np.int32).tofile(widthfile)
                            width_offsets.append(width_offsets[-1]+len(widths));versions.append(version);burn[row]=len(versions)-1
                            assert upper-lower in widths
                        row+=1
                    if r['event'] in ['Mint','Burn'] and r['amount']:
                        lower,upper=(r['lower']//spacing,r['upper']//spacing) if base0 else (-r['upper']//spacing,-r['lower']//spacing)
                        level=inv.setdefault(lower,{});v=level.get(upper,0)+(r['amount'] if r['event']=='Mint' else -r['amount'])
                        if v:level[upper]=v
                        else:level.pop(upper,None)
                        if not level:inv.pop(lower,None)
                    ledger.apply(r)
                dump(dest/'progress.json',dict(rows=row,total=n,seconds=time.time()-start))
        assert row==n
        for k,v in [('lower_offsets',lo_offsets),('width_offsets',width_offsets),('lower_versions',versions)]:np.save(dest/(k+'.npy'),np.asarray(v,np.int64))
        for v in [anchor,cap,burn]:v.flush()
        result=dict(pool=name,rows=n,buys_checked=buys,minimum_buy_margin=min_margin,burns=len(versions),lower_values=lo_offsets[-1],width_values=width_offsets[-1],seconds=time.time()-start)
        dump(dest/'COMPLETE.json',result);return result
    except Exception as exc:dump(dest/'FAILED.json',dict(error=repr(exc),row=row));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert (a.data/'VERIFIED.json').exists();a.out.mkdir(parents=True,exist_ok=True)
    names=list(json.loads((a.data/'dataset.json').read_text())['pools']);results={}
    with ProcessPoolExecutor(max_workers=2) as workers:
        futures={workers.submit(prepare,str(a.data),str(a.out),n):n for n in names}
        for f in as_completed(futures):
            results[futures[f]]=f.result();dump(a.out/'progress.json',dict(completed=list(results)))
    dump(a.out/'COMPLETE.json',results)


if __name__=='__main__':main()
