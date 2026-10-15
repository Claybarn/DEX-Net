"""Replay real early swaps in every orientation/fee/grid before shared training."""
import argparse,json,math
from pathlib import Path
import pyarrow.parquet as pq
from dexnet.pool_mechanics import CapacityLedger,swap_economic_base,swap_token0
from dexnet.tick_mechanics import PoolState,swap_base
from dexnet.utils import dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();results={}
    for folder in sorted((a.data/'pools').iterdir()):
        meta=json.loads((folder/'metadata.json').read_text());ledger=CapacityLedger(meta['tick_spacing']);checked=0;worst=0.
        for file in sorted((folder/'decoded').glob('*.parquet')):
            for r in pq.read_table(file).to_pylist():
                for k in ['amount','amount0','amount1','sqrt','liquidity']:r[k]=int(r[k])
                amount=r['amount0'] if meta['base_is_token0'] else r['amount1']
                if r['event']=='Swap' and amount and ledger.L>0:
                    state=PoolState(ledger.sqrt,ledger.tick,ledger.L);book=ledger.book()
                    variants=[swap_token0(state,book,r['amount0'],meta['fee'])[0],swap_base(state,book,r['amount1'],meta['fee'])[0]]
                    result=min(variants,key=lambda s:abs(s.sqrt-r['sqrt']))
                    error=abs(result.sqrt-r['sqrt'])/r['sqrt'];worst=max(worst,error)
                    # Reconstructing the swap's input/output mode from net log
                    # deltas can differ by native-unit rounding at a price limit.
                    if abs(result.sqrt-r['sqrt'])>1 or result.liquidity!=r['liquidity']:raise ValueError(f'Replay mismatch {folder.name} {r["block"]}:{r["log_index"]} error={error}')
                    checked+=1
                ledger.apply(r)
                if checked>=32:break
            if checked>=32:break
        assert checked==32;results[folder.name]=dict(swaps=checked,max_relative_sqrt_error=worst,fee=meta['fee'],spacing=meta['tick_spacing'],base_is_token0=meta['base_is_token0'])
    dump(a.out,results);print(json.dumps(results))


if __name__=='__main__':main()
