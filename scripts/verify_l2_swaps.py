"""Completeness check for collected L2 swaps against on-chain pool state.

Only swaps change a v3 pool's price, so the last collected swap price at or before
any block must equal slot0's sqrtPriceX96 there. Checks random blocks plus the end
of the period for every pool, and that no (block, log index) appears twice.

  DEXNET_RPC_URL_BASE=<endpoint> python -m scripts.verify_l2_swaps --chain base --out data/l2
"""
import argparse,json,os
from pathlib import Path
import numpy as np,pyarrow.dataset as ds
from dexnet.rpc import jsonrpc


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--chain',choices=['arbitrum','base'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoints',type=int,default=25);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();url=os.environ['DEXNET_RPC_URL_'+a.chain.upper()];folder=a.out/a.chain
    pools=json.loads((folder/'pools.json').read_text());prog=json.loads((folder/'progress.json').read_text())
    if prog['next_block']<=prog['last_block']:raise SystemExit('Collection incomplete')
    data=ds.dataset(sorted(str(p) for p in folder.glob('part-*.parquet')),format='parquet');rng=np.random.default_rng(a.seed);report={}
    for name,v in pools.items():
        t=data.to_table(columns=['block','log_index','sqrt_after'],filter=ds.field('pool')==name)
        o=np.lexsort((t['log_index'].to_numpy(),t['block'].to_numpy()));blocks=t['block'].to_numpy()[o];logs=t['log_index'].to_numpy()[o]
        sq=np.array(t['sqrt_after'].to_pylist(),dtype=object)[o]
        duplicates=int(np.sum((np.diff(blocks)==0)&(np.diff(logs)==0)))
        checks=sorted(rng.integers(prog['first_block']+1000,prog['last_block'],a.checkpoints).tolist())+[prog['last_block']];bad=0
        for b in checks:
            i=np.searchsorted(blocks,b,side='right')-1
            onchain=int(jsonrpc(url,'eth_call',[dict(to=v['pool'],data='0x3850c7bd'),hex(b)])[2:66],16)
            bad+=int(sq[i])!=onchain
        report[name]=dict(swaps=len(blocks),duplicates=duplicates,checkpoints=len(checks),mismatches=bad)
        print(name,report[name],flush=True)
    (folder/'verification.json').write_text(json.dumps(report,indent=1)+'\n')
    if any(r['duplicates'] or r['mismatches'] for r in report.values()):raise SystemExit('Verification FAILED')


if __name__=='__main__':main()
