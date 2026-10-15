"""Collect every Swap of the Uniswap v3 WETH/USDC pools on Arbitrum or Base (resumable).

Pools are resolved through the chain's v3 factory (fees 0.01%, 0.05%, 0.3%) and the
token contracts are checked. The pool state just before the period gives each pool's
starting price and protocol-fee setting. eth_getLogs ranges adapt to the provider's
result-size limit. Needs an archive endpoint that serves wide log ranges and includes
blockTimestamp in logs (current geth-compatible clients do), read from
DEXNET_RPC_URL_ARBITRUM or DEXNET_RPC_URL_BASE. URLs are never written to disk.

  DEXNET_RPC_URL_BASE=<endpoint> python -m scripts.collect_l2_swaps --chain base --out data/l2
"""
import argparse,datetime as dt,fcntl,json,os,re,sys,time
from pathlib import Path
import pyarrow as pa,pyarrow.parquet as pq
from dexnet.rpc import jsonrpc

SWAP='0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67'
CHAINS={
    'arbitrum':dict(chain_id=42161,factory='0x1F98431c8aD98523631AE4a59f267346ea31F984',WETH='0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',USDC='0xaf88d065e77c8cC2239327C5EDb3A432268e5831',max_span=100_000),
    'base':dict(chain_id=8453,factory='0x33128a8fC17869897dcE68Ed026d694621f6FDfD',WETH='0x4200000000000000000000000000000000000006',USDC='0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',max_span=20_000)}
FEES=(100,500,3000)
COLUMNS=['pool','block','log_index','timestamp','amount0','amount1','sqrt_after','liquidity','tick']


def word(a):return a.lower().replace('0x','').rjust(64,'0')
def signed(h):x=int(h,16);return x-(1<<256) if x>=1<<255 else x
def text(h):b=bytes.fromhex(h[2:]);n=int.from_bytes(b[32:64],'big');return b[64:64+n].decode()


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--chain',choices=list(CHAINS),required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--start',default='2025-09-01');p.add_argument('--end',default='2026-10-01',help='exclusive, UTC date')
    p.add_argument('--budget',type=int,default=125_000,help='maximum eth_getLogs calls')
    a=p.parse_args();cfg=CHAINS[a.chain]
    url=os.environ.get('DEXNET_RPC_URL_'+a.chain.upper())
    if not url:raise SystemExit('Set DEXNET_RPC_URL_'+a.chain.upper())
    call=lambda method,params:jsonrpc(url,method,params)
    if int(call('eth_chainId',[]),16)!=cfg['chain_id']:raise SystemExit('Endpoint is not '+a.chain)
    out=a.out/a.chain;out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit(a.chain+' is already being collected')
    ec=lambda to,data,block='latest':call('eth_call',[dict(to=to,data=data),block])
    if not (out/'pools.json').exists():
        for t,decimals in (('WETH',18),('USDC',6)):
            if text(ec(cfg[t],'0x95d89b41'))!=t or int(ec(cfg[t],'0x313ce567'),16)!=decimals:raise ValueError('Token check failed: '+t)
        pools={}
        for fee in FEES:
            addr='0x'+ec(cfg['factory'],'0x1698ee82'+word(cfg['WETH'])+word(cfg['USDC'])+word(hex(fee)))[-40:]
            if int(addr,16)==0:continue
            token0='0x'+ec(addr,'0x0dfe1681')[-40:]
            pools[f'{a.chain}-WETH-USDC-{fee}']=dict(chain=a.chain,pool=addr,fee=fee,token0=token0,base_is_token0=token0.lower()==cfg['WETH'].lower())
        (out/'pools.json').write_text(json.dumps(pools,indent=1))
    pools=json.loads((out/'pools.json').read_text());addr={v['pool'].lower():k for k,v in pools.items()}
    prog_path,budget_path=out/'progress.json',out/'budget.json'
    budget=json.loads(budget_path.read_text()) if budget_path.exists() else dict(calls=0)

    def block_at(ts):
        lo,hi=0,int(call('eth_blockNumber',[]),16)
        while lo<hi:
            mid=(lo+hi)//2;t=int(call('eth_getBlockByNumber',[hex(mid),False])['timestamp'],16)
            lo,hi=(mid+1,hi) if t<ts else (lo,mid)
        return lo
    if prog_path.exists():prog=json.loads(prog_path.read_text())
    else:
        utc=lambda d:int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp())
        b0,b1=block_at(utc(a.start)),block_at(utc(a.end))-1;state={}
        for x,name in addr.items():
            w=ec(x,'0x3850c7bd',hex(b0-1))[2:]
            state[name]=dict(sqrt=str(int(w[0:64],16)),fee_protocol=int(w[5*64:6*64],16))
        prog=dict(start=a.start,end=a.end,first_block=b0,last_block=b1,next_block=b0,span=2000,part=0,rows=0,start_state=state)
        prog_path.write_text(json.dumps(prog,indent=1))
    rows=[];began=time.time()

    def flush():
        nonlocal rows
        if rows:
            pq.write_table(pa.table(dict(zip(COLUMNS,zip(*rows)))),out/f"part-{prog['part']:05d}.parquet",compression='zstd')
            prog['part']+=1;prog['rows']+=len(rows);rows=[]
        prog_path.write_text(json.dumps(prog,indent=1));budget_path.write_text(json.dumps(budget))
    while prog['next_block']<=prog['last_block']:
        if budget['calls']>=a.budget:flush();sys.exit('Query budget reached; rerun with a larger --budget to resume')
        lo=prog['next_block'];hi=min(lo+prog['span']-1,prog['last_block']);budget['calls']+=1
        try:logs=call('eth_getLogs',[dict(address=list(addr),fromBlock=hex(lo),toBlock=hex(hi),topics=[SWAP])])
        except ValueError as e:
            m=re.search(r'\[(0x[0-9a-fA-F]+), (0x[0-9a-fA-F]+)\]',str(e))   # provider-suggested range
            prog['span']=max(1,int(m.group(2),16)-int(m.group(1),16)+1) if m and int(m.group(1),16)==lo else max(1,prog['span']//2)
            continue
        for r in logs:
            d=r['data'][2:]
            rows.append((addr[r['address'].lower()],int(r['blockNumber'],16),int(r['logIndex'],16),int(r['blockTimestamp'],16),
                         str(signed(d[0:64])),str(signed(d[64:128])),str(int(d[128:192],16)),str(int(d[192:256],16)),signed(d[256:320])))
        prog['next_block']=hi+1
        if len(logs)<5000:prog['span']=min(cfg['max_span'],int(prog['span']*1.5)+1)
        if len(rows)>=1_000_000:flush()
        if budget['calls']%200==0:
            flush();done=(prog['next_block']-prog['first_block'])/(prog['last_block']-prog['first_block']+1)
            print(f"{a.chain} {done:.1%} rows {prog['rows']} span {prog['span']} calls {budget['calls']} {time.time()-began:.0f}s",flush=True)
    flush();print(a.chain,'DONE',prog['rows'],'swaps',budget,flush=True)


if __name__=='__main__':main()
