"""Collect raw Initialize/Swap/Mint/Burn logs for the ten-pool Ethereum roster (resumable).

Endpoints come from DEXNET_RPC_URLS (comma-separated). They must serve historical
eth_getLogs and archive eth_call. Endpoint URLs are never written to disk.

  DEXNET_RPC_URLS=<endpoint> python -m scripts.collect_logs --out data/raw
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from dexnet.rpc import RotatingRPC

FACTORY='0x1f98431c8ad98523631ae4a59f267346ea31f984'
FIRST_BLOCK=12369621
ROSTER_BLOCK=21761563
END_BLOCK=25983637
TOKENS={
    'WETH':'0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
    'USDC':'0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
    'USDT':'0xdac17f958d2ee523a2206206994597c13d831ec7',
    'DAI':'0x6b175474e89094c44da98b954eedeac495271d0f',
    'WBTC':'0x2260fac5e5542a773aa44fbcfedf7c193bc2c599'}
ROSTER=[(token,'WETH',fee) for token in ['USDC','USDT','DAI','WBTC'] for fee in [500,3000]]+[('USDC','USDT',100),('USDC','USDT',500)]
SIGNATURES=['Initialize(uint160,int24)','Swap(address,address,int256,int256,uint160,uint128,int24)',
            'Mint(address,address,int24,int24,uint128,uint256,uint256)','Burn(address,int24,int24,uint128,uint256,uint256)']


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def validate_logs(logs,address,lo,hi,topics):
    seen=set(); previous=None
    for log in logs:
        block=int(log['blockNumber'],16); index=int(log['logIndex'],16)
        key=(block,index)
        if log.get('removed') or not lo<=block<=hi or log['address'].lower()!=address.lower() or log['topics'][0] not in topics:
            raise ValueError('Unexpected/removed log')
        if key in seen or (previous is not None and key<=previous):raise ValueError('Duplicate or unordered event')
        for field in ['transactionHash','blockHash']:
            if len(log[field])!=66:raise ValueError('Invalid hash')
        seen.add(key);previous=key


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-hours',type=float,default=24);p.add_argument('--max-gb',type=float,default=12)
    p.add_argument('--max-chunks',type=int);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rpc=RotatingRPC(interval=.6,cooldown=10)
    log_rpc=RotatingRPC(interval=.6,cooldown=10)
    def call(method,params,attempts=6):
        client=log_rpc if method=='eth_getLogs' else rpc
        return client(dict(jsonrpc='2.0',id=1,method=method,params=params),attempts=attempts)['result']
    def header(block):
        h=call('eth_getBlockByNumber',[hex(block),False]);return dict(number=int(h['number'],16),hash=h['hash'],timestamp=int(h['timestamp'],16))
    if int(call('eth_chainId',[]),16)!=1:raise ValueError('Wrong chain')
    selectors={}
    def selector(signature):
        if signature not in selectors:selectors[signature]=call('web3_sha3',['0x'+signature.encode().hex()])[:10]
        return selectors[signature]
    def ethcall(address,signature,args='',block=ROSTER_BLOCK):
        return call('eth_call',[dict(to=address,data=selector(signature)+args),hex(block)])
    fixed=dict(chain_id=1,factory=FACTORY,roster_block=ROSTER_BLOCK,first_block=FIRST_BLOCK,end_block=END_BLOCK,
        candidates=ROSTER,tokens=TOKENS,source='https://github.com/Uniswap/v3-periphery/blob/main/deploys.md',
        selection='Fixed asset/fee grid verified at historical training boundary; no current volume/survival filter. Absent candidates retained in manifest, no replacements.',
        status='Raw staging only; no model training from these pools until units, reconstruction and common temporal/held-out-pool splits are validated.',
        scope='Complete Initialize/Swap/Mint/Burn logs from factory deployment through fixed existing-data end. Empty chunks retained; no-trade exposure is explicit. Collect/Flash/protocol fee changes not covered; fetch before fee-accounting backtests.',
        calendar_splits=dict(train_before_block=21761563,validation_before_block=23795358,end_inclusive=END_BLOCK,held_out_pools=['WBTC-WETH-3000','USDC-USDT-500'],rule='Common boundaries across every pool; held-out pools excluded from fitting and tuning; purge outcome windows across boundaries.'),
        log_routes=len(log_rpc.urls),maximum_chunk_blocks=5000,
        budget=dict(max_hours=a.max_hours,max_gb=a.max_gb,interval=.6),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Normalize tuples through JSON before comparison on resume.
    fixed=json.loads(json.dumps(fixed));protocol=a.out/'protocol-v2.json'
    if protocol.exists() and json.loads(protocol.read_text())!=fixed:raise ValueError('Collection protocol changed')
    save(protocol,fixed)
    topics=[call('web3_sha3',['0x'+s.encode().hex()]) for s in SIGNATURES]
    save(a.out/'topics.json',dict(zip(SIGNATURES,topics)))
    pools=[]
    for token,other,fee in ROSTER:
        name=f'{token}-{other}-{fee}';folder=a.out/name;folder.mkdir(exist_ok=True);dest=folder/'metadata.json'
        if dest.exists():meta=json.loads(dest.read_text())
        else:
            left,right=sorted([TOKENS[token],TOKENS[other]])
            args=left[2:].zfill(64)+right[2:].zfill(64)+hex(fee)[2:].zfill(64)
            address='0x'+ethcall(FACTORY,'getPool(address,address,uint24)',args)[-40:]
            meta=dict(name=name,pair=[token,other],fee=fee,address=address,exists_at_roster=address!='0x'+'0'*40,roster_block=ROSTER_BLOCK)
            if meta['exists_at_roster']:
                for field,signature in [('factory','factory()'),('token0','token0()'),('token1','token1()')]:meta[field]='0x'+ethcall(address,signature)[-40:]
                meta['tick_spacing']=int(ethcall(address,'tickSpacing()'),16)
                if meta['factory']!=FACTORY or [meta['token0'],meta['token1']]!=[left,right] or int(ethcall(address,'fee()'),16)!=fee:raise ValueError('Pool metadata mismatch')
                meta['decimals0']=int(ethcall(left,'decimals()'),16);meta['decimals1']=int(ethcall(right,'decimals()'),16)
            save(dest,meta)
        pools.append(meta);print('METADATA',name,meta['address'],flush=True)
    save(a.out/'roster.json',dict(pools=pools,roster_header=header(ROSTER_BLOCK),end_header=header(END_BLOCK)))
    statepath=a.out/'state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else dict(pools={m['name']:dict(next_block=FIRST_BLOCK,chunk=5000,logs=0,chunks=0) for m in pools if m['exists_at_roster']},bytes=0,chunks=0)
    # Chunks are write-ahead records: recover exactly one committed chunk if state save was interrupted.
    for name,s in state['pools'].items():
        folder=a.out/name
        s['chunk']=min(s['chunk'],5000)
        while True:
            files=list(folder.glob(str(s['next_block'])+'-*.json.gz'))
            if not files:break
            if len(files)!=1:raise ValueError('Ambiguous chunk recovery')
            with gzip.open(files[0],'rt') as f:record=json.load(f)
            assert record['from_block']==s['next_block'] and record['to_block']<=END_BLOCK
            s['next_block']=record['to_block']+1;s['logs']+=len(record['logs']);s['chunks']+=1
            state['bytes']+=files[0].stat().st_size;state['chunks']+=1
    save(statepath,state);started=time.monotonic();n=0
    try:
        while any(s['next_block']<=END_BLOCK for s in state['pools'].values()):
            for m in pools:
                if not m['exists_at_roster']:continue
                s=state['pools'][m['name']]
                if s['next_block']>END_BLOCK:continue
                reason=None
                if time.monotonic()-started>a.max_hours*3600:reason='time budget'
                if state['bytes']>=a.max_gb*1024**3:reason='data budget'
                if shutil.disk_usage(a.out).free<20*1024**3:reason='20 GiB free-disk reserve'
                if a.max_chunks and n>=a.max_chunks:reason='chunk budget'
                if reason:
                    save(a.out/'PAUSED.json',dict(reason=reason,time=time.time(),state=state));return
                lo=s['next_block'];hi=min(END_BLOCK,lo+s['chunk']-1)
                payload=dict(address=m['address'],fromBlock=hex(lo),toBlock=hex(hi),topics=[topics])
                try:logs=call('eth_getLogs',[payload],attempts=2)
                except RuntimeError:
                    if s['chunk']<=100:raise
                    s['chunk']=max(100,s['chunk']//2);save(statepath,state);continue
                if len(logs)>=9000:
                    if s['chunk']<=1:raise ValueError('Single-block log cap')
                    s['chunk']=max(1,s['chunk']//2);save(statepath,state);continue
                validate_logs(logs,m['address'],lo,hi,topics)
                endheader=header(hi)
                if logs and int(logs[-1]['blockNumber'],16)==hi and logs[-1]['blockHash']!=endheader['hash']:raise ValueError('Noncanonical endpoint')
                record=dict(pool=m['address'],from_block=lo,to_block=hi,end_header=endheader,provider=log_rpc.active,logs=logs)
                dest=a.out/m['name']/f'{lo}-{hi}.json.gz';tmp=dest.with_suffix('.tmp')
                with gzip.open(tmp,'wt') as f:json.dump(record,f,separators=(',',':'))
                os.replace(tmp,dest)
                s['next_block']=hi+1;s['logs']+=len(logs);s['chunks']+=1
                if len(logs)<1500:s['chunk']=min(5000,s['chunk']*2)
                state['bytes']+=dest.stat().st_size;state['chunks']+=1;n+=1
                state['updated']=time.time();save(statepath,state)
                log_rpc.active=(log_rpc.active+1)%len(log_rpc.urls)
                print(json.dumps(dict(pool=m['name'],through=hi,events=len(logs),total_bytes=state['bytes'],chunks=state['chunks'])),flush=True)
        save(a.out/'COMPLETE.json',dict(time=time.time(),state=state,meaning='Raw log scope complete; reconstruction/coverage audit and common-time splits still required.'))
        (a.out/'PAUSED.json').unlink(missing_ok=True)
    except Exception as exc:save(a.out/'FAILED.json',dict(error=repr(exc),time=time.time()));raise

if __name__=='__main__':main()
