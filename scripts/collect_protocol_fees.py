"""Collect each pool's protocol-fee history (SetFeeProtocol events), checked against archive state.

The history starts from slot0's feeProtocol just before --from-block, applies every
SetFeeProtocol event in order, and must end at slot0's value at --to-block. Output
is data/protocol/<pool>-protocol.json, read by scripts.fee_lvr and scripts.timing_ceiling.

  DEXNET_RPC_URLS=<endpoint> python -m scripts.collect_protocol_fees --data data/prepared --out data/protocol
"""
import argparse,json
from pathlib import Path
from dexnet.rpc import RotatingRPC

TOPIC='0x973d8d92bb299f4af6ce49b52a8adb85ae46b9f214c4c4fc06ac77401237b133'   # SetFeeProtocol(uint8,uint8,uint8,uint8)


def collect(rpc,address,start,end,chunk):
    def call(method,params):
        r=rpc(dict(jsonrpc='2.0',id=1,method=method,params=params))
        if 'result' not in r:raise RuntimeError(str(r.get('error')))
        return r['result']
    def slot(b):
        x=call('eth_call',[dict(to=address,data='0x3850c7bd'),hex(b)])
        return int(x[2+5*64:2+6*64],16)
    initial=slot(start-1);current=initial;changes=[]
    for lo in range(start,end+1,chunk):
        logs=call('eth_getLogs',[dict(address=address,fromBlock=hex(lo),toBlock=hex(min(lo+chunk-1,end)),topics=[TOPIC])])
        for r in sorted(logs,key=lambda r:(int(r['blockNumber'],16),int(r['logIndex'],16))):
            v=[int(r['data'][2+i*64:2+(i+1)*64],16) for i in range(4)]
            if v[0]+16*v[1]!=current:raise ValueError('Discontinuous protocol fee history')
            current=v[2]+16*v[3]
            changes.append(dict(block=int(r['blockNumber'],16),event_index=int(r['logIndex'],16),fee_protocol=current,transaction=r['transactionHash']))
    final=slot(end)
    if current!=final:raise ValueError('History does not reconcile to archive state at the end block')
    return dict(pool=address,from_block=start,to_block=end,initial=initial,final=final,changes=changes)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data',type=Path,required=True,help='prepared data (for pool addresses and block range)')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--from-block',type=int,help='default: first prepared block')
    p.add_argument('--to-block',type=int,help='default: last prepared block')
    p.add_argument('--chunk',type=int,default=10000,help='eth_getLogs block range per request')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    header=json.loads((a.data/'headers/COMPLETE.json').read_text())
    start=a.from_block or header['first_block'];end=a.to_block or header['last_block']
    rpc=RotatingRPC(interval=.3)
    for folder in sorted((a.data/'pools').iterdir()):
        dest=a.out/f'{folder.name}-protocol.json'
        if dest.exists() or not (folder/'metadata.json').exists():continue
        meta=json.loads((folder/'metadata.json').read_text())
        history=collect(rpc,meta['address'],max(start,meta.get('initialized_block',start)),end,a.chunk)
        dest.write_text(json.dumps(history,indent=2)+'\n')
        print(folder.name,'initial',history['initial'],'final',history['final'],'changes',len(history['changes']),flush=True)


if __name__=='__main__':main()
