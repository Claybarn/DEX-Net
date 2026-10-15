"""Decode raw logs, reconstruct and audit exact pool state, and build the training dataset.

  python -m scripts.prepare_data --raw data/raw --headers data/headers --out data/prepared
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dexnet.events import (FEATURES,TARGETS,SCALE_FEATURES,decode,LiquidityLedger,
                                     orientation,event_features,eligible_indices)
from scripts.collect_logs import validate_logs
import dexnet.events as events_module


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda:f.read(8<<20),b''):h.update(piece)
    return h.hexdigest()


def headers(a,protocol):
    dest=a.out/'headers';dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():return
    from scripts.collect_headers import normalize,day_keys,BUCKET,rpc_headers
    from dexnet.rpc import RotatingRPC
    import pyarrow.fs as fs
    report=json.loads((a.headers/'report.json').read_text())
    assert (a.headers/'COMPLETE').exists() and report['full_parent_hash_chain_validated']
    first,last=protocol['first_block'],protocol['end_block']
    files=sorted((a.headers/'days').glob('*.parquet'))
    prefix=[];rpc=RotatingRPC(interval=.3)
    if first<report['first_block']:
        anchors=rpc_headers(rpc,[first,report['first_block']])
        lo=datetime.fromtimestamp(int(anchors.timestamp.iloc[0]),timezone.utc).date()
        hi=datetime.fromtimestamp(int(anchors.timestamp.iloc[1]),timezone.utc).date()
        s3=fs.S3FileSystem(anonymous=True,region='us-east-2',request_timeout=60,connect_timeout=15)
        day=lo
        while day<=hi:
            path=dest/(str(day)+'-prefix.parquet')
            if not path.exists():
                tables=[pq.ParquetFile(BUCKET+'/'+key,filesystem=s3).read(columns=['number','timestamp','hash','parent_hash']) for key in day_keys(str(day))]
                f=normalize(pa.concat_tables(tables).to_pandas())
                f=f[(f.block_id>=first)&(f.block_id<report['first_block'])]
                f.to_parquet(path,index=False)
            prefix.append(path);day+=timedelta(days=1)
    clock=np.lib.format.open_memmap(dest/'timestamp.npy',mode='w+',dtype='int64',shape=(last-first+1,))
    hashes=np.lib.format.open_memmap(dest/'hash.npy',mode='w+',dtype='V32',shape=(last-first+1,))
    cursor=first;previous=None;previous_time=None
    for file in prefix+files:
        f=normalize(pq.read_table(file,columns=['block_id','timestamp','hash','parent_hash']).to_pandas())
        f=f[(f.block_id>=first)&(f.block_id<=last)]
        if not len(f):continue
        b=f.block_id.to_numpy(np.int64);t=f.timestamp.to_numpy(np.int64)
        assert b[0]==cursor and np.all(np.diff(b)==1),'Header coverage gap'
        if previous is not None:
            assert f.parent_hash.iloc[0]==previous and t[0]>previous_time,'Header chain discontinuity'
        clock[b-first]=t;hashes[b-first]=np.asarray(f.hash.tolist(),dtype='V32')
        cursor=int(b[-1])+1;previous=f.hash.iloc[-1];previous_time=int(t[-1])
    assert cursor==last+1
    clock.flush();hashes.flush()
    anchors=rpc_headers(rpc,[first,report['first_block'],last])
    for r in anchors.itertuples():
        assert bytes(hashes[r.block_id-first])==r.hash and clock[r.block_id-first]==r.timestamp
    save(dest/'COMPLETE.json',dict(first_block=first,last_block=last,full_parent_chain=True,
                                  rpc_anchors=len(anchors),source_report_sha256=sha(a.headers/'report.json')))


def prepare_pool(raw,out,meta,protocol):
    raw,out=Path(raw),Path(out);folder=out/'pools'/meta['name'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'AUDIT_COMPLETE.json').exists():return json.loads((folder/'AUDIT_COMPLETE.json').read_text())
    decoded=folder/'decoded';decoded.mkdir(exist_ok=True)
    shards=folder/'shards';shards.mkdir(exist_ok=True)
    # Restart an incomplete pool from initialization; named outputs are overwritten.
    base=orientation(meta);tokens=protocol['tokens'];meta=dict(meta,base=base,
        base_is_token0=meta['token0'].lower()==tokens[base].lower(),held_out=meta['name'] in protocol['calendar_splits']['held_out_pools'])
    meta['quote']=next(x for x in meta['pair'] if x!=base)
    meta['metadata_feature_names']=['log_fee_fraction','log_tick_spacing','decimals0_div18','decimals1_div18','base_is_token0']
    meta['metadata_features']=[math.log(meta['fee']/1e6),math.log(meta['tick_spacing']),meta['decimals0']/18,meta['decimals1']/18,int(meta['base_is_token0'])]
    save(folder/'metadata.json',meta)
    first,last=protocol['first_block'],protocol['end_block']
    times=np.load(out/'headers/timestamp.npy',mmap_mode='r');hashes=np.load(out/'headers/hash.npy',mmap_mode='r')
    topics=json.loads((raw/'topics.json').read_text());topic_names={v:k.split('(')[0] for k,v in topics.items()}
    ledger=LiquidityLedger(meta['tick_spacing']);counts=Counter();model_counts=Counter();cursor=first;previous_key=None
    previous=None;init_block=None;buffer=[];numeric=[];parts=[];model_rows=0;decoded_rows=0;chunk_count=0;empty=0
    raw_hasher=hashlib.sha256();started=time.time();invalid=Counter()
    def flush():
        nonlocal buffer,numeric,model_rows,decoded_rows
        index=len(parts);path=decoded/f'{index:05d}.parquet'
        # Explicit schema retains exact uint128/uint160/int256 values as strings.
        ints=['block','log_index','tx_index','timestamp','lower','upper','tick','tick_before','tick_after']
        texts=['event','transaction','block_hash','owner','amount','amount0','amount1','sqrt','liquidity','sqrt_before','liquidity_before','sqrt_after','liquidity_after']
        schema=pa.schema([(k,pa.int64()) for k in ints]+[(k,pa.string()) for k in texts])
        pq.write_table(pa.Table.from_pylist(buffer,schema=schema),path,compression='zstd')
        if numeric:
            values={k:np.asarray([r[k] for r in numeric],dtype=dtype) for k,dtype in
                    [('raw_features','float32'),('feature_mask','bool'),('targets','float64'),('target_mask','bool'),
                     ('kind','int8'),('block','int64'),('timestamp','int64'),('log_index','int32'),('gap_blocks','int64')]}
            np.savez(shards/f'{index:05d}.npz',**values)
        parts.append(dict(part=index,decoded=len(buffer),model=len(numeric)))
        model_rows+=len(numeric);decoded_rows+=len(buffer);buffer=[];numeric=[]
    try:
        for path in sorted((raw/meta['name']).glob('*.json.gz'),key=lambda p:int(p.name.split('-')[0])):
            compressed=path.read_bytes();raw_hasher.update(path.name.encode());raw_hasher.update(hashlib.sha256(compressed).digest())
            record=json.loads(gzip.decompress(compressed));lo,hi=record['from_block'],record['to_block']
            assert lo==cursor and lo<=hi<=last and record['pool'].lower()==meta['address'].lower(),'Chunk gap/overlap/identity'
            end=record['end_header'];assert end['number']==hi and bytes(hashes[hi-first]).hex()==end['hash'][2:].lower() and times[hi-first]==end['timestamp']
            logs=record['logs'];validate_logs(logs,meta['address'],lo,hi,list(topic_names))
            empty+=not logs;chunk_count+=1
            for log in logs:
                r=decode(log,topic_names);key=(r['block'],r['log_index'])
                assert previous_key is None or key>previous_key,'Duplicate/out-of-order log across chunks'
                previous_key=key;r['timestamp']=int(times[r['block']-first])
                assert bytes(hashes[r['block']-first]).hex()==r['block_hash'][2:].lower(),'Noncanonical event block'
                if 'blockTimestamp' in log:assert int(log['blockTimestamp'],16)==r['timestamp']
                before=(ledger.L,ledger.sqrt,ledger.tick)
                available=ledger.ranges.get((r['lower'],r['upper']),0)
                ledger.apply(r);counts[r['event']]+=1
                if r['event']=='Initialize':
                    init_block=r['block'];previous=(r['block'],r['timestamp'])
                else:
                    value=event_features(r,before,ledger,meta,previous,available)
                    if value is not None:
                        f,fm,y,ym,kind,gb=value
                        assert np.isfinite(f).all() and np.isfinite(y).all()
                        numeric.append(dict(raw_features=f,feature_mask=fm,targets=y,target_mask=ym,kind=kind,
                                            block=r['block'],timestamp=r['timestamp'],log_index=r['log_index'],gap_blocks=gb))
                        model_counts[str(kind)]+=1;previous=(r['block'],r['timestamp'])
                        if not f[11]:invalid['no_pre_event_active_liquidity']+=1
                    else:invalid['zero_burn' if r['event']=='Burn' else 'zero_base_swap']+=1
                r.update(sqrt_before=str(before[1]),liquidity_before=str(before[0]),tick_before=before[2],
                         sqrt_after=str(ledger.sqrt),liquidity_after=str(ledger.L),tick_after=ledger.tick)
                for k in ['amount','amount0','amount1','sqrt','liquidity']:r[k]=str(r[k])
                buffer.append(r)
                if len(buffer)>=50000:flush()
            cursor=hi+1
            if chunk_count%100==0:
                save(folder/'progress.json',dict(stage='decode_reconstruct',through_block=hi,raw_events=sum(counts.values()),seconds=time.time()-started))
        if buffer:flush()
        state=json.loads((raw/'state.json').read_text())['pools'][meta['name']]
        assert cursor==last+1 and counts['Initialize']==1 and decoded_rows==state['logs'] and chunk_count==state['chunks']
        # Materialize contiguous memory-mappable arrays from bounded shards.
        spec={'raw_features':('float32',(len(FEATURES),)),'feature_mask':('bool',(len(FEATURES),)),
              'targets':('float64',(len(TARGETS),)),'target_mask':('bool',(len(TARGETS),)),
              'kind':('int8',()),'block':('int64',()),'timestamp':('int64',()),'log_index':('int32',()),'gap_blocks':('int64',())}
        arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=d,shape=(model_rows,)+shape) for k,(d,shape) in spec.items()}
        pos=0
        for part in parts:
            if not part['model']:continue
            with np.load(shards/f'{part["part"]:05d}.npz') as v:
                n=part['model']
                for k,arr in arrays.items():arr[pos:pos+n]=v[k]
                pos+=n
        for arr in arrays.values():arr.flush()
        assert pos==model_rows
        save(folder/'final-inventory.json',dict(ranges=[[lo,hi,str(v)] for (lo,hi),v in ledger.ranges.items()],
             positions=[[owner,lo,hi,str(v)] for (owner,lo,hi),v in ledger.positions.items()],
             sqrt=str(ledger.sqrt),tick=ledger.tick,liquidity=str(ledger.L)))
        save(folder/'parts.json',parts)
        audit=dict(pool=meta['name'],decoded_events=decoded_rows,model_events=model_rows,event_counts=dict(counts),
                   model_kind_counts=dict(model_counts),special_cases=dict(invalid),chunks=chunk_count,empty_chunks=empty,
                   first_block=first,last_block=last,initialized_block=init_block,all_log_and_chunk_endpoint_hashes_match=True,
                   swap_liquidity_checks=counts['Swap'],active_liquidity_mismatches=0,owner_burn_mismatches=0,
                   raw_chunk_manifest_sha256=raw_hasher.hexdigest(),seconds=time.time()-started)
        save(folder/'AUDIT_COMPLETE.json',audit);return audit
    except Exception as exc:
        save(folder/'FAILED.json',dict(error=repr(exc),block=cursor,time=time.time()));raise


def moments(folder,stop,field,maskfield):
    x=np.load(folder/(field+'.npy'),mmap_mode='r');mask=np.load(folder/(maskfield+'.npy'),mmap_mode='r')
    count=np.zeros(x.shape[1],np.int64);total=np.zeros(x.shape[1]);square=total.copy()
    for start in range(64,stop,100000):
        a=np.asarray(x[start:min(start+100000,stop)],dtype=float);m=mask[start:min(start+100000,stop)]
        count+=m.sum(0);total+=(a*m).sum(0);square+=(a*a*m).sum(0)
    return dict(count=count.tolist(),mean=(total/np.maximum(count,1)).tolist(),second=(square/np.maximum(count,1)).tolist())


def finalize(a,protocol,metas):
    split=protocol['calendar_splits'];train,val=split['train_before_block'],split['validation_before_block'];end=protocol['end_block']+1
    pooled={};fit={}
    for field,maskfield in [('raw_features','feature_mask'),('targets','target_mask')]:
        stats={}
        for m in metas:
            if m['name'] in split['held_out_pools']:continue
            folder=a.out/'pools'/m['name'];blocks=np.load(folder/'block.npy',mmap_mode='r')
            stats[m['name']]=moments(folder,int(np.searchsorted(blocks,train)),field,maskfield)
        means=np.array([s['mean'] for s in stats.values()]);seconds=np.array([s['second'] for s in stats.values()]);valid=np.array([s['count'] for s in stats.values()])>0
        n=valid.sum(0);assert np.all(n>0)
        mean=(means*valid).sum(0)/n;second=(seconds*valid).sum(0)/n;std=np.sqrt(np.maximum(second-mean*mean,1e-8))
        std=np.where(std<1e-4,1.,std)
        if field=='raw_features':
            keep=np.ones(len(mean),bool);keep[SCALE_FEATURES]=False;mean[keep]=0;std[keep]=1
        pooled[field]=dict(mean=mean.tolist(),std=std.tolist());fit[field]=stats
    save(a.out/'scalers.json',dict(method='Equal weight per eligible training pool, natural event frequency within pool; masks exclude inapplicable/undefined marks.',
                                 train_before_block=train,excluded_pools=split['held_out_pools'],scalers=pooled,fit_pool_moments=fit))
    summaries={}
    for m in metas:
        folder=a.out/'pools'/m['name'];meta=json.loads((folder/'metadata.json').read_text())
        blocks=np.load(folder/'block.npy',mmap_mode='r');timestamps=np.load(folder/'timestamp.npy',mmap_mode='r')
        raw=np.load(folder/'raw_features.npy',mmap_mode='r');mask=np.load(folder/'feature_mask.npy',mmap_mode='r')
        x=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype='float32',shape=raw.shape)
        mean=np.array(pooled['raw_features']['mean']);std=np.array(pooled['raw_features']['std'])
        maximum=np.zeros(len(mean))
        for start in range(0,len(raw),100000):
            v=np.where(mask[start:start+100000],(raw[start:start+100000]-mean)/std,0).astype(np.float32)
            assert np.isfinite(v).all();x[start:start+len(v)]=v;maximum=np.maximum(maximum,np.max(np.abs(v),axis=0))
        x.flush();indices=folder/'indices';indices.mkdir(exist_ok=True);numbers={}
        definitions=({'heldout_val':(train,val),'heldout_test':(val,end)} if meta['held_out'] else {'train':(protocol['first_block'],train),'val':(train,val),'test':(val,end)})
        for name,(lo,hi) in definitions.items():
            for h in [1,8,32,64]:
                rows=eligible_indices(blocks,lo,hi,horizon=h);np.save(indices/f'{name}-h{h}.npy',rows)
                numbers[f'{name}-h{h}']=len(rows)
                if len(rows):assert blocks[rows[0]]>=lo and blocks[rows[-1]+h-1]<hi
        # Daily counts explicitly include zero-event days after initialization.
        audit=json.loads((folder/'AUDIT_COMPLETE.json').read_text());first=protocol['first_block']
        headers_time=np.load(a.out/'headers/timestamp.npy',mmap_mode='r')
        day0=int(headers_time[audit['initialized_block']-first]//86400);day1=int(headers_time[-1]//86400)
        days=np.arange(day0,day1+1);kind=np.load(folder/'kind.npy',mmap_mode='r')
        daily={'utc_day_start':days*86400}
        for k in range(4):daily[f'kind_{k}_count']=np.bincount((timestamps[kind==k]//86400-day0).astype(int),minlength=len(days))
        pq.write_table(pa.table(daily),folder/'daily-exposure.parquet',compression='zstd')
        exposure=[]
        for name,(lo,hi) in {'train':(first,train),'val':(train,val),'test':(val,end)}.items():
            left=max(lo,audit['initialized_block']);right=hi-1
            if left>right:continue
            l,r=np.searchsorted(blocks,[left,hi]);prior=l-1
            exposure.append(dict(split=name,held_out=meta['held_out'],first_block=left,last_block=right,
                first_timestamp=int(headers_time[left-first]),last_timestamp=int(headers_time[right-first]),
                produced_blocks=right-left+1,events=int(r-l),
                previous_event_block=int(blocks[prior]) if prior>=0 else None,
                last_event_block=int(blocks[r-1]) if r>0 else None,
                right_censored_seconds=int(headers_time[right-first]-timestamps[r-1]) if r>0 else None))
        save(folder/'exposure.json',exposure)
        summaries[m['name']]=dict(rows=len(blocks),indices=numbers,held_out=meta['held_out'],max_abs_scaled_feature=maximum.tolist())
    save(a.out/'dataset.json',dict(version='pooled-v3-v1',features=FEATURES,targets=TARGETS,seq_len=64,horizons=[1,8,32,64],
        pools=summaries,scaler_file='scalers.json',kind_names=['base_sell','base_buy','mint','burn'],
        constraints=['Use dexnet.events.PooledDataset.','Masks required for zero-liquidity ratios; no clipping.',
                     'Choose training pools uniformly then rows naturally within pool; held-out pools never fitted.',
                     'Targets remain unstandardized; target scaler is supplied separately with masks.',
                     'No USD peg or dollar valuation; swap amount is relative to causal virtual base reserve.',
                     'Raw log, state, owner inventory and empty-range coverage audited; fee accounting not included.']))
    # A complete marker is written only after the full artifact set passed checks.
    save(a.out/'COMPLETE.json',dict(time=time.time(),pools=len(metas),model_rows=sum(s['rows'] for s in summaries.values()),
                                  preparation='decoded, exact liquidity audited, canonical header hashes checked, training-only scalers and purged indices built'))


def run(a):
    lock=(a.out/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (a.raw/'COMPLETE.json').exists()
    protocol=json.loads((a.raw/'protocol-v2.json').read_text());metas=json.loads((a.raw/'roster.json').read_text())['pools']
    assert all(m['exists_at_roster'] for m in metas)
    plan=dict(raw=str(a.raw),headers=str(a.headers),raw_protocol_sha256=sha(a.raw/'protocol-v2.json'),
              first_block=protocol['first_block'],end_block=protocol['end_block'],workers=2,
              source_sha256={p.name:sha(p) for p in [Path(__file__),Path(events_module.__file__)]})
    if (a.out/'protocol.json').exists():assert json.loads((a.out/'protocol.json').read_text())==plan
    else:save(a.out/'protocol.json',plan)
    save(a.out/'progress.json',dict(stage='canonical_headers',time=time.time()));headers(a,protocol)
    done={};failures={}
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(prepare_pool,str(a.raw),str(a.out),m,protocol):m['name'] for m in metas}
        for future in as_completed(jobs):
            name=jobs[future]
            try:done[name]=future.result()
            except Exception as exc:failures[name]=repr(exc)
            save(a.out/'progress.json',dict(stage='decode_reconstruct',completed=list(done),failures=failures,time=time.time()))
    if failures:raise RuntimeError(f'Pools failed audit: {failures}')
    save(a.out/'audit-summary.json',done)
    save(a.out/'progress.json',dict(stage='scaling_and_splits',time=time.time()));finalize(a,protocol,metas)
    save(a.out/'progress.json',dict(stage='complete',time=time.time()))


def main():
    p=argparse.ArgumentParser()
    for name in ['raw','headers','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    for k in ['raw','headers','out']:setattr(a,k,getattr(a,k).resolve())
    a.out.mkdir(parents=True,exist_ok=True)
    if a.launch:
        cmd=[sys.executable,'-u','-m','scripts.prepare_data','--raw',str(a.raw),'--headers',str(a.headers),'--out',str(a.out)]
        with (a.out/'queue.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=Path.cwd(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        value=dict(pid=child.pid,time=time.time(),command=cmd);save(a.out/'launch.json',value);print(json.dumps(value));return
    try:run(a)
    except Exception as exc:save(a.out/'FAILED.json',dict(error=repr(exc),time=time.time()));raise


if __name__=='__main__':main()
