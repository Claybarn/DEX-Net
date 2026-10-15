"""Labels for the direct baseline: exact outcomes at 5 min, 30 min, 2 h and 6 h after anchors
spaced every --stride blocks (including inactive blocks), plus training-only target scales.

  python -m scripts.prepare_direct_labels --data data/prepared --out runs/direct/labels
"""
import argparse,json,math,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pyarrow.parquet as pq
from dexnet.direct import HORIZONS,TARGET_NAMES,future_labels
from dexnet.utils import dump


def prepare(data,out,name,stride):
    data=Path(data);folder=data/'pools'/name;dest=Path(out)/name;dest.mkdir(parents=True,exist_ok=True)
    meta=json.loads((folder/'metadata.json').read_text());headers=json.loads((data/'headers/COMPLETE.json').read_text());exposure=json.loads((folder/'exposure.json').read_text());split=dict(train_before_block=next(v['first_block'] for v in exposure if v['split'] in ['val','heldout_val']),validation_before_block=next(v['first_block'] for v in exposure if v['split'] in ['test','heldout_test']))
    clock=np.load(data/'headers/timestamp.npy',mmap_mode='r');first=headers['first_block'];last=headers['last_block']
    blocks=np.load(folder/'block.npy',mmap_mode='r');timestamps=np.load(folder/'timestamp.npy',mmap_mode='r')
    arrays={k:[] for k in ['timestamp','logsqrt','logreserve','volume','turnover','invalid']}
    columns=['event','timestamp','amount0','amount1','sqrt_before','sqrt_after','liquidity_before','liquidity_after']
    sign=1 if meta['base_is_token0'] else -1;began=time.monotonic()
    for f in sorted((folder/'decoded').glob('*.parquet')):
        table=pq.read_table(f,columns=columns).to_pydict();n=len(table['timestamp'])
        sq=np.array([float(v) for v in table['sqrt_after']]);L=np.array([float(v) for v in table['liquidity_after']]);preL=np.array([float(v) for v in table['liquidity_before']]);preS=np.array([float(v) for v in table['sqrt_before']])
        with np.errstate(divide='ignore',invalid='ignore'):
            logreserve=np.log(L)+sign*(math.log(2**96)-np.log(sq));preR=np.log(preL)+sign*(math.log(2**96)-np.log(preS))
        swap=np.array([v=='Swap' for v in table['event']]);vol=np.array([abs(float(v)) for v in table['amount0' if sign==1 else 'amount1']]);vol=np.where(swap,vol,0.)
        valid=np.isfinite(preR);ratio=np.zeros(n);selected=swap&valid;ratio[selected]=vol[selected]*np.exp(-preR[selected])
        arrays['timestamp'].append(np.array(table['timestamp'],np.int64));arrays['logsqrt'].append(np.log(sq)*sign);arrays['logreserve'].append(logreserve);arrays['volume'].append(vol);arrays['turnover'].append(ratio);arrays['invalid'].append((swap&~valid).astype(np.int64))
    a={k:np.concatenate(v) for k,v in arrays.items()};del arrays
    raw_volume=a['volume'];raw_turnover=a['turnover']
    for k in ['volume','turnover','invalid']:a[k]=np.r_[0,np.cumsum(a[k],dtype=np.longdouble if k!='invalid' else np.int64)]
    # No outcome-based anchor sampling; equal block spacing across every pool.
    anchors=np.arange(max(first,int(blocks[63])),last+1,stride,dtype=np.int64);times=clock[anchors-first];ends=np.searchsorted(blocks,anchors,side='right')
    y,good,start,end=future_labels(a['timestamp'],a['logsqrt'],a['logreserve'],a['volume'],a['turnover'],a['invalid'],times)
    good &= (ends>=64)&(times+max(HORIZONS)<=clock[-1])
    age=np.stack([np.log1p(times-timestamps[ends-1]),np.log1p(anchors-blocks[ends-1])],-1).astype(np.float32)
    # Baseline: unchanged price and trailing-64-event volume/turnover rates.
    past_times=timestamps[np.maximum(ends-64,0)];past=np.searchsorted(a['timestamp'],past_times,side='left')
    span=np.maximum(times-past_times,1);vol=np.maximum(a['volume'][start+1]-a['volume'][past],0);turn=np.maximum(a['turnover'][start+1]-a['turnover'][past],0)
    baseline=np.zeros_like(y);baseline[:,:,1]=np.log1p(vol[:,None]*np.exp(-a['logreserve'][start,None])*np.array(HORIZONS)[None]/span[:,None]);baseline[:,:,2]=np.log1p(turn[:,None]*np.array(HORIZONS)[None]/span[:,None])
    train_time=int(clock[split['train_before_block']-first]);val_time=int(clock[split['validation_before_block']-first])
    masks={'train':times+max(HORIZONS)<train_time,'val':(times>=train_time)&(times+max(HORIZONS)<val_time),'test':(times>=val_time)&(times+max(HORIZONS)<=clock[-1])}
    for k,v in [('anchor_block',anchors),('history_end',ends),('age',age),('y',y.astype(np.float32)),('baseline',baseline.astype(np.float32))]:np.save(dest/(k+'.npy'),v)
    counts={}
    for splitname,mask in masks.items():
        if meta['held_out'] and splitname=='train':continue
        label=('heldout_'+splitname) if meta['held_out'] else splitname;rows=np.flatnonzero(good&mask);np.save(dest/(label+'.npy'),rows);counts[label]=len(rows)
    # Independent brute-force checks use decoded arrays at random eligible anchors.
    rng=np.random.default_rng(1982);verified=0
    for i in rng.choice(np.flatnonzero(good),min(64,int(good.sum())),replace=False):
        assert ends[i]==np.searchsorted(blocks,anchors[i],side='right') and blocks[ends[i]-1]<=anchors[i]
        for j,h in enumerate(HORIZONS):
            assert a['timestamp'][end[i,j]]<=times[i]+h
            if end[i,j]+1<len(a['timestamp']):assert a['timestamp'][end[i,j]+1]>times[i]+h
            direct_v=np.log1p(raw_volume[start[i]+1:end[i,j]+1].sum(dtype=np.longdouble)*np.exp(-a['logreserve'][start[i]]))
            direct_t=np.log1p(raw_turnover[start[i]+1:end[i,j]+1].sum(dtype=np.longdouble))
            if not np.allclose(y[i,j,1:],[direct_v,direct_t],rtol=1e-5,atol=1e-9):raise ValueError('Prefix target precision audit failed')
        verified+=1
    result=dict(pool=name,counts=counts,anchors=len(anchors),inactive_fraction=float(np.mean(age[good,1]>0)),excluded_invalid=int((~good).sum()),checked_anchors=verified,seconds=time.monotonic()-began)
    dump(dest/'COMPLETE.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stride',type=int,default=25);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    names=list(json.loads((a.data/'dataset.json').read_text())['pools']);results={}
    with ProcessPoolExecutor(max_workers=2) as ex:
        fs={ex.submit(prepare,str(a.data),str(a.out),n,a.stride):n for n in names}
        for f in as_completed(fs):results[fs[f]]=f.result();dump(a.out/'progress.json',results)
    moments=[];increments=[]
    for name in names:
        folder=a.out/name
        if not (folder/'train.npy').exists():continue
        rows=np.load(folder/'train.npy');y=np.load(folder/'y.npy',mmap_mode='r')[rows]
        moments.append(np.mean(y.astype(np.float64)**2,axis=0));inc=np.diff(np.concatenate([np.zeros((len(y),1,2)),y[:,:,1:]],axis=1),axis=1);increments.append(np.mean(inc.astype(np.float64)**2,axis=0))
    scales=np.maximum(np.sqrt(np.mean(moments,axis=0)),np.array([1.,1e-5,1e-5]));inc=np.maximum(np.sqrt(np.mean(increments,axis=0)),1e-5)
    dump(a.out/'scales.json',dict(scales=scales.tolist(),increments=inc.tolist(),fit='Equal-pool training-only second moments; no heldout pools'))
    dump(a.out/'COMPLETE.json',dict(pools=results,horizons_seconds=list(HORIZONS),targets=list(TARGET_NAMES),anchor_stride_blocks=a.stride,cutoff='All events in blocks with timestamp <= anchor timestamp + horizon; six-hour labels wholly inside split',volume='Absolute economic-base amounts, includes exact decoded swaps omitted from model event histories',zero_L='Anchors with undefined initial reserve or future swaps with undefined reserve excluded and counted'))


if __name__=='__main__':main()
