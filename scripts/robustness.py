"""Robustness checks for the rollout comparisons (energy score, 512 events by default).

  shrinkage    a blend baseline: equal mixture of the recent and expanding history
               windows (128 of each), testing whether DEX-Net's edge in volatile regimes
               is only shrinkage toward long-run behavior; reported overall and by
               thirds of recent/long-run volatility
  seeds        bootstrap over anchors AND training seeds (each draw resamples three
               seeds per family), and a pool-clustered bootstrap for seen pools
  pools        per-pool means of the paired difference
  endpoints    fair CRPS of each scaled endpoint, for what drives the energy score

  python -m scripts.robustness --rollouts depth=eval/final/rollouts-test-depth \
      --rollouts reference=eval/final/rollouts-test-reference --out results/robustness-test.json
"""
import argparse,json
from pathlib import Path
import numpy as np
from dexnet.diagnostics import proper_scores,ENDPOINT_SCALES

NAMES=('return','liquidity','blocks','volume')


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rollouts',action='append',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--horizon',default='512');a=p.parse_args();h=a.horizon;rng=np.random.default_rng(0)
    fams=dict(x.split('=',1) for x in a.rollouts);first=next(iter(fams))
    files={f:{x.name:x for x in (Path(d)/'anchors').glob('*.json')} for f,d in fams.items()};names=sorted(set.intersection(*(set(v) for v in files.values())))
    rows=[]
    for n in names:
        R={f:json.loads(files[f][n].read_text()) for f in fams};r0=R[first];y=np.asarray(r0['actual'][h],float)
        rec=np.array([q['values'][h] for q in r0['paths']['baseline:recent']]);ex=np.array([q['values'][h] for q in r0['paths']['baseline:expanding']])
        blend=np.concatenate([rec[::2],ex[::2]])
        row=dict(pool=r0['pool'],group='heldout' if r0['split'].startswith('heldout') else 'seen',seeds={},crps={})
        for f,r in R.items():
            per=[]
            for lab,ps in r['paths'].items():
                if lab.startswith('baseline:'):continue
                x=np.array([q['values'][h] for q in ps if h in q['values']]);e,c=proper_scores(x,y);per.append(e)
                row['crps'].setdefault(f,[]).append((np.asarray(c)/ENDPOINT_SCALES).tolist())
            row['seeds'][f]=per;row['crps'][f]=np.mean(row['crps'][f],0).tolist()
        for b,x in (('recent',rec),('expanding',ex),('blend',blend)):
            e,c=proper_scores(x,y);row[b]=e;row['crps'][b]=(np.asarray(c)/ENDPOINT_SCALES).tolist()
        width=lambda v:float(np.subtract(*np.quantile(v,[.95,.05])))
        row['regime']=width(rec[:,0])/max(width(ex[:,0]),1e-9);rows.append(row)
    def ci(d,draws):return dict(mean=float(np.mean(d)),ci95=[float(x) for x in np.quantile(draws,[.025,.975])])
    result=dict(method=__doc__,horizon=h,groups={})
    for group in ('seen','heldout'):
        G=[r for r in rows if r['group']==group];out={}
        model=lambda r,f:float(np.mean(r['seeds'][f]))
        # Shrinkage baseline, overall and by regime thirds.
        sh={}
        for b in ('recent','blend'):
            d=np.array([model(r,first)-r[b] for r in G]);idx=rng.integers(0,len(d),(4000,len(d)))
            sh[f'{first}-{b}']=ci(d,d[idx].mean(1))
            v=np.array([r['regime'] for r in G]);q=np.quantile(v,[1/3,2/3]);thirds={}
            for name,m in (('calm',v<=q[0]),('mid',(v>q[0])&(v<=q[1])),('volatile',v>q[1])):
                dm=d[m];ix=rng.integers(0,len(dm),(4000,len(dm)));thirds[name]=ci(dm,dm[ix].mean(1))
            sh[f'{first}-{b} by regime']=thirds
        out['shrinkage']=sh;out['baseline_energy']={b:float(np.mean([r[b] for r in G])) for b in ('recent','expanding','blend')}
        # Seed-and-anchor bootstrap, and pool-clustered bootstrap, for every pair of families and model vs recent.
        pairs=[(f,g) for f in fams for g in fams if f<g]+[(f,'recent') for f in fams]
        sb={}
        for f,g in pairs:
            A=np.array([r['seeds'][f] for r in G]);B=np.array([r['seeds'][g] for r in G]) if g in fams else np.array([[r[g]]*A.shape[1] for r in G])
            draws=[]
            for _ in range(2000):
                ia=rng.integers(0,len(G),len(G));sa=rng.integers(0,A.shape[1],A.shape[1]);sb_=rng.integers(0,B.shape[1],B.shape[1])
                draws.append(A[ia][:,sa].mean()-B[ia][:,sb_].mean())
            entry=dict(anchors_and_seeds=ci(A.mean(1)-B.mean(1),draws))
            pools=sorted({r['pool'] for r in G})
            if len(pools)>2:
                pd={pl:(A.mean(1)-B.mean(1))[[r['pool']==pl for r in G]] for pl in pools};cd=[]
                for _ in range(2000):
                    pick=rng.choice(pools,len(pools));cd.append(np.concatenate([pd[x] for x in pick]).mean())
                entry['pool_clustered']=ci(A.mean(1)-B.mean(1),cd)
            entry['per_pool']={pl:float((A.mean(1)-B.mean(1))[[r['pool']==pl for r in G]].mean()) for pl in pools}
            sb[f'{f}-{g}']=entry
        out['bootstrap']=sb
        out['endpoint_crps']={f:dict(zip(NAMES,np.mean([r['crps'][f] for r in G],0).tolist())) for f in list(fams)+['recent','expanding','blend']}
        result['groups'][group]=out
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    f=lambda x:f"{x['mean']:+.2f} [{x['ci95'][0]:+.2f},{x['ci95'][1]:+.2f}]"
    for g,o in result['groups'].items():
        print(f'== {g} (h={h}) baselines',{k:round(v,2) for k,v in o['baseline_energy'].items()})
        for k,v in o['shrinkage'].items():print('  ',k,f(v) if 'mean' in v else '  '.join(f"{n} {f(x)}" for n,x in v.items()))
        for k,v in o['bootstrap'].items():print('  ',k,'seeds+anchors',f(v['anchors_and_seeds']),'| pool-clustered',f(v['pool_clustered']) if 'pool_clustered' in v else '-','| per pool',{p:round(x,2) for p,x in v['per_pool'].items()})
        print('   endpoint CRPS (scaled)',{k:{n:round(x,2) for n,x in v.items()} for k,v in o['endpoint_crps'].items()})


if __name__=='__main__':main()
