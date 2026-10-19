"""Where does the model beat the recent-history baseline? Score differences by volatility regime.

For each anchor and horizon h: the model's fair ensemble energy (mean over seeds) minus
the recent baseline's, scored identically. Anchors are split into thirds by

  regime    recent volatility relative to the pool's long-run volatility: the 5-95%
            width of h-event returns in the recent baseline windows divided by that in
            the expanding (all-history) windows; known at the anchor
  surprise  |realized h-event return| / recent 5-95% width; uses the outcome, so it is
            a diagnostic only

and the mean difference per third is bootstrapped over anchors.

  python -m scripts.regime_analysis --rollouts eval/final/rollouts-test-depth --out results/regime-test.json
"""
import argparse,json
from pathlib import Path
import numpy as np
from dexnet.diagnostics import proper_scores


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rng=np.random.default_rng(0);rows=[]
    for f in sorted((a.rollouts/'anchors').glob('*.json')):
        r=json.loads(f.read_text())
        for h in ('32','128','512'):
            y=np.asarray(r['actual'][h],float)
            seeds=[proper_scores(np.array([q['values'][h] for q in ps if h in q['values']]),y)[0] for lab,ps in r['paths'].items() if not lab.startswith('baseline:')]
            rec=np.array([q['values'][h] for q in r['paths']['baseline:recent']]);ex=np.array([q['values'][h] for q in r['paths']['baseline:expanding']])
            width=lambda v:float(np.subtract(*np.quantile(v,[.95,.05])))
            pre=width(rec[:,0])
            rows.append(dict(pool=r['pool'],row=r['row'],group='heldout' if r['split'].startswith('heldout') else 'seen',h=h,
                             difference=float(np.mean(seeds))-proper_scores(rec,y)[0],regime=pre/max(width(ex[:,0]),1e-9),surprise=abs(y[0])/max(pre,1e-9)))
    result=dict(method=__doc__,cells={})
    for group in ('seen','heldout'):
        for h in ('32','128','512'):
            R=[x for x in rows if x['group']==group and x['h']==h]
            for feat in ('regime','surprise'):
                v=np.array([x[feat] for x in R]);d=np.array([x['difference'] for x in R]);q=np.quantile(v,[1/3,2/3]);cell={}
                for name,m in (('low',v<=q[0]),('mid',(v>q[0])&(v<=q[1])),('high',v>q[1])):
                    idx=rng.integers(0,m.sum(),(4000,m.sum()))
                    cell[name]=dict(anchors=int(m.sum()),mean=float(d[m].mean()),ci95=[float(x) for x in np.quantile(d[m][idx].mean(1),[.025,.975])],
                                    feature_range=[float(v[m].min()),float(v[m].max())])
                result['cells'][f'{group}:{h}:{feat}']=cell
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for k,c in result['cells'].items():print(k,'  '.join(f"{n} {v['mean']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]" for n,v in c.items()))


if __name__=='__main__':main()
