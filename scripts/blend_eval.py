"""Forecast blend: a mixture of the model's paths and the recent-history baseline's windows.

At each anchor and horizon the blended forecast is the mixture alpha * model + (1 - alpha) * recent.
Its energy score (scaled endpoints, as in Table 4) is exact for a two-component mixture:

  E(alpha) = alpha a_M + (1 - alpha) a_B - 1/2 [alpha^2 S_MM + 2 alpha (1 - alpha) S_MB + (1 - alpha)^2 S_BB]

with a_M = mean |M - y|, S_MM the fair (off-diagonal) mean |M - M'|, S_MB the mean over all
model/baseline pairs, and likewise for B; it reduces to the fair model and baseline scores at
alpha = 1 and 0. A seed's forecast is its completed paths; terms are averaged over seeds.

alpha is fitted per horizon on the --fit rollouts (seen pools only) by minimizing the mean
score (a convex quadratic), either globally or per third of the regime ratio of
scripts.regime_analysis (recent 5-95% width of h-event returns over the expanding one, known
at the anchor; thirds from the fit anchors). It is then applied unchanged to the --eval
rollouts, seen and held-out pools, with paired bootstrap CIs over anchors.

  python -m scripts.blend_eval --fit eval/final/rollouts-val-depth --eval eval/final/rollouts-test-depth --out results/blend-test.json
"""
import argparse,json
from pathlib import Path
import numpy as np
from dexnet.diagnostics import ENDPOINT_SCALES

HORIZONS=('32','128','512')


def pair_mean(a,b,fair=False):
    d=np.linalg.norm(a[:,None]-b[None],axis=-1)
    return d.sum()/(len(a)*(len(a)-1)) if fair else d.mean()


def width(v):return float(np.subtract(*np.quantile(v,[.95,.05])))


def terms(folder,min_paths=16):
    """{(pool,row,h): dict(group, ratio, a_M, a_B, S_MM, S_MB, S_BB)} with model terms averaged over seeds."""
    out={}
    for f in sorted((Path(folder)/'anchors').glob('*.json')):
        r=json.loads(f.read_text());group='heldout' if r['split'].startswith('heldout') else 'seen'
        for h in HORIZONS:
            y=np.asarray(r['actual'][h],float)/ENDPOINT_SCALES
            B=np.array([q['values'][h] for q in r['paths']['baseline:recent']],float)/ENDPOINT_SCALES
            ex=np.array([q['values'][h] for q in r['paths']['baseline:expanding']],float)
            seeds=[]
            for label,paths in r['paths'].items():
                if label.startswith('baseline:'):continue
                M=np.array([q['values'][h] for q in paths if h in q['values']],float)
                if len(M)<min_paths:continue
                M=M/ENDPOINT_SCALES
                seeds.append([np.linalg.norm(M-y,axis=-1).mean(),pair_mean(M,M,True),pair_mean(M,B)])
            if not seeds:continue
            aM,SMM,SMB=np.mean(seeds,0)
            out[(r['pool'],r['row'],h)]=dict(group=group,ratio=width(B[:,0]*ENDPOINT_SCALES[0])/max(width(ex[:,0]),1e-9),
                a_M=aM,a_B=np.linalg.norm(B-y,axis=-1).mean(),S_MM=SMM,S_MB=SMB,S_BB=pair_mean(B,B,True))
    return out


def score(t,alpha):
    return alpha*t['a_M']+(1-alpha)*t['a_B']-.5*(alpha**2*t['S_MM']+2*alpha*(1-alpha)*t['S_MB']+(1-alpha)**2*t['S_BB'])


def fit_alpha(rows):
    """Minimizer on [0, 1] of the mean score: E = c0 + c1 alpha + c2 alpha^2."""
    m={k:np.mean([t[k] for t in rows]) for k in ('a_M','a_B','S_MM','S_MB','S_BB')}
    c1=m['a_M']-m['a_B']-(m['S_MB']-m['S_BB']);c2=m['S_MB']-.5*(m['S_MM']+m['S_BB'])
    return float(np.clip(-c1/(2*c2),0,1)) if c2>0 else float(c1<0)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--eval',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rng=np.random.default_rng(0)
    fit,ev=terms(a.fit),terms(a.eval);result=dict(method=__doc__,fit=str(a.fit),eval=str(a.eval),alphas={},cells={})
    for h in HORIZONS:
        F=[t for (pool,row,hh),t in fit.items() if hh==h and t['group']=='seen']
        q=np.quantile([t['ratio'] for t in F],[1/3,2/3]);third=lambda t:int(np.searchsorted(q,t['ratio']))
        glob=fit_alpha(F);by=[fit_alpha([t for t in F if third(t)==k]) for k in range(3)]
        result['alphas'][h]=dict(global_alpha=glob,regime_thresholds=q.tolist(),regime_alpha=by)
        for group in ('seen','heldout'):
            T=[t for (pool,row,hh),t in ev.items() if hh==h and t['group']==group]
            s=dict(model=np.array([score(t,1.) for t in T]),recent=np.array([score(t,0.) for t in T]),
                   blend=np.array([score(t,glob) for t in T]),blend_regime=np.array([score(t,by[third(t)]) for t in T]))
            idx=rng.integers(0,len(T),(4000,len(T)))
            diff=lambda x,y:dict(mean=float((x-y).mean()),relative=float((x-y).mean()/y.mean()),ci95=[float(v) for v in np.quantile((x-y)[idx].mean(1),[.025,.975])])
            result['cells'][f'{group}:{h}']=dict(anchors=len(T),energy={k:float(v.mean()) for k,v in s.items()},
                differences={f'{b}-{base}':diff(s[b],s[base]) for b in ('blend','blend_regime') for base in ('model','recent')})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for h,v in result['alphas'].items():print('alpha',h,round(v['global_alpha'],3),'by regime',[round(x,3) for x in v['regime_alpha']])
    for k,c in result['cells'].items():
        e=c['energy'];print(k,c['anchors'],' '.join(f'{n} {x:.2f}' for n,x in e.items()),'|',
              ' '.join(f"{n} {d['mean']:+.2f} ({100*d['relative']:+.1f}%) [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]" for n,d in c['differences'].items()))


if __name__=='__main__':main()
