"""Held-out per-pool rollout energy (DEX-Net, no-profile, small, recent) and liquidity-collapse shares; usage: python scripts/heldout_per_pool.py <experiment dir>."""
import json,glob,sys
import numpy as np
sys.path.insert(0,'.')
from dexnet.diagnostics import proper_scores
E=sys.argv[1]
for split in ('test','val'):
    acc={}
    for f in sorted(glob.glob(f'{E}/final/rollouts-{split}-depth/anchors/*.json')):
        n=f.split('/')[-1];R={fam:json.load(open(f'{E}/final/rollouts-{split}-{fam}/anchors/{n}')) for fam in ('depth','reference','depth-small')}
        r0=R['depth']
        if not r0['split'].startswith('heldout'):continue
        for h in ('32','128','512'):
            y=np.asarray(r0['actual'][h],float);d=acc.setdefault((r0['pool'],h),{})
            for fam,r in R.items():
                d.setdefault(fam,[]).append(np.mean([proper_scores(np.array([q['values'][h] for q in ps if h in q['values']]),y)[0] for lab,ps in r['paths'].items() if not lab.startswith('baseline:')]))
            d.setdefault('recent',[]).append(proper_scores(np.array([q['values'][h] for q in r0['paths']['baseline:recent']]),y)[0])
            d.setdefault('collapse_real',[]).append(float(y[1]<-1))
            d.setdefault('collapse_depth',[]).append(float(np.mean([q['values'][h][1]<-1 for lab,ps in R['depth']['paths'].items() if not lab.startswith('baseline:') for q in ps if h in q['values']])))
            d.setdefault('collapse_ref',[]).append(float(np.mean([q['values'][h][1]<-1 for lab,ps in R['reference']['paths'].items() if not lab.startswith('baseline:') for q in ps if h in q['values']])))
    for (pool,h),d in sorted(acc.items()):
        print(split,pool,h,'n',len(d['depth']),' '.join(f"{k} {np.mean(v):.2f}" for k,v in d.items()))
