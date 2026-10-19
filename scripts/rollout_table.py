"""Rollout energy for several model families and the history baselines on ONE common anchor set.

Per split, pool group and horizon, an anchor enters only if every path of every model
seed reached that horizon (the same set for every family and baseline, so differences
are paired). Each family's score at an anchor is the mean over its seeds of the
unbiased ensemble energy of that seed's paths; baselines are scored on the same
anchors. Paired differences between families are bootstrapped over anchors.

  python -m scripts.rollout_table --rollouts depth=eval/final/rollouts-test-depth \
      --rollouts reference=eval/final/rollouts-test-reference --out results/rollout-table-test.json
"""
import argparse,json
from pathlib import Path
import numpy as np
from dexnet.diagnostics import proper_scores


def load(folder):
    return {(r['pool'],r['row']):r for r in (json.loads(f.read_text()) for f in sorted((Path(folder)/'anchors').glob('*.json')))}


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rollouts',action='append',required=True,help='family=folder (repeatable; first family is the reference for differences)')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    fams=dict(x.split('=',1) for x in a.rollouts);data={f:load(d) for f,d in fams.items()};rng=np.random.default_rng(0)
    keys=sorted(set.intersection(*(set(d) for d in data.values())));first=next(iter(data))
    result=dict(method=__doc__,families=list(fams),cells={})
    for group in ('seen','heldout'):
        gk=[k for k in keys if data[first][k]['split'].startswith('heldout')==(group=='heldout')]
        for h in sorted({h for k in gk for h in data[first][k]['actual']},key=int):
            ok=[k for k in gk if all(h in p['values'] for d in data.values() for lab,ps in d[k]['paths'].items() if not lab.startswith('baseline:') for p in ps)]
            score={}
            for f,d in data.items():
                score[f]=np.array([np.mean([proper_scores(np.array([p['values'][h] for p in ps]),np.asarray(d[k]['actual'][h]))[0]
                                            for lab,ps in d[k]['paths'].items() if not lab.startswith('baseline:')]) for k in ok])
            for b in ('recent','expanding'):
                score[b]=np.array([proper_scores(np.array([p['values'][h] for p in data[first][k]['paths'][f'baseline:{b}']]),np.asarray(data[first][k]['actual'][h]))[0] for k in ok])
            diffs={}
            for f in list(fams)[1:]:
                for g in [x for x in score if x!=f]:
                    d=score[f]-score[g];idx=rng.integers(0,len(d),(4000,len(d)))
                    diffs[f'{f}-{g}']=dict(mean=float(d.mean()),ci95=[float(x) for x in np.quantile(d[idx].mean(1),[.025,.975])])
            result['cells'][f'{group}:{h}']=dict(anchors=len(ok),anchors_in_group=len(gk),energy={k:float(v.mean()) for k,v in score.items()},differences=diffs)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for k,c in result['cells'].items():print(k,f"{c['anchors']}/{c['anchors_in_group']}",' '.join(f'{f} {v:.2f}' for f,v in c['energy'].items()))


if __name__=='__main__':main()
