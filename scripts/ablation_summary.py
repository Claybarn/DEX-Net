"""Per-pool rollout energy, per-endpoint CRPS and liquidity-collapse shares for several families.

For each family and anchor (common anchors only), a seed's forecast is its completed paths
(at least 16), scored with the unbiased ensemble energy; per-endpoint CRPS is divided by the
energy-score scales. A path collapses if it loses more than 63% of active liquidity
(log-liquidity change below -1); the real share uses each anchor's realized outcome.

  python -m scripts.ablation_summary --rollouts depth=eval/final/rollouts-test-depth \
      --rollouts reference=... --rollouts mech=... --horizon 512 --out results/ablation-summary-test.json
"""
import argparse,json
from pathlib import Path
import numpy as np
from dexnet.diagnostics import ENDPOINT_SCALES,proper_scores
from scripts.rollout_table import load


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rollouts',action='append',required=True,help='family=folder (repeatable)')
    p.add_argument('--horizon',default='512');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    fams=dict(x.split('=',1) for x in a.rollouts);data={f:load(d) for f,d in fams.items()};h=a.horizon
    keys=sorted(set.intersection(*(set(d) for d in data.values())))
    energy={};crps={};collapse={}
    for key in keys:
        r0=data[next(iter(fams))][key];group='heldout' if r0['split'].startswith('heldout') else 'seen'
        y=np.asarray(r0['actual'][h],float)
        for scope in (group,key[0]):collapse.setdefault(scope,{}).setdefault('real',[]).append(bool(y[1]<-1))
        for f in fams:
            seeds=[]
            for label,paths in data[f][key]['paths'].items():
                if label.startswith('baseline:'):continue
                x=np.array([q['values'][h] for q in paths if h in q['values']],float)
                for scope in (group,key[0]):collapse[scope].setdefault(f,[]).extend((x[:,1]<-1).tolist())
                if len(x)>=16:
                    e,c=proper_scores(x,y);seeds.append((e,c/ENDPOINT_SCALES))
            if not seeds:continue
            e=float(np.mean([s[0] for s in seeds]));c=np.mean([s[1] for s in seeds],0)
            for scope in (group,key[0]):
                energy.setdefault(scope,{}).setdefault(f,[]).append(e);crps.setdefault(scope,{}).setdefault(f,[]).append(c)
    result=dict(method=__doc__,horizon=int(h),families=list(fams),endpoints=['return_bp','log_liquidity','log_blocks','log_volume'],scopes={})
    for scope in energy:
        result['scopes'][scope]=dict(anchors=len(energy[scope][next(iter(fams))]),
            energy={f:float(np.mean(v)) for f,v in energy[scope].items()},
            scaled_crps={f:np.mean(v,0).tolist() for f,v in crps[scope].items()},
            collapse_share={f:float(np.mean(v)) for f,v in collapse[scope].items()})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for scope,v in result['scopes'].items():
        print(scope,v['anchors'],{f:round(e,2) for f,e in v['energy'].items()},'collapse%',{f:round(100*c,1) for f,c in v['collapse_share'].items()})


if __name__=='__main__':main()
