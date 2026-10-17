"""Head-to-head comparison of two model families evaluated on the same anchors.

Reads two scripts.rollout_eval output folders built from the same anchor set (for
example the reference model and a variant, each with several seeds). Per anchor and
horizon, each family's score is the mean over its seeds of the unbiased ensemble
energy (four endpoints) and the price CRPS; family B minus family A is bootstrapped
over anchors. Also reported per family, against the realised outcomes at the same
anchors:

  collapse  share of paths whose active liquidity falls by more than 63%
            (log change < -1), and the 1st percentile of the log change
  spread    width of the 5-95% range of price outcomes, simulated / realised

  python -m scripts.compare_rollouts --a eval/rollouts-test --b eval/rollouts-test-depth \
      --label-a reference --label-b depth --out results/compare.json
"""
import argparse,json
from pathlib import Path
import numpy as np
from dexnet.diagnostics import proper_scores


def load(folder):
    out={}
    for f in sorted((folder/'anchors').glob('*.json')):
        r=json.loads(f.read_text());out[(r['pool'],r['row'])]=r
    return out


def family_scores(r,h):
    """Seed-mean energy and price CRPS at horizon h, or None if any seed's ensemble is incomplete."""
    y=np.asarray(r['actual'][h],float);e=[];c=[]
    for label,paths in r['paths'].items():
        if label.startswith('baseline:'):continue
        x=np.array([p['values'][h] for p in paths if h in p['values']],float)
        if len(x)<len(paths) or len(x)<4:return None
        es,crps=proper_scores(x,y);e.append(es);c.append(crps[0])
    return float(np.mean(e)),float(np.mean(c))


def endpoints(r,h,dim):
    return [p['values'][h][dim] for label,paths in r['paths'].items() if not label.startswith('baseline:') for p in paths if h in p['values']]


def bootstrap(d,rng,draws=4000):
    d=np.asarray(d);idx=rng.integers(0,len(d),(draws,len(d)))
    return dict(mean=float(d.mean()),ci95=[float(x) for x in np.quantile(d[idx].mean(1),[.025,.975])],n=len(d))


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--a',type=Path,required=True);p.add_argument('--b',type=Path,required=True)
    p.add_argument('--label-a',default='A');p.add_argument('--label-b',default='B');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rng=np.random.default_rng(0)
    A,B=load(a.a),load(a.b);keys=sorted(set(A)&set(B))
    result=dict(method=__doc__,anchors=len(keys),cells={})
    for group in ('seen','heldout'):
        gk=[k for k in keys if A[k]['split'].startswith('heldout')==(group=='heldout')]
        for h in sorted({h for k in gk for h in A[k]['actual']},key=int):
            sa,sb=[],[]
            for k in gk:
                x,y=family_scores(A[k],h),family_scores(B[k],h)
                if x and y:sa.append(x);sb.append(y)
            if not sa:continue
            sa,sb=np.array(sa),np.array(sb)
            cell=dict(anchors=len(sa),energy={a.label_a:float(sa[:,0].mean()),a.label_b:float(sb[:,0].mean()),'difference':bootstrap(sb[:,0]-sa[:,0],rng)},
                      crps_bp={a.label_a:float(sa[:,1].mean()),a.label_b:float(sb[:,1].mean()),'difference':bootstrap(sb[:,1]-sa[:,1],rng)})
            real_liq=np.array([A[k]['actual'][h][1] for k in gk]);real_ret=np.array([A[k]['actual'][h][0] for k in gk])
            width=lambda v:float(np.subtract(*np.quantile(v,[.95,.05])))
            cell['collapse']={'real':dict(share_lost_63pct=float(np.mean(real_liq<-1)),q01=float(np.quantile(real_liq,.01)))}
            cell['spread_ratio']={}
            for label,F in ((a.label_a,A),(a.label_b,B)):
                liq=np.concatenate([endpoints(F[k],h,1) for k in gk]);ret=np.concatenate([endpoints(F[k],h,0) for k in gk])
                cell['collapse'][label]=dict(share_lost_63pct=float(np.mean(liq<-1)),q01=float(np.quantile(liq,.01)))
                cell['spread_ratio'][label]=width(ret)/width(real_ret)
            result['cells'][f'{group}:{h}']=cell
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for key,c in result['cells'].items():
        e,r=c['energy']['difference'],c['crps_bp']['difference'];col=c['collapse']
        print(f"\n== {key} ({c['anchors']} anchors)  {a.label_b} - {a.label_a}: energy {e['mean']:+.3f} [{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}]  "
              f"CRPS {r['mean']:+.2f} bp [{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}]")
        print('   liquidity lost >63%: '+'  '.join(f"{k} {v['share_lost_63pct']:.3f}" for k,v in col.items())+'   | worst-1% log change: '+'  '.join(f"{k} {v['q01']:.2f}" for k,v in col.items()))
        print('   5-95% price spread vs real: '+'  '.join(f'{k} {v:.2f}x' for k,v in c['spread_ratio'].items()))


if __name__=='__main__':main()
