"""Does predicted skew get the direction right where it matters?

Uses rollout_eval output plus the anchors (for full baseline samples). Skew is the
quantile skew (q90 + q10 - 2 q50) / (q90 - q10) of the forecast price return.
Skew and the q10/q90 tail thresholds always come from disjoint draws:

  model      skew from two seeds (64 paths), thresholds from the third (32 paths);
             all three rotations are averaged
  baselines  skew from 128 history windows, thresholds from the other 128

Two conditional views, per horizon and pool group:

  confident  anchors in the top or bottom 20% of predicted skew for that forecaster:
             do realised breakouts land on the predicted long side?
  trend      anchors after the largest prior moves, the top/bottom 20% of the real
             return over the h events before the anchor, scaled by that anchor's
             typical h-event move: does predicted skew match what followed?

Reported: realised share above q90 and below q10, breakout direction share
up/(up+down), P(return > 0), mean predicted skew, and the correlation between
predicted skew and realised tail asymmetry (above - below) with a bootstrap CI.

  python -m scripts.skew_analysis --rollouts eval/rollouts-test --anchors eval/anchors-test --out results/skew.json
"""
import argparse,json
from pathlib import Path
import numpy as np


def qskew(x):
    q10,q50,q90=np.quantile(x,[.1,.5,.9])
    return float((q90+q10-2*q50)/(q90-q10)) if q90>q10 else 0.


def split_stats(skew_draws,threshold_draws,y):
    lo,hi=np.quantile(threshold_draws,[.1,.9])
    return qskew(skew_draws),float(y>hi),float(y<lo)


def summarize(rows,rng):
    """rows: list of (skew, above, below, y)."""
    if len(rows)<5:return None
    s,a,b,y=map(np.asarray,zip(*rows));up,down=a.sum(),b.sum()
    out=dict(anchors=len(rows),mean_predicted_skew=float(s.mean()),realised_above_q90=float(a.mean()),realised_below_q10=float(b.mean()),
             breakout_up_share=float(up/(up+down)) if up+down else None,p_return_positive=float((y>0).mean()))
    return out


def correlation(rows,rng,draws=2000):
    s,a,b,_=map(np.asarray,zip(*rows));t=a-b
    if np.std(s)==0 or np.std(t)==0:return None
    r=float(np.corrcoef(s,t)[0,1]);idx=rng.integers(0,len(s),(draws,len(s)))
    boot=[np.corrcoef(s[i],t[i])[0,1] for i in idx if np.std(s[i])>0 and np.std(t[i])>0]
    return dict(r=r,ci95=[float(x) for x in np.quantile(boot,[.025,.975])],n=len(s))


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--anchors',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fraction',type=float,default=.2);a=p.parse_args();rng=np.random.default_rng(0)
    data={}            # (group, h) -> forecaster -> list of (skew, above, below, y, trend)
    for f in sorted((a.rollouts/'anchors').glob('*.json')):
        r=json.loads(f.read_text());group='heldout' if r['split'].startswith('heldout') else 'seen'
        anchor=json.loads((a.anchors/r['pool']/f"{r['row']}.json").read_text())
        models=sorted(l for l in r['paths'] if not l.startswith('baseline:'))
        for h,actual in r['actual'].items():
            y=float(actual[0]);recent=anchor['baselines']['recent'].get(h) or anchor['baselines']['recent'].get(str(h))
            if not recent or len(recent)<10:continue
            prior=np.asarray(recent,float)[:,0];scale=np.median(np.abs(prior-np.median(prior)))
            if scale<=0:continue
            trend=float(prior[-1]/scale)          # last recent window is the h events just before the anchor
            cell=data.setdefault((group,int(h)),{})
            draws={m:np.array([q['values'][h][0] for q in r['paths'][m] if h in q['values']],float) for m in models}
            if all(len(v)>=16 for v in draws.values()) and len(models)==3:
                rot=[split_stats(np.r_[draws[models[i]],draws[models[j]]],draws[models[k]],y) for i,j,k in ((0,1,2),(0,2,1),(1,2,0))]
                s,ab,be=np.mean(rot,0);cell.setdefault('model (3 seeds)',[]).append((s,ab,be,y,trend))
            for name,samples in anchor['baselines'].items():
                x=np.asarray(samples.get(h) or samples.get(str(h)) or [],float)
                if len(x)<32:continue
                x=x[rng.permutation(len(x))][:,0];half=len(x)//2
                s,ab,be=split_stats(x[:half],x[half:],y);cell.setdefault('baseline:'+name,[]).append((s,ab,be,y,trend))
    result=dict(method=__doc__,cells={})
    for (group,h),by in sorted(data.items()):
        cell={}
        for label,rows in by.items():
            rows=np.asarray(rows,float);s=rows[:,0];trend=rows[:,4];k=max(5,int(len(rows)*a.fraction))
            take=lambda m:[tuple(x[:4]) for x in rows[m]]
            top=np.argsort(-s)[:k];bottom=np.argsort(s)[:k];up=np.argsort(-trend)[:k];down=np.argsort(trend)[:k]
            cell[label]=dict(all=summarize(take(slice(None)),rng),skew_tail_correlation=correlation(take(slice(None)),rng),
                confident_up_skew=summarize(take(top),rng),confident_down_skew=summarize(take(bottom),rng),
                after_up_trend=summarize(take(up),rng),after_down_trend=summarize(take(down),rng),
                trend_skew_correlation=float(np.corrcoef(trend,s)[0,1]) if np.std(s)>0 else None)
        result['cells'][f'{group}:{h}']=cell
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    fmt=lambda d:(f"{d['realised_above_q90']:.3f}/{d['realised_below_q10']:.3f} up {d['breakout_up_share']:.2f} skew {d['mean_predicted_skew']:+.2f}" if d and d['breakout_up_share'] is not None else 'n/a')
    for key,cell in result['cells'].items():
        print(f'\n== {key}')
        for label,v in cell.items():
            c=v['skew_tail_correlation']
            print(f"{label:22s} skew~tail r {c['r']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]  trend~skew r {v['trend_skew_correlation']:+.3f}" if c else f'{label:22s} n/a')
            print(f"{'':22s} confident up-skew   {fmt(v['confident_up_skew'])}")
            print(f"{'':22s} confident down-skew {fmt(v['confident_down_skew'])}")
            print(f"{'':22s} after up-trend      {fmt(v['after_up_trend'])}   P(ret>0) {v['after_up_trend']['p_return_positive']:.2f}")
            print(f"{'':22s} after down-trend    {fmt(v['after_down_trend'])}   P(ret>0) {v['after_down_trend']['p_return_positive']:.2f}")


if __name__=='__main__':main()
