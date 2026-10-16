"""Forecast calibration, paired comparisons and asymmetry tests from rollout_eval output.

For every anchor, horizon and forecaster (model seeds and history-only baselines):

  scores     unbiased energy score (four endpoints) and CRPS of the price return;
             paired differences against each baseline with a bootstrap over anchors
             (anchors are spread across the test period and do not overlap)
  PIT        where the realised return falls in the forecast ensemble; a calibrated
             forecaster gives uniform PIT. Reported as a 10-bin histogram, the
             share in the outer 10% on each side, and a chi-square statistic
  asymmetry  the forecast's quantile skew (q90 + q10 - 2 q50) / (q90 - q10). Anchors
             are split by the sign and size of predicted skew; within each group we
             compare predicted and realised upside and downside tail frequencies.
             Informative asymmetric intervals put more realised breakouts on the side
             the forecast says is longer.

  python -m scripts.calibration --rollouts eval/rollouts-test --out results/calibration.json
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from dexnet.diagnostics import proper_scores


def quantile_skew(x):
    q10,q50,q90=np.quantile(x,[.1,.5,.9])
    return float((q90+q10-2*q50)/(q90-q10)) if q90>q10 else 0.


def bootstrap_mean(diff,rng,draws=4000):
    d=np.asarray(diff);idx=rng.integers(0,len(d),(draws,len(d)))
    return dict(mean=float(d.mean()),ci95=[float(x) for x in np.quantile(d[idx].mean(1),[.025,.975])],n=len(d))


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();rng=np.random.default_rng(a.seed)
    rows=defaultdict(list)      # (group, horizon) -> per-anchor dict of label -> stats
    for f in sorted((a.rollouts/'anchors').glob('*.json')):
        r=json.loads(f.read_text());group='heldout' if r['split'].startswith('heldout') else 'seen'
        for h,y in r['actual'].items():
            y=np.asarray(y,float);entry={}
            for label,paths in r['paths'].items():
                x=np.array([q['values'][h] for q in paths if h in q['values']],float)
                if len(x)<len(paths) or len(x)<4:entry=None;break      # compare only anchors complete for every forecaster
                es,crps=proper_scores(x,y);ret=x[:,0]
                lo,hi=np.quantile(ret,[.1,.9])
                entry[label]=dict(energy=es,crps=float(crps[0]),pit=float((np.sum(ret<y[0])+.5*np.sum(ret==y[0]))/len(ret)),
                                  skew=quantile_skew(ret),above=float(y[0]>hi),below=float(y[0]<lo),pool=r['pool'])
            if entry:rows[(group,int(h))].append(entry)
    result=dict(method=__doc__,cells={})
    for (group,h),entries in sorted(rows.items()):
        labels=sorted(entries[0]);models=[l for l in labels if not l.startswith('baseline:')];baselines=[l for l in labels if l.startswith('baseline:')]
        cell=dict(anchors=len(entries),forecasters={})
        for l in labels:
            e=[x[l] for x in entries];pit=np.array([x['pit'] for x in e]);hist=np.histogram(pit,bins=10,range=(0,1))[0]
            expected=len(pit)/10;skew=np.array([x['skew'] for x in e]);above=np.array([x['above'] for x in e]);below=np.array([x['below'] for x in e])
            groups={}
            for name,mask in [('up_skew',skew>.1),('symmetric',np.abs(skew)<=.1),('down_skew',skew<-.1)]:
                if mask.sum()>=10:
                    groups[name]=dict(anchors=int(mask.sum()),mean_predicted_skew=float(skew[mask].mean()),
                                      realised_above_q90=float(above[mask].mean()),realised_below_q10=float(below[mask].mean()))
            cell['forecasters'][l]=dict(mean_energy=float(np.mean([x['energy'] for x in e])),mean_crps_bp=float(np.mean([x['crps'] for x in e])),
                pit_histogram=hist.tolist(),pit_chi2=float(((hist-expected)**2/expected).sum()),
                outer_10=dict(below=float(below.mean()),above=float(above.mean())),
                skew_spread=float(np.std(skew)),skew_groups=groups,
                skew_tail_correlation=float(np.corrcoef(skew,above-below)[0,1]) if np.std(skew)>0 and np.std(above-below)>0 else None)
        for m in models:
            for b in baselines:
                for metric in ('energy','crps'):
                    diff=[x[m][metric]-x[b][metric] for x in entries]
                    cell.setdefault('paired',{})[f'{m} - {b} ({metric})']=bootstrap_mean(diff,rng)
        if len(models)>1:
            for metric in ('energy','crps'):
                seed_mean=[np.mean([x[m][metric] for m in models]) for x in entries]
                for b in baselines:
                    cell['paired'][f'seed mean - {b} ({metric})']=bootstrap_mean(np.array(seed_mean)-np.array([x[b][metric] for x in entries]),rng)
        result['cells'][f'{group}:{h}']=cell
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for key,cell in result['cells'].items():
        print(f"\n== {key}  ({cell['anchors']} anchors)")
        print(f"{'forecaster':22s} {'energy':>8s} {'CRPS bp':>8s} {'PIT chi2':>9s} {'<q10':>6s} {'>q90':>6s} {'skew sd':>8s} {'skew~tail r':>11s}")
        for l,v in cell['forecasters'].items():
            c=v['skew_tail_correlation']
            print(f"{l:22s} {v['mean_energy']:8.3f} {v['mean_crps_bp']:8.2f} {v['pit_chi2']:9.1f} {v['outer_10']['below']:6.3f} {v['outer_10']['above']:6.3f} {v['skew_spread']:8.3f} {('%.3f'%c) if c is not None else 'n/a':>11s}")
        for k,v in cell.get('paired',{}).items():
            if k.startswith('seed mean'):print(f"  {k:48s} {v['mean']:+8.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")


if __name__=='__main__':main()
