"""LP-facing evaluation: does the simulator predict what a liquidity provider earns?

At each anchor a hypothetical marginal position covers [p0 e^-w, p0 e^+w] around the
anchor price p0 (in the economic base/quote orientation). For any price path the
position's outcome after h events follows from the path alone, because a marginal
position's share of each swap's fee does not depend on other liquidity:

  fee yield        fees earned by the position (fee tier f; a swap moving sqrt(p) from a
                   to b pays f/(1-f) * L * |clip(b) - clip(a)| in quote when the price
                   rises, and f/(1-f) * L * |1/clip(b) - 1/clip(a)| in base when it
                   falls, with clip() to the range), valued at the final price
  divergence       position value minus the value of holding its initial tokens, at the
                   final price (impermanent loss; always <= 0)
  P&L vs holding   fee yield + divergence

all in bp of the initial position value, gross of any protocol fee share. Simulated
paths use their stored per-event price traces; the realised outcome uses the anchor's
realised trace; the two history baselines use the same window rules as
scripts.prepare_anchors (recent: 256 evenly spaced windows from the last 32h events;
expanding: 256 windows from all earlier history) on the pool's realised price returns.
Each forecaster is scored by CRPS per metric (mean over its seeds for models);
differences against the recent baseline are bootstrapped over anchors.

  python -m scripts.lp_eval --anchors eval/anchors-test --data data/prepared \
      --rollouts depth=eval/final/rollouts-test-depth --rollouts reference=... --out results/lp-test.json
"""
import argparse,json
from pathlib import Path
import numpy as np

HORIZONS=('32','128','512');WIDTHS=(.01,.002);METRICS=('fee_bp','divergence_bp','pnl_bp')


def outcomes(logp,f,w):
    """logp: log price relative to the anchor, after each event (n,). Returns fee, divergence, P&L in bp."""
    s=np.exp(np.r_[0.,logp]/2);lo,hi=np.exp(-w/2),np.exp(w/2)      # sqrt price relative to the anchor; range edges
    c=np.clip(s,lo,hi);up=np.diff(c).clip(min=0).sum();down=np.diff(1/c).clip(min=0).sum()
    k=f/(1-f);p_end=s[-1]**2
    x0=1/1-1/hi;y0=1-lo                                            # initial base and quote per unit liquidity at sqrt p0 = 1
    ce=c[-1];x1=1/ce-1/hi;y1=ce-lo
    v0=x0*1+y0
    fee=k*up+k*down*p_end
    div=(x1*p_end+y1)-(x0*p_end+y0)
    return 1e4*fee/v0,1e4*div/v0,1e4*(fee+div)/v0


def crps(x,y):
    x=np.asarray(x,float);n=len(x)
    return float(np.abs(x-y).mean()-np.abs(x[:,None]-x[None]).sum()/(2*n*(n-1)))


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--anchors',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--rollouts',action='append',required=True,help='family=folder');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();fams=dict(x.split('=',1) for x in a.rollouts);rng=np.random.default_rng(0)
    files={f:{x.name:x for x in (Path(d)/'anchors').glob('*.json')} for f,d in fams.items()}
    names=sorted(set.intersection(*(set(v) for v in files.values())));lr={};rows=[]
    for name in names:
        R={f:json.loads(files[f][name].read_text()) for f in fams};r0=next(iter(R.values()));pool=r0['pool'];row=r0['row']
        fee=json.loads((a.data/'pools'/pool/'metadata.json').read_text())['fee']/1e6
        if pool not in lr:lr[pool]=np.load(a.data/'pools'/pool/'raw_features.npy',mmap_mode='r')[:,1].astype(np.float64)
        ret=lr[pool];real=np.asarray(r0['actual_trace']['return_bp'],float)/1e4
        # Realised path from the trace must equal the prepared per-event returns.
        if not np.allclose(real,np.cumsum(ret[row:row+len(real)]),atol=1e-6):raise ValueError(f'{name}: trace and prepared returns disagree')
        rec=dict(pool=pool,row=row,group='heldout' if r0['split'].startswith('heldout') else 'seen',cells={})
        for h in HORIZONS:
            H=int(h)
            if len(real)<H:continue
            starts={'recent':np.unique(np.linspace(max(1,row-32*H),row-H,256).astype(int)),'expanding':rng.integers(1,row-H+1,256)}
            for w in WIDTHS:
                y=outcomes(real[:H],fee,w);cell={}
                for f,r in R.items():
                    seeds=[]
                    for lab,ps in r['paths'].items():
                        if lab.startswith('baseline:'):continue
                        sims=np.array([outcomes(np.asarray(q['trace']['return_bp'][:H],float)/1e4,fee,w) for q in ps if q['trace'] and len(q['trace']['return_bp'])>=H])
                        if len(sims)>=16:seeds.append([crps(sims[:,m],y[m]) for m in range(3)]+[float(np.median(sims[:,2])-y[2]),float(np.mean((np.quantile(sims[:,2],.05)<=y[2])&(y[2]<=np.quantile(sims[:,2],.95))))])
                    if seeds:cell[f]=np.mean(seeds,0).tolist()
                for b,st in starts.items():
                    st=st[(st>=1)&(st+H<=row)]
                    sims=np.array([outcomes(np.cumsum(ret[s0:s0+H]),fee,w) for s0 in st])
                    cell[b]=[crps(sims[:,m],y[m]) for m in range(3)]+[float(np.median(sims[:,2])-y[2]),float(np.mean((np.quantile(sims[:,2],.05)<=y[2])&(y[2]<=np.quantile(sims[:,2],.95))))]
                cell['real']=list(y)
                rec['cells'][f'{h}:{w}']=cell
        rows.append(rec)
    result=dict(method=__doc__,anchors=len(rows),families=list(fams),cells={})
    for group in ('seen','heldout'):
        for h in HORIZONS:
            for w in WIDTHS:
                key=f'{h}:{w}';rs=[r['cells'][key] for r in rows if r['group']==group and key in r['cells'] and all(f in r['cells'][key] for f in fams)]
                if not rs:continue
                forecasters=list(fams)+['recent','expanding'];out=dict(anchors=len(rs),real_median={m:float(np.median([r['real'][i] for r in rs])) for i,m in enumerate(METRICS)},scores={})
                for fc in forecasters:
                    v=np.array([r[fc] for r in rs]);out['scores'][fc]=dict(**{f'crps_{m}':float(v[:,i].mean()) for i,m in enumerate(METRICS)},pnl_median_bias=float(v[:,3].mean()),pnl_coverage90=float(v[:,4].mean()))
                diffs={}
                for fc in fams:
                    d=np.array([r[fc][2]-r['recent'][2] for r in rs]);idx=rng.integers(0,len(d),(4000,len(d)))
                    diffs[f'{fc}-recent']=dict(mean=float(d.mean()),ci95=[float(x) for x in np.quantile(d[idx].mean(1),[.025,.975])])
                out['pnl_crps_difference']=diffs
                result['cells'][f'{group}:{h}:{w}']=out
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for k,c in result['cells'].items():
        print(k,c['anchors'],'| P&L CRPS '+' '.join(f"{f} {v['crps_pnl_bp']:.2f}" for f,v in c['scores'].items()),'| cov90 '+' '.join(f"{f} {v['pnl_coverage90']:.2f}" for f,v in c['scores'].items()))


if __name__=='__main__':main()
