"""Stylized facts of simulated versus real event sequences after the same anchors.

Reads scripts.rollout_eval output (per-path traces and the realised trace stored with
each anchor). All statistics are computed identically for real and simulated
sequences, pooled over anchors, separately for seen and held-out pools:

  mix          share of sells, buys, mints, burns
  clock        share of events in the same block as the previous event; median and
               90th percentile block gap
  returns      per-event return (bp) lag-1 autocorrelation; excess kurtosis
  clustering   autocorrelation of |per-event return| at lags 1, 2, 5, 10
  persistence  P(swap side equals previous swap's side)
  sizes        median and 90th percentile of log swap size, centred on each pool's
               real median (pools have different units)
  volatility   standard deviation of cumulative return after h events, h = 1..512,
               and the simulated/real ratio (plus a robust 5-95% width ratio)

Paths that failed (an infeasible sampled event) contribute their events up to the
failure. Real statistics have one sequence per anchor; simulated, one per path.

  python -m scripts.stylized_facts --rollouts eval/rollouts-test --out results/stylized-facts.json
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

HORIZONS=(1,2,4,8,16,32,64,128,256,512)
LAGS=(1,2,5,10)


def acf(x,lag):
    x=np.asarray(x,float)
    if len(x)<=lag+2 or np.std(x)==0:return None
    return float(np.corrcoef(x[:-lag],x[lag:])[0,1])


class Accumulator:
    """Pools sequence statistics; autocorrelations are computed within each sequence and averaged."""
    def __init__(self):
        self.kinds=np.zeros(4);self.dt=[];self.ret=[];self.acf1=[];self.absacf={l:[] for l in LAGS};self.same=[0,0]
        self.size=defaultdict(list);self.cum={h:[] for h in HORIZONS}

    def add(self,trace,pool):
        kind=np.asarray(trace['kind'],int);dt=np.asarray(trace['dt'],int);cum=np.asarray(trace['return_bp'],float);vol=np.asarray(trace['base_volume'],float)
        if not len(kind):return
        self.kinds+=np.bincount(kind,minlength=4);self.dt.extend(dt.tolist())
        r=np.diff(np.r_[0.,cum]);swap=kind<2;rs=r[swap]
        self.ret.extend(rs.tolist())
        a=acf(rs,1)
        if a is not None:self.acf1.append(a)
        for l in LAGS:
            a=acf(np.abs(rs),l)
            if a is not None:self.absacf[l].append(a)
        sides=kind[swap];self.same[0]+=int(np.sum(sides[1:]==sides[:-1]));self.same[1]+=max(len(sides)-1,0)
        self.size[pool].extend(np.log(vol[swap&(vol>0)]).tolist())
        for h in HORIZONS:
            if len(cum)>=h:self.cum[h].append(cum[h-1])

    def result(self,centres):
        ret=np.asarray(self.ret);dt=np.asarray(self.dt)
        sizes=np.concatenate([np.asarray(v)-centres[p] for p,v in self.size.items() if p in centres]) if self.size else np.array([])
        k=float(((ret-ret.mean())**4).mean()/ret.var()**2-3) if len(ret)>3 and ret.var()>0 else None
        return dict(
            events=int(self.kinds.sum()),mix=(self.kinds/max(self.kinds.sum(),1)).tolist(),
            same_block=float(np.mean(dt==0)) if len(dt) else None,gap_median=float(np.median(dt)) if len(dt) else None,gap_p90=float(np.quantile(dt,.9)) if len(dt) else None,
            return_acf1=float(np.mean(self.acf1)) if self.acf1 else None,excess_kurtosis=k,
            abs_return_acf={l:(float(np.mean(v)) if v else None) for l,v in self.absacf.items()},
            side_persistence=self.same[0]/self.same[1] if self.same[1] else None,
            log_size_median=float(np.median(sizes)) if len(sizes) else None,log_size_p90=float(np.quantile(sizes,.9)) if len(sizes) else None,
            volatility={h:dict(sd=float(np.std(v)),width_5_95=float(np.subtract(*np.quantile(v,[.95,.05]))),n=len(v)) for h,v in self.cum.items() if len(v)>2})


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    acc=defaultdict(Accumulator);real_sizes=defaultdict(list);models=set()
    for f in sorted((a.rollouts/'anchors').glob('*.json')):
        r=json.loads(f.read_text());group='heldout' if r['split'].startswith('heldout') else 'seen';pool=r['pool']
        if r.get('actual_trace'):
            acc[(group,'real')].add(r['actual_trace'],pool)
            t=r['actual_trace'];real_sizes[pool].extend(np.log([v for k,v in zip(t['kind'],t['base_volume']) if k<2 and v>0]).tolist())
        for label,paths in r['paths'].items():
            if label.startswith('baseline:'):continue
            models.add(label)
            for path in paths:
                if path.get('trace'):acc[(group,label)].add(path['trace'],pool)
    centres={pool:float(np.median(v)) for pool,v in real_sizes.items() if v}
    result=dict(method=__doc__,horizons=list(HORIZONS),groups={})
    for (group,label),x in sorted(acc.items()):result['groups'].setdefault(group,{})[label]=x.result(centres)
    for group,by in result['groups'].items():
        real=by.get('real',{}).get('volatility',{})
        for label,v in by.items():
            if label=='real':continue
            v['volatility_ratio']={h:dict(sd=v['volatility'][h]['sd']/real[h]['sd'],width=v['volatility'][h]['width_5_95']/real[h]['width_5_95'])
                                   for h in v['volatility'] if h in real and real[h]['sd']>0 and real[h]['width_5_95']>0}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for group,by in result['groups'].items():
        print(f'\n== {group} pools')
        print(f"{'':12s} {'sell':>6s} {'buy':>6s} {'mint':>6s} {'burn':>6s} {'sameblk':>8s} {'r acf1':>7s} {'kurt':>7s} {'|r|acf1':>8s} {'|r|acf10':>9s} {'persist':>8s} {'vol h32':>8s} {'vol h512':>9s}")
        for label,v in sorted(by.items(),key=lambda kv:kv[0]!='real'):
            vr=v.get('volatility_ratio',{});fmt=lambda x,f='{:7.3f}':f.format(x) if x is not None else '   n/a '
            print(f"{label:12s} "+' '.join(f'{m:6.3f}' for m in v['mix'])+f" {fmt(v['same_block'],'{:8.3f}')} {fmt(v['return_acf1'])} {fmt(v['excess_kurtosis'],'{:7.1f}')}"
                  f" {fmt(v['abs_return_acf'][1],'{:8.3f}')} {fmt(v['abs_return_acf'][10],'{:9.3f}')} {fmt(v['side_persistence'],'{:8.3f}')}"
                  +(f" {vr[32]['sd']:8.2f}x {vr[512]['sd']:8.2f}x" if 32 in vr and 512 in vr else (f" {v['volatility'][32]['sd']:7.1f}bp {v['volatility'][512]['sd']:7.1f}bp" if 512 in v['volatility'] else '')))


if __name__=='__main__':main()
