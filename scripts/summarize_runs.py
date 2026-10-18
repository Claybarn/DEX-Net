"""Next-event NLL tables for a finished run matrix (scripts.run_matrix).

For every family: seen-pool (test.json) and held-out-pool (heldout_test.json) macro
joint NLL per seed, mean and sample s.d. across seeds, per-head macro NLL and type
accuracy, and per-pool seed means. Differences against --base use the seed means;
their standard error combines both families' seed-to-seed spread (seeds are
independent trainings, so this is the noise a reader should compare against).

  python -m scripts.summarize_runs --matrix paper/final/matrix.json --runs runs/final \
      --base depth --out results/final-nll.json
"""
import argparse,json,math
from pathlib import Path
import numpy as np

SPLITS=(('seen','test.json'),('heldout','heldout_test.json'))


def macro(result,key):
    return float(np.mean([v['loss'][key] for v in result.values()]))


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--matrix',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--base',default='depth');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();matrix=json.loads(a.matrix.read_text());families={}
    for family in matrix['families']:
        f=families[family]={}
        for split,file in SPLITS:
            per_seed={};heads={};acc=[];pools={}
            for seed in matrix['seeds']:
                path=a.runs/f'{family}-seed{seed}'/'training'/file
                if not path.exists():continue
                r=json.loads(path.read_text());per_seed[seed]=macro(r,'joint');acc.append(float(np.mean([v['type_accuracy'] for v in r.values()])))
                for k in next(iter(r.values()))['loss']:heads.setdefault(k,[]).append(macro(r,k))
                for pool,v in r.items():pools.setdefault(pool,[]).append(v['loss']['joint'])
            if not per_seed:continue
            x=np.array(list(per_seed.values()))
            f[split]=dict(seeds=per_seed,mean=float(x.mean()),sd=float(x.std(ddof=1)) if len(x)>1 else None,
                          heads={k:float(np.mean(v)) for k,v in heads.items()},type_accuracy=float(np.mean(acc)),per_pool={k:float(np.mean(v)) for k,v in pools.items()})
    differences={}
    for family,f in families.items():
        if family==a.base or a.base not in families:continue
        for split in f:
            b=families[a.base].get(split)
            if not b or f[split]['sd'] is None or b['sd'] is None:continue
            se=math.sqrt(f[split]['sd']**2/len(f[split]['seeds'])+b['sd']**2/len(b['seeds']))
            differences[f'{family}-{a.base}:{split}']=dict(difference=f[split]['mean']-b['mean'],standard_error=se)
    result=dict(method=__doc__,base=a.base,families=families,differences=differences)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    print(f"{'family':14s} {'seen':>18s} {'held-out':>18s}  seeds")
    for family,f in families.items():
        cell=lambda s:f"{f[s]['mean']:.3f} ± {f[s]['sd']:.3f}" if s in f and f[s]['sd'] is not None else (f"{f[s]['mean']:.3f}" if s in f else '-')
        print(f"{family:14s} {cell('seen'):>18s} {cell('heldout'):>18s}  {len(f.get('seen',{}).get('seeds',{}))}")
    for k,v in differences.items():print(f"  {k:28s} {v['difference']:+.3f} (s.e. {v['standard_error']:.3f})")


if __name__=='__main__':main()
