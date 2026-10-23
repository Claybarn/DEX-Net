"""Training rows aligned to just-in-time (JIT) liquidity, for oversampled fine-tuning.

Rows are absolute event rows of each seen pool's training split (a row's target is that
event; its history ends just before it). Two sets:

  history          rows whose history ends with a large mint, one or two events back
                   (large: minted liquidity > 10x the active liquidity before it). The
                   rows are chosen from the history alone, so oversampling them shifts
                   which situations the model practices, not what it predicts in them.
  history_outcome  history plus the rows of every strict JIT pair: the mint, the swap and
                   the burn (consecutive, one block, same range). These are chosen by
                   their outcome, so oversampling them deliberately raises the model's
                   JIT rate; it is a calibration choice, not an unbiased refit.

Ranges come from the training targets (lower = anchor + offset, width), as in
dexnet.training_data. Burn amounts are not compared (the per-event features hold amounts
relative to active liquidity only), so a pair is a same-block, same-range mint-swap-burn.

  python -m scripts.jit_rows --data data/prepared --context data/context --out data/jit-rows
"""
import argparse,json,math
from pathlib import Path
import numpy as np

LP_RATIO=3


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--context',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    scalers=json.loads((a.data/'scalers.json').read_text())['scalers']['raw_features']
    mean,std=scalers['mean'][LP_RATIO],scalers['std'][LP_RATIO]
    manifest=json.loads((a.data/'dataset.json').read_text())['pools'];summary={}
    for name,m in sorted(manifest.items()):
        if m['held_out']:continue
        folder=a.data/'pools'/name
        kind=np.load(folder/'kind.npy');block=np.load(folder/'block.npy')
        f=np.load(folder/'features.npy',mmap_mode='r');fm=np.load(folder/'feature_mask.npy',mmap_mode='r')
        ratio=np.asarray(f[:,LP_RATIO],np.float64)*std+mean;has=np.asarray(fm[:,LP_RATIO])
        large=(kind==2)&has&(ratio>math.log(10))
        targets=np.load(folder/'targets.npy',mmap_mode='r');anchor=np.load(a.context/name/'anchor.npy',mmap_mode='r')
        lower=np.rint(np.asarray(anchor,np.float64)+np.asarray(targets[:,4],np.float64));width=np.rint(np.asarray(targets[:,5],np.float64))
        train=np.load(folder/'indices'/'train-h1.npy');lo,hi=int(train.min()),int(train.max())
        n=len(kind);i=np.arange(n-2)
        pair=(kind[i]==2)&(kind[i+1]<2)&(kind[i+2]==3)&(block[i]==block[i+1])&(block[i+1]==block[i+2])&(lower[i]==lower[i+2])&(width[i]==width[i+2])
        mints=np.flatnonzero(pair)
        history=np.union1d(np.flatnonzero(large)+1,np.flatnonzero(large)+2)
        outcome=np.unique(np.concatenate([mints,mints+1,mints+2]))
        keep=lambda r:r[(r>=lo)&(r<=hi)]
        sets=dict(history=keep(history),history_outcome=keep(np.union1d(history,outcome)))
        for label,rows in sets.items():
            (a.out/label).mkdir(parents=True,exist_ok=True);np.save(a.out/label/(name+'.npy'),rows.astype(np.int64))
        tr=(np.arange(n)>=lo)&(np.arange(n)<=hi)
        summary[name]=dict(train_events=int(tr.sum()),large_mints=int((large&tr).sum()),strict_pairs=int(((np.r_[pair,False,False])&tr).sum()),
                           history_rows=int(len(sets['history'])),history_outcome_rows=int(len(sets['history_outcome'])))
        print(name,summary[name],flush=True)
    (a.out/'summary.json').write_text(json.dumps(dict(method=__doc__,pools=summary),indent=1)+'\n')


if __name__=='__main__':main()
