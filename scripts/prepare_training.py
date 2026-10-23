"""Freeze the pool-balanced coverage schedule and training protocol for one run.

The reference model in the paper is d_model 128, 4 layers, no pool embeddings,
seed 260930 (replicates 260931, 260932):

  python -m scripts.prepare_training --data data/prepared --context data/context --out runs/ref --seed 260930
  python -m scripts.train --out runs/ref
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from dexnet.coverage import make_schedule
from dexnet.utils import dump


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data',type=Path,required=True,help='output of scripts.prepare_data')
    p.add_argument('--context',type=Path,required=True,help='output of scripts.prepare_context')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=260930)
    p.add_argument('--d-model',type=int,default=128)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--embeddings',action='store_true',help='learned pool-ID embeddings (held-out pools use neutral ID 0)')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--visible-history',type=int,help='ablation: events of history the model may see (default all 64)')
    p.add_argument('--depth',type=Path,help='liquidity-depth profiles from scripts.prepare_depth (depth-aware model)')
    p.add_argument('--fused',action='store_true',help='one 1,024-example forward/backward per update with background loader workers (faster)')
    p.add_argument('--static',action='store_true',help='fused, plus the sync-free static loss with statistics read every 32 updates (faster)')
    p.add_argument('--compile',action='store_true',help='with --static: torch.compile the loss (forward and backward)')
    p.add_argument('--loader-workers',type=int,default=6,help='batch-building worker processes (fused/static paths)')
    p.add_argument('--pools',help='comma-separated seen pools to train on (single-pool ablation; default all); final test scores still cover every pool')
    p.add_argument('--init-checkpoint',type=Path,help='fine-tune: start from these weights (same configuration) instead of a fresh initialization')
    p.add_argument('--learning-rate',type=float,default=.0001)
    p.add_argument('--special-rows',type=Path,help='folder of <pool>.npy absolute rows to oversample (e.g. scripts.jit_rows)')
    p.add_argument('--special-share',type=float,default=.25,help='share of every batch drawn from --special-rows')
    p.add_argument('--examples-per-pool',type=int,help='examples per pool per epoch (default: the largest pool\'s training rows); sets the budget of a short fine-tuning run')
    p.add_argument('--select-final',action='store_true',help='keep the last round instead of the best validation round (fixed-budget fine-tuning)')
    p.add_argument('--learned-transitions',action='store_true',help='mechanics ablation: learn each event\'s effect on price and liquidity instead of the exact replica (dexnet.learned)')
    p.add_argument('--compile-mode',default=None,help="torch.compile mode, e.g. 'reduce-overhead' for CUDA graphs")
    a=p.parse_args()
    if a.learned_transitions and a.depth:raise ValueError('The learned-transition ablation has no book, so no liquidity profile')
    data,context=a.data.resolve(),a.context.resolve()
    if not (data/'COMPLETE.json').exists() or not (context/'COMPLETE.json').exists():
        raise ValueError('Prepared data and context must be complete')
    manifest=json.loads((data/'dataset.json').read_text())
    lengths={n:v['indices']['train-h1'] for n,v in manifest['pools'].items() if not v['held_out']}
    if a.pools:
        chosen=a.pools.split(',')
        if not set(chosen)<=set(lengths):raise ValueError(f'Unknown or held-out pools: {sorted(set(chosen)-set(lengths))}')
        lengths={n:lengths[n] for n in chosen}
    special=None
    if a.special_rows:
        special={}
        for n in lengths:
            idx=np.load(data/'pools'/n/'indices'/'train-h1.npy',mmap_mode='r');rows=np.load(a.special_rows/(n+'.npy'))
            pos=np.searchsorted(idx,rows);ok=(pos<len(idx))&(np.asarray(idx)[np.minimum(pos,len(idx)-1)]==rows);special[n]=pos[ok]
    schedule=make_schedule(a.out/'schedule',lengths,epochs=a.epochs,seed=a.seed,special=special,share=a.special_share,examples_per_pool=a.examples_per_pool)
    pool_ids={n:i+1 for i,n in enumerate(sorted(lengths))}
    plan=dict(data=str(data),context=str(context),schedule=str((a.out/'schedule').resolve()),seed=a.seed,pool_ids=pool_ids,embeddings=a.embeddings,
        epochs=a.epochs,d_model=a.d_model,layers=a.layers,learning_rate=a.learning_rate,learning_rate_schedule='1000-update warmup; cosine decay to 10% of peak',
        batches_per_round=8192,rounds=math.ceil(schedule['total_batches']/8192),training_examples=schedule['total_examples'],
        microbatch=256,effective_batch=1024,initialization=f'From {a.init_checkpoint.resolve()}' if a.init_checkpoint else f'Fresh; seed {a.seed}',
        init_checkpoint=str(a.init_checkpoint.resolve()) if a.init_checkpoint else None,special_rows=str(a.special_rows.resolve()) if a.special_rows else None,
        special_share=a.special_share if a.special_rows else 0.,select_final=a.select_final,visible_history=a.visible_history,depth=str(a.depth.resolve()) if a.depth else None,fused=a.fused or a.static,static=a.static,compile=a.compile,compile_mode=a.compile_mode,loader_workers=a.loader_workers,learned_transitions=a.learned_transitions,pools=sorted(lengths) if a.pools else None,
        sampling=schedule['coverage'],selection='Minimum seen-pool validation macro joint NLL; fixed training budget, no early stopping',
        heldout='Held-out pools use neutral ID 0; no held-out weights/scaler fitting or checkpoint selection',
        validation='8192 fixed examples per seen pool each round',final_evaluation='32768 fixed examples per pool; selected checkpoint only',
        data_sha256=hashlib.sha256((data/'COMPLETE.json').read_bytes()).hexdigest(),
        context_sha256=hashlib.sha256((context/'COMPLETE.json').read_bytes()).hexdigest())
    for sub in ['smoke','training']:
        dest=a.out/sub;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'protocol.json').exists():raise RuntimeError('Existing protocol; use a new --out')
        dump(dest/'protocol.json',plan)
    print(json.dumps(dict(out=str(a.out),rounds=plan['rounds'],examples=plan['training_examples'])))


if __name__=='__main__':main()
