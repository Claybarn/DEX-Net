"""Freeze the pool-balanced coverage schedule and training protocol for one run.

The reference model in the paper is d_model 128, 4 layers, no pool embeddings,
seed 260930 (replicates 260931, 260932):

  python -m scripts.prepare_training --data data/prepared --context data/context --out runs/ref --seed 260930
  python -m scripts.train --out runs/ref
"""
import argparse,hashlib,json,math
from pathlib import Path
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
    a=p.parse_args()
    data,context=a.data.resolve(),a.context.resolve()
    if not (data/'COMPLETE.json').exists() or not (context/'COMPLETE.json').exists():
        raise ValueError('Prepared data and context must be complete')
    manifest=json.loads((data/'dataset.json').read_text())
    lengths={n:v['indices']['train-h1'] for n,v in manifest['pools'].items() if not v['held_out']}
    schedule=make_schedule(a.out/'schedule',lengths,epochs=a.epochs,seed=a.seed)
    pool_ids={n:i+1 for i,n in enumerate(sorted(lengths))}
    plan=dict(data=str(data),context=str(context),schedule=str((a.out/'schedule').resolve()),seed=a.seed,pool_ids=pool_ids,embeddings=a.embeddings,
        epochs=a.epochs,d_model=a.d_model,layers=a.layers,learning_rate=.0001,learning_rate_schedule='1000-update warmup; cosine decay to 10% of peak',
        batches_per_round=8192,rounds=math.ceil(schedule['total_batches']/8192),training_examples=schedule['total_examples'],
        microbatch=256,effective_batch=1024,initialization=f'Fresh; seed {a.seed}',visible_history=a.visible_history,depth=str(a.depth.resolve()) if a.depth else None,
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
