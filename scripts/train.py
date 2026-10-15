"""Run every training round of a prepared run in order; safe to interrupt and rerun.

Each round is a fresh process (python -m scripts.train_round) that resumes from the
previous round's checkpoint, so memory never accumulates across the 202M-example run.
After the last round the validation-selected checkpoint is scored on test and held-out pools.

  python -m scripts.train --out runs/ref [--smoke]
"""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out',type=Path,required=True,help='run folder from scripts.prepare_training')
    p.add_argument('--smoke',action='store_true',help='128-batch GPU check in <out>/smoke; weights are not used')
    a=p.parse_args()
    if a.smoke:
        subprocess.run([sys.executable,'-u','-m','scripts.train_round','--out',str(a.out/'smoke'),'--round','1','--smoke'],check=True)
        print((a.out/'smoke'/'benchmark.json').read_text());return
    dest=a.out/'training';rounds=json.loads((dest/'protocol.json').read_text())['rounds']
    for r in range(1,rounds+1):
        if (dest/'COMPLETE.json').exists():break
        if (dest/f'round-{r}.json').exists() and (dest/f'round-{r}.pt').exists() and r<rounds:continue
        with (a.out/f'round-{r}.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.train_round','--out',str(dest),'--round',str(r)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'round',r,'of',rounds,'done',flush=True)
    print('COMPLETE' if (dest/'COMPLETE.json').exists() else 'INCOMPLETE',flush=True)


if __name__=='__main__':main()
