"""Table 1 inputs: per-pool event counts by kind, date range and chronological split sizes.

  python paper/tables/dataset_table.py --data data/prepared --out paper/results/final/dataset-table.json
"""
import argparse,json,datetime
from pathlib import Path
import numpy as np

KINDS=('sell','buy','mint','burn')


def date(ts):return datetime.datetime.fromtimestamp(int(ts),datetime.timezone.utc).strftime('%Y-%m-%d')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.data/'dataset.json').read_text());out={}
    for name,m in manifest['pools'].items():
        f=a.data/'pools'/name;meta=json.loads((f/'metadata.json').read_text())
        kind=np.load(f/'kind.npy',mmap_mode='r');ts=np.load(f/'timestamp.npy',mmap_mode='r')
        counts=np.bincount(np.asarray(kind),minlength=4)
        splits={}
        # Held-out pools have no training rows; their periods are named heldout_val / heldout_test.
        for split in (('heldout_val','heldout_test') if m['held_out'] else ('train','val','test')):
            idx=np.load(f/'indices'/f'{split}-h1.npy',mmap_mode='r')
            splits[split.replace('heldout_','')]=dict(rows=int(len(idx)),first=date(ts[int(idx[0])]),last=date(ts[int(idx[-1])]))
        out[name]=dict(held_out=m['held_out'],fee_tier=meta.get('fee'),tick_spacing=meta.get('tick_spacing'),events=int(len(kind)),
                       **{k:int(c) for k,c in zip(KINDS,counts)},first=date(ts[0]),last=date(ts[-1]),splits=splits)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=1)+'\n')
    for n,v in out.items():print(n,v['events'],v['first'],v['last'],{s:x['rows'] for s,x in v['splits'].items()})


if __name__=='__main__':main()
