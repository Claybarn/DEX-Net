"""Prepare and train a matrix of runs (families x seeds) on a multi-GPU machine.

The matrix file names the seeds, the families (scripts.prepare_training options per
family) and options common to every run. Runs share GPUs (--per-gpu at a time on
each); every run is a scripts.train process, which resumes from its last finished
round, so the launcher can be stopped and rerun at any time. Failed runs are
restarted up to --retries times.

  python -m scripts.run_matrix --matrix paper/final/matrix.json --data data/prepared \
      --context data/context --depth data/depth --out runs/final --gpus 0,1,2,3 --per-gpu 4
  python -m scripts.run_matrix --matrix paper/final/matrix.json --out runs/final --status
  python -m scripts.run_matrix ... --smoke     # one GPU smoke check per family, no training
"""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def runs(matrix):
    """(name, prepare_training options) for every family x seed, heaviest models first."""
    out=[]
    for family,options in matrix['families'].items():
        for seed in matrix['seeds']:
            out.append((f'{family}-seed{seed}',{**matrix.get('common',{}),**options,'seed':seed}))
    return sorted(out,key=lambda r:-r[1].get('d_model',128)*r[1].get('layers',4))


def prepare(name,options,a):
    folder=a.out/name
    if (folder/'training'/'protocol.json').exists():return name,'existing'
    cmd=[sys.executable,'-m','scripts.prepare_training','--data',str(a.data),'--context',str(a.context),'--out',str(folder)]
    for k,v in options.items():
        if k=='depth':
            if v:cmd+=['--depth',str(a.depth)]
        elif v is True:cmd.append('--'+k.replace('_','-'))
        elif v is not False and v is not None:cmd+=['--'+k.replace('_','-'),str(v)]
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'prepare.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    return name,'prepared'


def status(names,out):
    rows=[]
    for name in names:
        t=out/name/'training';p=t/'progress.json'
        if (t/'COMPLETE.json').exists():
            c=json.loads((t/'COMPLETE.json').read_text());rows.append(f"{name:28s} COMPLETE  best round {c['selected_round']}");continue
        if not p.exists():rows.append(f'{name:28s} waiting');continue
        g=json.loads(p.read_text());left=(g['total_batches']-g['batch'])*256/max(g['examples_per_second'],1)
        rows.append(f"{name:28s} round {g['round']:3d}/{g['rounds']}  {g['examples_per_second']:7.0f} ex/s  nll {g['joint_nll']:.3f}  ~{left/3600:.1f} h left (at this rate)")
    return '\n'.join(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--matrix',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--context',type=Path);p.add_argument('--depth',type=Path)
    p.add_argument('--gpus',default='0');p.add_argument('--per-gpu',type=int,default=4);p.add_argument('--retries',type=int,default=2)
    p.add_argument('--status',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();matrix=json.loads(a.matrix.read_text());todo=runs(matrix);names=[n for n,_ in todo]
    if a.status:print(status(names,a.out));return
    a.out.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(8,os.cpu_count() or 1)) as ex:
        for name,what in ex.map(lambda r:prepare(*r,a),todo):print(name,what,flush=True)
    if a.prepare_only:return
    gpus=a.gpus.split(',')
    if a.smoke:
        # One smoke per family, all at once, spread over the GPUs.
        first={};[first.setdefault(n.rsplit('-seed',1)[0],n) for n in names]
        procs=[(n,subprocess.Popen([sys.executable,'-m','scripts.train','--out',str(a.out/n),'--smoke'],stdout=(a.out/n/'smoke.log').open('w'),stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus[i%len(gpus)]})) for i,n in enumerate(first.values())]
        for n,pr in procs:
            code=pr.wait();b=a.out/n/'smoke'/'benchmark.json'
            print(n,'OK' if code==0 else f'FAILED ({code})',json.loads(b.read_text())['examples_per_second'] if b.exists() else '',flush=True)
        return
    queue=[n for n in names if not (a.out/n/'training'/'COMPLETE.json').exists()];running={};failures={}
    slots={g:0 for g in gpus}
    while queue or running:
        for n,(pr,g) in list(running.items()):
            code=pr.poll()
            if code is None:continue
            del running[n];slots[g]-=1
            if code and failures.get(n,0)<a.retries:failures[n]=failures.get(n,0)+1;queue.insert(0,n);print(time.strftime('%H:%M'),n,'failed; retry',failures[n],flush=True)
            else:print(time.strftime('%H:%M'),n,'finished' if code==0 else f'FAILED ({code})',flush=True)
        while queue and min(slots.values())<a.per_gpu:
            g=min(slots,key=slots.get);n=queue.pop(0);slots[g]+=1
            log=(a.out/f'{n}.log').open('a')
            running[n]=(subprocess.Popen([sys.executable,'-u','-m','scripts.train','--out',str(a.out/n)],stdout=log,stderr=subprocess.STDOUT,
                                         env={**os.environ,'CUDA_VISIBLE_DEVICES':g}),g)
            print(time.strftime('%H:%M'),'started',n,'on GPU',g,flush=True)
        time.sleep(30)
    print(status(names,a.out),flush=True)


if __name__=='__main__':main()
