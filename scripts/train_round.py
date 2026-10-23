"""Train one round (8,192 microbatches) of a run prepared by scripts.prepare_training; resumable.

Default path: four 256-example microbatches per optimizer update, gradients
accumulated. With "fused": true in the protocol, the same four microbatches are
merged into one 1,024-example forward/backward (the same mean loss and gradient,
up to floating-point order and dropout draws), built ahead by loader worker
processes. That cuts per-step launch and synchronisation overhead about fourfold.

  python -m scripts.train_round --out runs/ref/training --round 1
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from dexnet.model import PooledEventModel
from dexnet.learned import LearnedTransitionModel
from dexnet.coverage import CoverageSchedule
from dexnet.utils import dump,atomic_save
from dexnet.evaluation import load_sets,device_batch,host_batch,merge_batches,expand_candidates,evaluate,evaluate_fast


def attach_ids(sets,plan):
    for name,ds in sets.items():
        ds.pool_id=plan['pool_ids'].get(name,0)
        if plan.get('visible_history') is not None:ds.visible_history=plan['visible_history']
    return sets


class UpdateBatches(torch.utils.data.Dataset):
    """Merged CPU batch for each optimizer update, in schedule order (no randomness)."""
    def __init__(self,sets,schedule,first,count,accum,static=False):
        self.sets,self.schedule,self.first,self.count,self.accum,self.static=sets,schedule,first,count,accum,static
    def __len__(self):return self.count
    def __getitem__(self,u):
        start=(self.first+u)*self.accum;parts=[self.schedule[i] for i in range(start,start+self.accum)]
        b=merge_batches([host_batch(self.sets[name],indices) for name,indices in parts])
        return (expand_candidates(b) if self.static else b),[(name,len(indices)) for name,indices in parts]


def learning_rate(step,total,peak):
    warmup=min(1000,max(1,total//20))
    if step<warmup:return peak*(step+1)/warmup
    fraction=min(1.,(step-warmup)/max(1,total-warmup))
    return peak*(.1+.9*.5*(1+math.cos(math.pi*fraction)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--round',type=int,default=1);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    plan=json.loads((a.out/'protocol.json').read_text());digest=hashlib.sha256((a.out/'protocol.json').read_bytes()).hexdigest()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    scalers=json.loads((Path(plan['data'])/'scalers.json').read_text())['scalers'];torch.manual_seed(plan['seed'])
    config=dict(d_model=plan['d_model'],layers=plan['layers'],pool_count=len(plan['pool_ids']) if plan['embeddings'] else 0,pool_dropout=.1)
    if plan.get('depth'):config['liquidity_context']=True
    if plan.get('learned_transitions'):config['learned_transitions']=True
    model=(LearnedTransitionModel if plan.get('learned_transitions') else PooledEventModel)(scalers,**config).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=.01)
    best=math.inf
    if a.round==1 and plan.get('init_checkpoint'):
        start=torch.load(plan['init_checkpoint'],map_location='cpu',weights_only=False)
        if {k:start['model_config'].get(k) for k in config}!=config:raise ValueError('Fine-tuning checkpoint configuration differs')
        model.load_state_dict(start['model'])
    if a.round>1:
        ck=torch.load(a.out/f'round-{a.round-1}.pt',map_location='cpu',weights_only=False)
        if ck['protocol_sha256']!=digest:raise ValueError('Resume protocol mismatch')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);best=ck['best_validation']
    sets=attach_ids(load_sets(plan,'train'),plan);schedule=CoverageSchedule(plan['schedule'])
    if set(sets)!=set(plan['pool_ids']) or any(len(sets[n])!=schedule.spec['lengths'][n] for n in sets):raise ValueError('Training roster/coverage mismatch')
    micro=schedule.spec['batch_size'];effective=1024;accum=effective//micro
    begin=(a.round-1)*plan['batches_per_round'];end=min(len(schedule),begin+(128 if a.smoke else plan['batches_per_round']))
    if begin%accum or (end-begin)%accum:raise ValueError('Incomplete optimizer update')
    torch.manual_seed(plan['seed']+a.round*1000);model.train();optimizer.zero_grad(set_to_none=True)
    start=time.monotonic();totals={};counts={};seen={n:0 for n in sets};maxnorm=0.;clipped=0;updates=0
    if plan.get('static'):
        # Sync-free path: static loss (optionally compiled), statistics kept on the GPU and read every 32 updates.
        loss_fn=torch.compile(model.losses_static,dynamic=False,mode=plan.get('compile_mode') or 'default') if plan.get('compile') else model.losses_static
        loader=torch.utils.data.DataLoader(UpdateBatches(sets,schedule,begin//accum,(end-begin)//accum,accum,static=True),batch_size=None,shuffle=False,
                                           num_workers=plan.get('loader_workers',6),pin_memory=True,prefetch_factor=4,persistent_workers=False)
        dev=torch.device('cuda');gtot={};gcnt={};gmax=torch.zeros((),device=dev);gclip=torch.zeros((),device=dev);bad=torch.zeros((),dtype=torch.bool,device=dev)
        def report(i):
            nonlocal maxnorm,clipped
            if bool(bad):raise ValueError(f'Nonfinite loss/gradient or physical-support violation before batch {i+1}')
            for k in gtot:totals[k]=float(gtot[k]);counts[k]=int(gcnt[k])
            maxnorm=float(gmax);clipped=int(gclip)
            elapsed=time.monotonic()-start;examples=(i-begin+1)*micro
            progress=dict(round=a.round,rounds=plan['rounds'],batch=i+1,total_batches=len(schedule),examples_completed=(i+1)*micro,
                round_examples=examples,optimizer_steps=(i+1)//accum,seconds=elapsed,examples_per_second=examples/elapsed,
                joint_nll=totals['joint']/counts['joint'],gradient_norm_max=maxnorm,learning_rate=optimizer.param_groups[0]['lr'])
            dump(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
        for u,(cpu,parts_info) in enumerate(loader):
            b={k:v.to(dev,non_blocking=True) for k,v in cpu.items()};loss,parts=loss_fn(b)
            loss.backward()
            for name,n in parts_info:seen[name]+=n
            for k,(v,n) in parts.items():
                if k=='violations':bad=bad|(v>0);continue
                gtot[k]=gtot.get(k,0)+v.detach();gcnt[k]=gcnt.get(k,0)+n
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            bad=bad|~torch.isfinite(norm)|~torch.isfinite(loss.detach());gmax=torch.maximum(gmax,norm.detach());gclip=gclip+(norm>1)
            lr=plan['learning_rate'] if a.smoke else learning_rate(begin//accum+u,len(schedule)//accum,plan['learning_rate'])
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            i=begin+(u+1)*accum-1
            if u%32==0 or i+1==end:report(i)
    elif plan.get('fused'):
        loader=torch.utils.data.DataLoader(UpdateBatches(sets,schedule,begin//accum,(end-begin)//accum,accum),batch_size=None,shuffle=False,
                                           num_workers=plan.get('loader_workers',6),pin_memory=True,prefetch_factor=4,persistent_workers=False)
        for u,(cpu,parts_info) in enumerate(loader):
            b={k:v.to('cuda',non_blocking=True) for k,v in cpu.items()};loss,parts=model.losses(b)
            if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss at update {begin//accum+u}')
            loss.backward()
            for name,n in parts_info:seen[name]+=n
            for k,(v,n) in parts.items():totals[k]=totals.get(k,0)+float(v.detach());counts[k]=counts.get(k,0)+int(n)
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));maxnorm=max(maxnorm,norm);clipped+=norm>1
            lr=plan['learning_rate'] if a.smoke else learning_rate(begin//accum+u,len(schedule)//accum,plan['learning_rate'])
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            i=begin+(u+1)*accum-1
            if u%32==0 or i+1==end:
                elapsed=time.monotonic()-start;examples=(i-begin+1)*micro
                progress=dict(round=a.round,rounds=plan['rounds'],batch=i+1,total_batches=len(schedule),examples_completed=(i+1)*micro,
                    round_examples=examples,optimizer_steps=(i+1)//accum,seconds=elapsed,examples_per_second=examples/elapsed,
                    joint_nll=totals['joint']/counts['joint'],gradient_norm_max=maxnorm,learning_rate=optimizer.param_groups[0]['lr'])
                dump(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    for i in (range(0) if plan.get('fused') or plan.get('static') else range(begin,end)):
        name,indices=schedule[i];b=device_batch(sets[name],indices,'cuda');loss,parts=model.losses(b)
        if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss at batch {i}')
        (loss/accum).backward();seen[name]+=len(indices)
        for k,(v,n) in parts.items():totals[k]=totals.get(k,0)+float(v.detach());counts[k]=counts.get(k,0)+int(n)
        if (i+1)%accum==0:
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));maxnorm=max(maxnorm,norm);clipped+=norm>1
            # Smoke tests the peak rate; full runs warm up before cosine decay.
            lr=plan['learning_rate'] if a.smoke else learning_rate(i//accum,len(schedule)//accum,plan['learning_rate'])
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
        if (i-begin)%128==0 or i+1==end:
            elapsed=time.monotonic()-start;examples=(i-begin+1)*micro
            progress=dict(round=a.round,rounds=plan['rounds'],batch=i+1,total_batches=len(schedule),examples_completed=(i+1)*micro,
                round_examples=examples,optimizer_steps=(i+1)//accum,seconds=elapsed,examples_per_second=examples/elapsed,
                joint_nll=totals['joint']/counts['joint'],gradient_norm_max=maxnorm,learning_rate=optimizer.param_groups[0]['lr'])
            dump(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    elapsed=time.monotonic()-start
    result=dict(round=a.round,training_examples=seen,training_seconds=elapsed,examples_per_second=(end-begin)*micro/elapsed,
        loss={k:totals[k]/max(counts[k],1) for k in totals},gradient_norm_max=maxnorm,clipped_step_fraction=clipped/max(updates,1))
    score_sets=(lambda m,sets,limit,device:evaluate_fast(m,sets,limit,device,workers=plan.get('loader_workers',6))) if plan.get('static') else evaluate
    result['validation']=score_sets(model,attach_ids(load_sets(plan,'val'),plan),256 if a.smoke else 8192,'cuda')
    score=float(np.mean([v['loss']['joint'] for v in result['validation'].values()]));result['validation_macro_joint']=score
    improved=score<best or bool(plan.get('select_final'));best=min(best,score)
    checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),model_config=config,round=a.round,
        scalers=scalers,pool_ids=plan['pool_ids'],best_validation=best,protocol_sha256=digest)
    atomic_save(checkpoint,a.out/f'round-{a.round}.pt')
    if improved:atomic_save(checkpoint,a.out/'best.pt')
    dump(a.out/f'round-{a.round}.json',result)
    if a.smoke:
        dump(a.out/'benchmark.json',dict(parameters=sum(p.numel() for p in model.parameters()),examples=(end-begin)*micro,updates=updates,
            seconds=elapsed,examples_per_second=(end-begin)*micro/elapsed,estimated_training_seconds=len(schedule)/(end-begin)*elapsed,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(0)))
        (a.out/'COMPLETE').write_text('GPU smoke passed; discarded before full training.\n')
    elif end==len(schedule):
        chosen=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(chosen['model'])
        for split in ['test','heldout_val','heldout_test']:
            dump(a.out/(split+'.json'),score_sets(model,attach_ids(load_sets(plan,split),plan),32768,'cuda'))
        dump(a.out/'COMPLETE.json',dict(selected_round=chosen['round'],selection='Final round (fixed budget)' if plan.get('select_final') else 'Seen-pool validation macro joint NLL',epochs=plan['epochs'],examples=len(schedule)*micro))


if __name__=='__main__':main()
