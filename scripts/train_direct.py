"""Train the direct baseline (paper: 128x4, AdamW, bf16, 16.8M examples, from scratch).

Each pool contributes 4,096-example chunks in turn (equal pool weight); every
microbatch has a fixed RNG seed, so runs are reproducible. Validation every 2.1M
examples selects best.pt; the selected model is then scored on all splits.

  python -m scripts.prepare_direct_labels --data data/prepared --out runs/direct/labels
  python -m scripts.train_direct --data data/prepared --labels runs/direct/labels --out runs/direct/seed1 \
      --init-seed 88011 --train-seed 61001
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from dexnet.direct import DirectForecast,DirectDataset,HORIZONS
from dexnet.utils import dump,atomic_save


def sets_for(data,labels,split):
    manifest=json.loads((Path(data)/'dataset.json').read_text())['pools']
    return {n:DirectDataset(data,labels,n,split) for n,m in manifest.items() if m['held_out']==split.startswith('heldout')}


def sample_scores(samples,target,scales):
    p=samples.shape[1];z=(samples/scales).flatten(2);y=(target/scales).flatten(1)
    pair=torch.cdist(z,z,compute_mode='donot_use_mm_for_euclid_dist');energy=torch.linalg.vector_norm(z-y[:,None],dim=-1).mean(1)-pair.sum((1,2))/(2*p*(p-1))
    sorted_x=samples.sort(dim=1).values;w=torch.arange(1,p+1,device=samples.device,dtype=samples.dtype)*2-p-1
    crps=(samples-target[:,None]).abs().mean(1)-(sorted_x*w[None,:,None,None]).sum(1)/(p*(p-1))
    lo,hi=torch.quantile(samples,torch.tensor([.05,.95],device=samples.device),dim=1)
    coverage=((target>=lo)&(target<=hi)).float();bias=samples.mean(1)-target
    objective=.5*energy/math.sqrt(z.shape[-1])+.5*(crps/scales).mean((1,2))
    return objective,energy,crps,coverage,bias


@torch.no_grad()
def evaluate(model,sets,limit,paths=128):
    model.eval();result={};start=time.monotonic()
    for no,(name,ds) in enumerate(sorted(sets.items())):
        picks=np.sort(np.random.default_rng(31001+no).choice(len(ds),min(limit,len(ds)),replace=False));values=[];energies=[];crps=[];cover=[];bias=[];baseline=[];base_energy=[]
        torch.manual_seed(91001+no)
        for offset in range(0,len(picks),128):
            b=ds.batch(picks[offset:offset+128],'cuda');x=model.sample(b,paths);obj,e,c,cov,err=sample_scores(x,b['y'],model.scales)
            values.extend(obj.cpu().tolist());energies.extend(e.cpu().tolist());crps.extend(c.cpu().tolist());cover.extend(cov.cpu().tolist());bias.extend(err.cpu().tolist())
            base=(b['baseline']-b['y']).abs();be=torch.linalg.vector_norm(((b['baseline']-b['y'])/model.scales).flatten(1),dim=-1)
            baseline.extend((.5*be/math.sqrt(12)+.5*(base/model.scales).mean((1,2))).cpu().tolist());base_energy.extend(base.cpu().tolist())
        result[name]=dict(examples=len(picks),objective=float(np.mean(values)),joint_energy=float(np.mean(energies)),marginal_crps=np.mean(crps,0).tolist(),coverage90=np.mean(cover,0).tolist(),mean_bias=np.mean(bias,0).tolist(),mean_forecast_rmse=np.sqrt(np.mean(np.array(bias)**2,0)).tolist(),recent_rate_baseline_objective=float(np.mean(baseline)),recent_rate_baseline_mae=np.mean(base_energy,0).tolist())
    return dict(pools=result,macro_objective=float(np.mean([v['objective'] for v in result.values()])),seconds=time.monotonic()-start,paths=paths)


def train(a):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    scalers=json.loads((a.data/'scalers.json').read_text())['scalers'];scales=json.loads((a.labels/'scales.json').read_text())
    sets=sets_for(a.data,a.labels,'train');names=sorted(sets)
    torch.manual_seed(a.init_seed)
    m=DirectForecast(scalers,scales['scales'],scales['increments'],d_model=a.d_model,layers=a.layers).cuda()
    effective,micro,budget=a.effective,a.micro,a.budget
    if effective>4096 or 4096%effective or effective%micro or budget%effective:raise ValueError('Batch must exactly tile the fixed 4096-example sampling chunks')
    steps=budget//effective
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr,weight_decay=.01)
    spec=dict(optimizer='adamw',effective=effective,lr=a.lr,bf16=not a.no_bf16,pretrained=False,d_model=a.d_model,layers=a.layers,gas=False,init_seed=a.init_seed,train_seed=a.train_seed)
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    dump(out/'protocol.json',dict(spec=spec,budget_examples=budget,micro=micro,train_paths=16,data=str(a.data),labels=str(a.labels),
         trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad)))
    valsets=sets_for(a.data,a.labels,'val');start=time.monotonic();history=[];best=math.inf;interval=2097152
    for step in range(steps):
        examples=step*effective;block=examples//4096;position=examples%4096;ds=sets[names[block%len(names)]]
        indices=np.random.default_rng(881001+block).integers(0,len(ds),4096)[position:position+effective]
        m.train();opt.zero_grad(set_to_none=True);total=0.
        for offset in range(0,effective,micro):
            batch=ds.batch(indices[offset:offset+micro],'cuda')
            torch.manual_seed(a.train_seed+examples+offset)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=spec['bf16']):
                loss=m.loss(batch,paths=16)
            (loss*micro/effective).backward();total=total+loss.detach()*micro/effective
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        lr=a.lr*min(1.,(examples+effective)/262144)*(.1+.9*.5*(1+math.cos(math.pi*examples/budget)))
        for group in opt.param_groups:group['lr']=lr
        opt.step()
        if not torch.isfinite(total):raise ValueError('Nonfinite direct training loss')
        done=examples+effective;elapsed=time.monotonic()-start
        if step%32==0 or done==budget:
            progress=dict(examples=done,budget_examples=budget,updates=step+1,loss=float(total),gradient_norm=float(norm),seconds=elapsed,examples_per_second=done/elapsed)
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        if done%interval==0 or done==budget:
            validation=evaluate(m,valsets,2048,paths=128)
            history.append(dict(examples=done,seconds=time.monotonic()-start,validation=validation));dump(out/'history.json',history);score=validation['macro_objective']
            checkpoint=dict(model=m.state_dict(),scalers=scalers,scales=scales,model_config=dict(d_model=a.d_model,layers=a.layers,gas_dim=0),
                            gas_scaler=None,spec=spec,examples=done,optimizer=opt.state_dict(),validation=score)
            atomic_save(checkpoint,out/'latest.pt')
            if score<best:best=score;atomic_save(checkpoint,out/'best.pt')
    chosen=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(chosen['model'])
    for split in ['val','heldout_val','test','heldout_test']:dump(out/(split+'.json'),evaluate(m,sets_for(a.data,a.labels,split),4096,paths=128))
    dump(out/'COMPLETE.json',dict(examples=budget,seconds=time.monotonic()-start,best_validation=best,selected_examples=chosen['examples']))


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--d-model',type=int,default=128);p.add_argument('--layers',type=int,default=4)
    p.add_argument('--init-seed',type=int,default=88011);p.add_argument('--train-seed',type=int,default=61001)
    p.add_argument('--budget',type=int,default=16777216);p.add_argument('--effective',type=int,default=1024);p.add_argument('--micro',type=int,default=512)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--no-bf16',action='store_true')
    train(p.parse_args())


if __name__=='__main__':main()
