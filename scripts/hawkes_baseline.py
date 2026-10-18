"""Point-process baselines for the clock and event-type parts of next-event NLL.

Both baselines predict exactly what the neural model's clock and type heads predict,
on the same rows (the fixed evaluation rows of dexnet.evaluation.evaluate, 32,768 per
pool), in the same units (nats per event):

  clock  P(next event in the same block) and, otherwise, the block gap d >= 1
  type   sell / buy / mint / burn, given the gap (the neural type head also sees it)

  unconditional  renewal clock (constant same-block probability, 3-component
                 geometric mixture over gaps) and constant type frequencies
  hawkes         4-type discrete-time Hawkes process over the same last 64 events the
                 neural model sees: type intensities mu_m + sum_j alpha[s, k_j, m]
                 exp(-beta_s * age_j) with two time scales s; per-block hazard
                 1 - exp(-lambda(t)) for gaps >= 1 (sums over blocks in closed form),
                 same-block probability logistic in log lambda at the current block,
                 type probabilities lambda_m / lambda at the event's block

Seen pools: parameters fitted by maximum likelihood on a sample of that pool's training
rows. Held-out pools: one parameter set fitted on the pooled seen-pool samples (equal
rows per pool), mirroring the neural model's use of shared weights. Marks are not
modelled; the no-history ablation is the corresponding comparison for marks.

  python -m scripts.hawkes_baseline --data data/prepared --runs runs/final --out results/hawkes.json
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

SEQ=64;EVAL_ROWS=32768;SEEDS=(260930,260931,260932)


def pool_arrays(data,pool):
    f=Path(data)/'pools'/pool
    return {k:np.load(f/(k+'.npy'),mmap_mode='r') for k in ('block','kind','gap_blocks')},f


def batch(arrays,rows):
    """History blocks/kinds of the 64 previous events, the current block, gap and type for each row."""
    rows=np.asarray(rows,np.int64);idx=rows[:,None]+np.arange(-SEQ,0)[None]
    hb=np.asarray(arrays['block'])[idx].astype(np.float64);hk=np.asarray(arrays['kind'])[idx].astype(np.int64)
    dt=np.asarray(arrays['gap_blocks'])[rows].astype(np.float64);k=np.asarray(arrays['kind'])[rows].astype(np.int64)
    b=np.asarray(arrays['block'])
    if not np.array_equal(dt,(b[rows]-b[rows-1]).astype(np.float64)):raise ValueError('gap_blocks is not the block gap to the previous event')
    age=hb[:,-1:]-hb  # ages relative to the current block (the previous event's block)
    return dict(age=torch.tensor(age,dtype=torch.float32),hk=torch.tensor(hk),dt=torch.tensor(dt,dtype=torch.float32),k=torch.tensor(k))


class Unconditional(torch.nn.Module):
    def __init__(self):
        super().__init__();self.c=torch.nn.Parameter(torch.zeros(()));self.pi=torch.nn.Parameter(torch.zeros(3))
        self.rate=torch.nn.Parameter(torch.tensor([-4.,-1.,1.]));self.types=torch.nn.Parameter(torch.zeros(4))
    def nll(self,b):
        dt=b['dt'];r=F.softplus(self.rate)+1e-6
        gap=torch.logsumexp(self.pi.log_softmax(0)+torch.log(-torch.expm1(-r))-(dt[:,None]-1).clamp(min=0)*r,-1)
        clock=torch.where(dt==0,F.softplus(-self.c),F.softplus(self.c)-gap)
        return clock,-self.types.log_softmax(0)[b['k']]


class Hawkes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logmu=torch.nn.Parameter(torch.full((4,),-3.));self.logalpha=torch.nn.Parameter(torch.full((2,4,4),-4.))
        self.logbeta=torch.nn.Parameter(torch.tensor([0.,-4.]));self.c=torch.nn.Parameter(torch.tensor([0.,1.]))
    def nll(self,b):
        mu=self.logmu.exp();alpha=self.logalpha.exp();beta=self.logbeta.exp()
        onehot=F.one_hot(b['hk'],4).float();dt=b['dt']
        decay=torch.exp(-beta[:,None,None]*b['age'][None])           # (2, N, 64)
        counts=torch.einsum('snj,njk->snk',decay,onehot)            # decayed source counts at the current block
        X=torch.einsum('snk,skm->snm',counts,alpha)                  # excitation per target type, per scale
        A=X.sum(-1)                                                  # (2, N) total excitation per scale
        lam0=mu.sum()+A.sum(0)
        z=self.c[0]+self.c[1]*lam0.log()
        e=torch.exp(-beta)[:,None]
        d=dt.clamp(min=1)
        cum=mu.sum()*(d-1)+(A*e*(1-torch.exp(-beta[:,None]*(d-1)))/(1-e)).sum(0)
        lamd=mu.sum()+(A*torch.exp(-beta[:,None]*d)).sum(0)
        clock=torch.where(dt==0,F.softplus(-z),F.softplus(z)+cum-torch.log(-torch.expm1(-lamd)))
        lam_types=mu[None]+(X*torch.exp(-beta[:,None,None]*dt[None,:,None])).sum(0)
        types=-(lam_types.gather(1,b['k'][:,None]).squeeze(1).log()-lam_types.sum(1).log())
        return clock,types


def fit(model,b,device,steps=800):
    model=model.to(device);b={k:v.to(device) for k,v in b.items()}
    opt=torch.optim.Adam(model.parameters(),lr=.05)
    for step in range(steps):
        if step==steps//2:
            for g in opt.param_groups:g['lr']=.01
        c,t=model.nll(b);loss=(c+t).mean();opt.zero_grad();loss.backward();opt.step()
    # Full-batch L-BFGS refinement to convergence (the pooled held-out fit needs it).
    lbfgs=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=500,tolerance_grad=1e-9,tolerance_change=1e-12,history_size=50,line_search_fn='strong_wolfe')
    def closure():
        lbfgs.zero_grad();c,t=model.nll(b);loss=(c+t).mean();loss.backward();return loss
    lbfgs.step(closure)
    return model


@torch.no_grad()
def score(model,b,device):
    c,t=model.to(device).nll({k:v.to(device) for k,v in b.items()})
    return float(c.mean()),float(t.mean())


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-rows',type=int,default=200000);p.add_argument('--steps',type=int,default=800);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args();torch.manual_seed(0)
    manifest=json.loads((a.data/'dataset.json').read_text())['pools']
    seen=[n for n,m in manifest.items() if not m['held_out']];held=[n for n,m in manifest.items() if m['held_out']]
    rng=np.random.default_rng(0);train={};evalb={}
    for no,pool in enumerate(seen):
        arrays,_=pool_arrays(a.data,pool);idx=np.load(a.data/'pools'/pool/'indices'/'train-h1.npy',mmap_mode='r')
        train[pool]=batch(arrays,np.sort(rng.choice(np.asarray(idx),min(a.train_rows,len(idx)),replace=False)))
    for group,pools,split in (('seen',seen,'test'),('heldout',held,'heldout_test')):
        for no,pool in enumerate(pools):
            arrays,_=pool_arrays(a.data,pool);idx=np.load(a.data/'pools'/pool/'indices'/f'{split}-h1.npy',mmap_mode='r')
            # Same rows as dexnet.evaluation.evaluate: dataset positions, then mapped to pool rows.
            pos=np.sort(np.random.default_rng(74401+no).choice(len(idx),min(EVAL_ROWS,len(idx)),replace=False))
            evalb[pool]=(group,batch(arrays,np.asarray(idx)[pos]))
    pooled={k:torch.cat([train[p][k][:a.train_rows//len(seen)] for p in seen]) for k in train[seen[0]]}
    fitted={}
    for name,cls in (('unconditional',Unconditional),('hawkes',Hawkes)):
        start=time.time();fitted[name]={pool:fit(cls(),train[pool],a.device,a.steps) for pool in seen};fitted[name]['__pooled__']=fit(cls(),pooled,a.device,a.steps)
        print(name,'fitted in',round(time.time()-start),'s',flush=True)
    result=dict(method=__doc__,train_rows_per_pool=a.train_rows,steps=a.steps,pools={},macro={})
    for pool,(group,b) in evalb.items():
        r=dict(group=group,rows=len(b['dt']))
        for name in fitted:
            m=fitted[name][pool if group=='seen' else '__pooled__'];c,t=score(m,b,a.device);r[name]=dict(clock=c,type=t,clock_plus_type=c+t)
        for fam in ('reference','depth'):
            file='test.json' if group=='seen' else 'heldout_test.json'
            runs=[json.loads((a.runs/f'{fam}-seed{s}'/'training'/file).read_text())[pool] for s in SEEDS]
            c=np.array([x['loss']['wait'] for x in runs]);t=np.array([x['loss']['kind'] for x in runs])
            if any(x['examples']!=r['rows'] for x in runs):raise ValueError(f'{pool}: row count differs from the neural evaluation')
            r[fam]=dict(clock=float(c.mean()),type=float(t.mean()),clock_plus_type=float((c+t).mean()),clock_plus_type_sd=float((c+t).std(ddof=1)))
        result['pools'][pool]=r
    for group in ('seen','heldout'):
        ps=[v for v in result['pools'].values() if v['group']==group]
        result['macro'][group]={name:{k:float(np.mean([v[name][k] for v in ps])) for k in ('clock','type','clock_plus_type')} for name in ('unconditional','hawkes','reference','depth')}
    hp=fitted['hawkes']['__pooled__'];result['pooled_hawkes_parameters']=dict(mu=hp.logmu.exp().tolist(),beta=hp.logbeta.exp().tolist(),alpha=hp.logalpha.exp().tolist(),c=hp.c.tolist())
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for group,v in result['macro'].items():
        print(f'\n== {group} (macro, nats/event)')
        for name,x in v.items():print(f"  {name:14s} clock {x['clock']:.3f}  type {x['type']:.3f}  clock+type {x['clock_plus_type']:.3f}")


if __name__=='__main__':main()
