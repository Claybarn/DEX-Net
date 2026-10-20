"""Fit post-hoc path reweighting on validation rollouts and score it on test rollouts.

Each simulated path keeps its events; only its weight in the forecast changes.
Path features are the path's endpoints at every horizon relative to its own
ensemble median (signed and absolute, scaled by the typical fit-set ensemble
spread), plus a failure flag. Two weight models (dexnet.reweighting.PathWeighter):

  global       one coefficient vector shared by all anchors
  conditional  coefficients adjusted by anchor context: --context absolute (default)
               uses the log recent realised spread of returns at each horizon from
               the history-only windows; --context relative uses the log ratio of
               that spread to the pool's long-run spread (expanding windows)

Protocol: fit on the earliest 75% (by time) of seen-pool validation anchors; choose
the KL penalty per model on the latest 25%; average three fitting seeds. Held-out
pools are never used for fitting or selection. The objective sums, over horizons,
the weighted energy score normalised by the uniform-weight score.

Test scoring uses one formula for everything. --estimator weighted (default): the
proper score of the issued finite weighted distribution (V-statistic, including the
diagonal). --estimator fair: the weighted analogue of the fair (unbiased) ensemble
estimator, sum_{i!=j} w_i w_j d_ij / (1 - sum w_i^2) for the spread term, which equals
the usual fair estimator for uniform weights and does not penalise small ensembles
against the 256-window baselines. --weights-out saves each test path's weight
(global model). Reported per pool
group and horizon: energy (four endpoints) and price CRPS (bp), paired bootstrap
of reweighted minus uniform, effective sample size and maximum weight.

  python -m scripts.path_reweight --fit eval/rollouts-val --fit-anchors eval/anchors-val \
      --test eval/rollouts-test --test-anchors eval/anchors-test --out results/reweight.json
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from dexnet.diagnostics import ENDPOINT_SCALES
from dexnet.reweighting import PathWeighter,energy as weighted_energy,diagnostics


CONTEXT='absolute'


def extract(rollouts,anchors,seen_only):
    """One record per (anchor, model seed) ensemble, plus baseline ensembles for scoring."""
    out=[]
    for f in sorted((rollouts/'anchors').glob('*.json')):
        r=json.loads(f.read_text());held=r['split'].startswith('heldout')
        if seen_only and held:continue
        anchor=json.loads((anchors/r['pool']/f"{r['row']}.json").read_text())
        hs=sorted(int(h) for h in r['actual'])
        context=[]
        for h in hs:
            rec=np.asarray(anchor['baselines']['recent'].get(str(h)) or anchor['baselines']['recent'].get(h) or [[0]*4],float)[:,0]
            spread=max(np.subtract(*np.quantile(rec,[.75,.25])),1e-3)
            if CONTEXT=='relative':
                ex=np.asarray(anchor['baselines']['expanding'].get(str(h)) or anchor['baselines']['expanding'].get(h) or [[0]*4],float)[:,0]
                spread/=max(np.subtract(*np.quantile(ex,[.75,.25])),1e-3)
            context.append(np.log(spread))
        y=np.array([r['actual'][str(h)] for h in hs],float)
        for label,paths in r['paths'].items():
            x=np.full((len(paths),len(hs),4),np.nan)
            for i,q in enumerate(paths):
                for j,h in enumerate(hs):
                    if str(h) in q['values']:x[i,j]=q['values'][str(h)]
            out.append(dict(pool=r['pool'],row=r['row'],group='heldout' if held else 'seen',time=anchor.get('initial_timestamp',0),
                            label=label,x=x,y=y,context=np.array(context),horizons=hs))
    return out


def features(x,scale):
    """Ensemble-relative signed and absolute deviations at every horizon and endpoint, plus failure flag."""
    med=np.nanmedian(x,axis=0,keepdims=True)
    z=np.nan_to_num((x-med)/scale[None]);fail=np.isnan(x).any((1,2)).astype(float)[:,None]
    return np.concatenate([z.reshape(len(x),-1),np.abs(z).reshape(len(x),-1),fail],1)


def tensors(records,scale):
    f=torch.tensor(np.stack([features(r['x'],scale) for r in records]),dtype=torch.float32)
    c=torch.tensor(np.stack([r['context'] for r in records]),dtype=torch.float32)
    return f,c


def horizon_terms(records):
    """Per-horizon distance tensors for complete ensembles: to-truth (B,P) and pairwise (B,P,P)."""
    terms=[]
    for j in range(len(records[0]['horizons'])):
        keep=[i for i,r in enumerate(records) if not np.isnan(r['x'][:,j]).any()]
        u=torch.tensor(np.stack([records[i]['x'][:,j]/ENDPOINT_SCALES for i in keep]),dtype=torch.float32)
        v=torch.tensor(np.stack([records[i]['y'][j]/ENDPOINT_SCALES for i in keep]),dtype=torch.float32)
        terms.append((torch.tensor(keep),torch.linalg.vector_norm(u-v[:,None],dim=-1),torch.cdist(u,u)))
    return terms


def objective(weights,terms,normalizers):
    total=0.
    for (keep,to_truth,pairwise),norm in zip(terms,normalizers):
        total=total+weighted_energy(weights[keep],to_truth,pairwise).mean()/norm
    return total/len(terms)


def fit(feats,ctx,terms,normalizers,conditional,kl,seed,steps=400):
    torch.manual_seed(seed);model=PathWeighter(feats.shape[-1],ctx.shape[-1],conditional=conditional)
    opt=torch.optim.Adam(model.parameters(),lr=.02)
    for _ in range(steps):
        w=model(feats,ctx);d=diagnostics(w)
        loss=objective(w,terms,normalizers)+kl*d['kl_to_uniform'].mean()
        opt.zero_grad();loss.backward();opt.step()
    return model


def ensemble_weights(models,feats,ctx):
    with torch.no_grad():
        w=torch.stack([m(feats,ctx) for m in models]).mean(0)
    return w/w.sum(-1,keepdim=True)


FAIR=False


def scores(record,w,j):
    """Weighted proper scores at horizon j for one ensemble (None if incomplete)."""
    x=record['x'][:,j]
    if np.isnan(x).any():return None
    y=record['y'][j];u=x/ENDPOINT_SCALES;v=y/ENDPOINT_SCALES
    to=np.linalg.norm(u-v,axis=-1);pair=np.linalg.norm(u[:,None]-u[None],axis=-1)
    k=1/(1-float(w@w)) if FAIR else 1.
    e=float(w@to-.5*k*w@pair@w)
    r=x[:,0];crps=float(w@np.abs(r-y[0])-.5*k*w@np.abs(r[:,None]-r[None])@w)
    return e,crps


def bootstrap(d,rng,draws=4000):
    d=np.asarray(d);idx=rng.integers(0,len(d),(draws,len(d)))
    return dict(mean=float(d.mean()),ci95=[float(x) for x in np.quantile(d[idx].mean(1),[.025,.975])],n=len(d))


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--fit-anchors',type=Path,required=True)
    p.add_argument('--test',type=Path,required=True);p.add_argument('--test-anchors',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--estimator',choices=('weighted','fair'),default='weighted');p.add_argument('--weights-out',type=Path)
    p.add_argument('--context',choices=('absolute','relative'),default='absolute');p.add_argument('--weights-model',choices=('global','conditional'),default='global')
    a=p.parse_args();rng=np.random.default_rng(0);torch.set_num_threads(8)
    global FAIR,CONTEXT;FAIR=a.estimator=='fair';CONTEXT=a.context
    fit_all=[r for r in extract(a.fit,a.fit_anchors,seen_only=True) if not r['label'].startswith('baseline:')]
    times=np.array([r['time'] for r in fit_all]);cut=np.quantile(times,.75)
    fit_set=[r for r in fit_all if r['time']<cut];tune_set=[r for r in fit_all if r['time']>=cut]
    iqr=np.stack([np.nanquantile(r['x'],.75,axis=0)-np.nanquantile(r['x'],.25,axis=0) for r in fit_set])
    scale=np.maximum(np.nanmedian(iqr,axis=0),1e-6)
    f_fit,c_fit=tensors(fit_set,scale);f_tune,c_tune=tensors(tune_set,scale)
    t_fit,t_tune=horizon_terms(fit_set),horizon_terms(tune_set)
    uniform=lambda n,k:torch.full((n,k),1./k)
    norm=[float(weighted_energy(uniform(len(keep),to.shape[1]),to,pair).mean()) for keep,to,pair in t_fit]
    chosen={};selection={}
    for name,conditional in (('global',False),('conditional',True)):
        for kl in (.001,.01,.1):
            models=[fit(f_fit,c_fit,t_fit,norm,conditional,kl,seed) for seed in range(3)]
            w=ensemble_weights(models,f_tune,c_tune);score=float(objective(w,t_tune,norm))
            selection[f'{name} kl={kl}']=dict(tune_objective=score,tune_effective_paths=float(diagnostics(w)['effective_paths'].mean()))
            print(f'{name:12s} kl={kl:<6} tune objective {score:.5f}  ESS {selection[f"{name} kl={kl}"]["tune_effective_paths"]:.1f}',flush=True)
            if name not in chosen or score<chosen[name][0]:chosen[name]=(score,kl,models)
    w_uniform=float(objective(uniform(len(tune_set),f_tune.shape[1]),t_tune,norm));print(f'uniform      tune objective {w_uniform:.5f}')
    test=extract(a.test,a.test_anchors,seen_only=False)
    models_test=[r for r in test if not r['label'].startswith('baseline:')];base_test=[r for r in test if r['label'].startswith('baseline:')]
    f_test,c_test=tensors(models_test,scale)
    weights={name:ensemble_weights(m,f_test,c_test).numpy() for name,(_,_,m) in chosen.items()}
    if a.weights_out:
        a.weights_out.parent.mkdir(parents=True,exist_ok=True)
        a.weights_out.write_text(json.dumps({f"{r['pool']}|{r['row']}|{r['label']}":weights[a.weights_model][i].tolist() for i,r in enumerate(models_test)})+'\n')
    result=dict(method=__doc__,estimator=a.estimator,context=a.context,selection=selection,chosen={k:dict(kl=v[1],tune_objective=v[0]) for k,v in chosen.items()},uniform_tune_objective=w_uniform,cells={})
    hs=models_test[0]['horizons']
    for group in ('seen','heldout'):
        for j,h in enumerate(hs):
            cell={};idx=[i for i,r in enumerate(models_test) if r['group']==group]
            uni={};rew={name:{} for name in weights}
            for i in idx:
                r=models_test[i];n=len(r['x']);s=scores(r,np.full(n,1/n),j)
                if s is None:continue
                key=(r['pool'],r['row']);uni.setdefault(key,[]).append(s)
                for name,w in weights.items():rew[name].setdefault(key,[]).append(scores(r,w[i],j))
            keys=sorted(uni)
            if not keys:continue
            def seedmean(d):
                return np.array([np.mean([x[0] for x in d[k]]) for k in keys]),np.array([np.mean([x[1] for x in d[k]]) for k in keys])
            ue,uc=seedmean(uni);cell['model uniform']=dict(energy=float(ue.mean()),crps_bp=float(uc.mean()),anchors=len(keys))
            for name in weights:
                re,rc=seedmean(rew[name]);ess=diagnostics(torch.tensor(weights[name][idx]))
                cell[f'model reweighted ({name})']=dict(energy=float(re.mean()),crps_bp=float(rc.mean()),
                    energy_vs_uniform=bootstrap(re-ue,rng),crps_vs_uniform=bootstrap(rc-uc,rng),
                    mean_effective_paths=float(ess['effective_paths'].mean()),max_weight=float(ess['max_weight'].max()))
            for b in sorted({r['label'] for r in base_test}):
                bs={(r['pool'],r['row']):scores(r,np.full(len(r['x']),1/len(r['x'])),j) for r in base_test if r['label']==b and r['group']==group}
                be=np.array([bs[k][0] for k in keys if bs.get(k)]);bc=np.array([bs[k][1] for k in keys if bs.get(k)])
                if len(be)==len(keys):
                    cell[b]=dict(energy=float(be.mean()),crps_bp=float(bc.mean()))
                    for name in weights:
                        re,rc=seedmean(rew[name]);cell[f'model reweighted ({name})'][f'energy_vs_{b}']=bootstrap(re-be,rng);cell[f'model reweighted ({name})'][f'crps_vs_{b}']=bootstrap(rc-bc,rng)
            result['cells'][f'{group}:{h}']=cell
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for key,cell in result['cells'].items():
        print(f'\n== {key}  ({cell["model uniform"]["anchors"]} anchors; seed-averaged; weighted-distribution scores)')
        for label,v in cell.items():
            line=f"{label:34s} energy {v['energy']:8.3f}  CRPS {v['crps_bp']:8.2f} bp"
            if 'energy_vs_uniform' in v:
                e=v['energy_vs_uniform'];line+=f"  vs uniform {e['mean']:+.3f} [{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}]  ESS {v['mean_effective_paths']:.1f}"
                if 'energy_vs_baseline:recent' in v:
                    e=v['energy_vs_baseline:recent'];line+=f"  vs recent {e['mean']:+.3f} [{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}]"
            print(line)


if __name__=='__main__':main()
