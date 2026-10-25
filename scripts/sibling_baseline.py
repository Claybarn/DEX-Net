"""History baselines borrowed from a sibling pool: forecasting a pool from another pool's past.

For a held-out pool there may be no history of its own. A natural fallback is the history of
a seen pool with the same pair at another fee tier (its sibling). At every anchor of the
target pool, sibling windows are built from sibling events in blocks strictly before the
anchor's last history block (no leakage), with the rules of scripts.prepare_anchors:

  sibling-recent     256 evenly spaced h-event windows within the sibling's last 32h events
  sibling-expanding  256 h-event windows drawn uniformly from all of the sibling's history

Endpoints are the same relative quantities as for every forecaster (return in bp, change in
log1p active liquidity, log1p elapsed blocks, log1p base volume over the initial virtual
reserve), computed from the sibling's post-event states in its decoded event files. Events
the model never sees (Initialize, zero-amount swaps, mints and burns) are dropped, as in
dexnet.events. Each anchor is scored with the fair energy score (32-window subsamples, as for
the other baselines) and compared, paired over anchors, with DEX-Net and the target pool's
own recent baseline.

  python -m scripts.sibling_baseline --data data/prepared --anchors eval/anchors-test \
      --rollouts eval/final/rollouts-test-depth --pair WBTC-WETH-3000=WBTC-WETH-500 \
      --pair USDC-USDT-500=USDC-USDT-100 --out results/sibling-test.json
"""
import argparse,json,math,zlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from dexnet.diagnostics import proper_scores
from scripts.prepare_anchors import window_endpoints,BASELINE_SAMPLES,RECENT_SPAN

HORIZONS=('32','128','512')


def post_states(data,pool):
    """Post-event states of the model-visible events of a pool, in order."""
    folder=Path(data)/'pools'/pool;meta=json.loads((folder/'metadata.json').read_text());base0=meta['base_is_token0']
    rows=dict(block=[],logsqrt=[],logL=[],cumvol=[]);vol=0.
    for f in sorted((folder/'decoded').glob('*.parquet')):
        t=pq.read_table(f,columns=['event','block','amount','amount0','amount1','sqrt_after','liquidity_after']).to_pydict()
        for e,b,a,a0,a1,s,L in zip(t['event'],t['block'],t['amount'],t['amount0'],t['amount1'],t['sqrt_after'],t['liquidity_after']):
            if e=='Initialize':continue
            base=int(a0 if base0 else a1)
            if e=='Swap' and base==0:continue
            if e in ('Mint','Burn') and not int(a or 0):continue
            if e=='Swap':vol+=abs(base)
            s=int(s);L=int(L)
            rows['block'].append(int(b));rows['logsqrt'].append(math.log(s) if s>0 else np.nan);rows['logL'].append(math.log(L) if L>0 else np.nan);rows['cumvol'].append(vol)
    post={k:np.asarray(v,float) for k,v in rows.items()};post['log1pL']=np.logaddexp(0.,post['logL'])
    return post,base0


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--anchors',type=Path,required=True);p.add_argument('--rollouts',type=Path,required=True)
    p.add_argument('--pair',action='append',required=True,help='target=sibling (repeatable)');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rng=np.random.default_rng(0);rows=[]
    for target,sibling in (x.split('=',1) for x in a.pair):
        post,base0=post_states(a.data,sibling);blocks=post['block']
        for f in sorted((a.anchors/target).glob('*.json'),key=lambda f:int(f.stem) if f.stem.isdigit() else -1):
            if not f.stem.isdigit():continue
            anchor=json.loads(f.read_text());rec=json.loads((a.rollouts/'anchors'/f'{target}-{anchor["row"]}.json').read_text())
            k=int(np.searchsorted(blocks,anchor['initial_block'],side='left'))   # sibling events in strictly earlier blocks
            local=np.random.default_rng(zlib.crc32(f'{target}-{anchor["row"]}'.encode()))
            for h in HORIZONS:
                H=int(h);hi=k-H
                if hi<1:continue
                starts=dict(recent=np.unique(np.linspace(max(1,k-RECENT_SPAN*H),hi,BASELINE_SAMPLES).astype(int)),expanding=local.integers(1,hi+1,BASELINE_SAMPLES))
                y=np.asarray(rec['actual'][h],float);row=dict(target=target,sibling=sibling,row=anchor['row'],h=h)
                for name,st in starts.items():
                    w=window_endpoints(post,st,H,base0)
                    if len(w)<2:break
                    row['sibling_'+name]=proper_scores(w[local.choice(len(w),min(32,len(w)),replace=False)],y)[0]
                if 'sibling_expanding' not in row:continue
                row['dexnet']=float(np.mean([proper_scores(np.array([q['values'][h] for q in ps if h in q['values']]),y)[0]
                                              for lab,ps in rec['paths'].items() if not lab.startswith('baseline:') and sum(h in q['values'] for q in ps)>=16]))
                for name in ('recent','expanding'):
                    row['own_'+name]=proper_scores(np.array([q['values'][h] for q in rec['paths']['baseline:'+name]]),y)[0]
                rows.append(row)
    result=dict(method=__doc__,cells={})
    for target in sorted({r['target'] for r in rows})+['all']:
        for h in HORIZONS:
            R=[r for r in rows if r['h']==h and (target=='all' or r['target']==target)]
            if not R:continue
            v={k:np.array([r[k] for r in R]) for k in ('dexnet','own_recent','own_expanding','sibling_recent','sibling_expanding')}
            idx=rng.integers(0,len(R),(4000,len(R)))
            def better(a_,b_):
                """How much better a_ is than b_, relative to b_ (positive: a_ better)."""
                d=v[b_]-v[a_];base=v[b_].mean()
                return dict(relative=float(d.mean()/base),ci95=[float(x/base) for x in np.quantile(d[idx].mean(1),[.025,.975])])
            result['cells'][f'{target}:{h}']=dict(anchors=len(R),energy={k:float(x.mean()) for k,x in v.items()},
                dexnet_vs_sibling_recent=better('dexnet','sibling_recent'),own_recent_vs_sibling_recent=better('own_recent','sibling_recent'),
                dexnet_vs_own_recent=better('dexnet','own_recent'))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1)+'\n')
    for k,c in result['cells'].items():
        e=c['energy'];f=lambda d:f"{100*d['relative']:+.1f}% [{100*d['ci95'][0]:+.1f}%, {100*d['ci95'][1]:+.1f}%]"
        print(k,c['anchors'],'| DEX-Net %.2f own-recent %.2f sibling-recent %.2f sibling-expanding %.2f'%(e['dexnet'],e['own_recent'],e['sibling_recent'],e['sibling_expanding']),
              '| DEX-Net better than sibling-recent by',f(c['dexnet_vs_sibling_recent']),'| own-recent better than sibling-recent by',f(c['own_recent_vs_sibling_recent']))


if __name__=='__main__':main()
