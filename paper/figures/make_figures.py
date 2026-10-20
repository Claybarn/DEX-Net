"""Paper figures from paper/results (and, for the forecast cones, saved rollout anchors).

  python paper/figures/make_figures.py --cones <folder with depth/ and reference/ anchor JSONs>

Writes PDF (for the paper) and PNG (for review) into paper/figures/. Every number is
read from result files; nothing is typed in by hand.
"""
import argparse,json,glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';OUT=ROOT/'figures'
FINAL=RES/'final'
# Okabe-Ito colours, one per family throughout the paper.
COL=dict(depth='#0072B2',reference='#E69F00',small='#009E73',real='#000000',recent='#999999',expanding='#CC79A7')
NAME=dict(depth='DEX-Net',reference='No profile',small='DEX-Net-small',recent='Recent baseline',expanding='Expanding baseline')
FAM=dict(depth='depth',reference='reference',small='depth-small')
SEEDS=(260930,260931,260932)
plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight','pdf.fonttype':42})


def pool_label(key):
    """'mainnet:USDC-WETH-500' / 'arbitrum-WETH-USDC-100' / 'USDT-WETH-3000' -> ('Ethereum', 'USDC/WETH 0.05%')."""
    key=key.split(':')[-1];parts=key.split('-')
    chain='Ethereum'
    if parts[0] in ('arbitrum','base'):chain=parts[0].capitalize();parts=parts[1:]
    return chain,f"{parts[0]}/{parts[1]} {int(parts[2])/1e4:g}%"


def save(fig,name):
    OUT.mkdir(exist_ok=True)
    for ext in ('pdf','png'):fig.savefig(OUT/f'{name}.{ext}',dpi=200)
    plt.close(fig);print('wrote',name)


def load(path):return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- Figure 1
def fig1():
    fig,ax=plt.subplots(figsize=(7.2,3.2));ax.set_xlim(0,104);ax.set_ylim(-7,47);ax.axis('off')
    def box(x,y,w,h,text,colour,bold=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc=colour,ec='#333333',lw=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=6.3,weight='bold' if bold else 'normal')
    def arrow(x0,y0,x1,y1,text=None,dy=1.2):
        ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='->',lw=.9,color='#333333'))
        if text:ax.text((x0+x1)/2,(y0+y1)/2+dy,text,ha='center',fontsize=6.5,color='#333333')
    learned='#DCEBF7';exact='#F3F3F3';cond='#FFF2CC'
    ax.text(1,45,'Learned (stochastic)',fontsize=7.5,weight='bold',color=COL['depth'])
    ax.text(76,45,'Exact (deterministic)',fontsize=7.5,weight='bold',color='#555555')
    box(1,20,17,11,'Last 64 events\n12 features + masks\nevent kinds',cond)
    box(1,4,17,11,'Pool metadata\n+ liquidity profile\n(17 offsets, ±4%)',cond)
    box(23,8,14,19,'Transformer\nencoder\n(query token)',learned,True)
    box(42,23,18,9,'Clock: same block /\ngeometric mixture',learned)
    box(42,12.5,18,8,'Event type:\nsell · buy · mint · burn',learned)
    box(42,1,18,9,'Marks: size flow,\nrange mixtures, amounts',learned)
    box(63.5,10,20.5,13,'Uniswap v3\nmechanics: tick-\ncrossing swap,\nmint / burn, fees',exact,True)
    box(88,20,15,10,'Pool state:\nprice, active L,\ninventory',exact)
    box(88,4,15,10,'Liquidity book\n(exact profile\nat every step)',exact)
    arrow(18,25.5,23,20);arrow(18,9.5,23,14)
    for y in (27.5,16.5,5.5):arrow(37,17.5,42,y)
    for y in (27.5,16.5,5.5):arrow(60,y,64,16.5)
    arrow(83,18,88,25);arrow(83,14,88,9)
    ax.annotate('',xy=(9.5,31.6),xytext=(95.5,30.6),arrowprops=dict(arrowstyle='->',lw=.9,color=COL['depth'],connectionstyle='arc3,rad=0.14'))
    ax.text(52,40.2,'sampled event appended to history',ha='center',fontsize=6.5,color=COL['depth'])
    ax.annotate('',xy=(9.5,3.5),xytext=(95.5,3.5),arrowprops=dict(arrowstyle='->',lw=.9,color='#555555',connectionstyle='arc3,rad=-0.12'))
    ax.text(52,-6.5,'profile recomputed from the simulated book',ha='center',fontsize=6.5,color='#555555')
    save(fig,'fig1_architecture')


# ---------------------------------------------------------------- Figure 2
def fig2(cones):
    files=sorted(glob.glob(str(Path(cones)/'depth'/'*.json')))
    fig,axes=plt.subplots(1,len(files),figsize=(7.2,2.1),sharex=True)
    for ax,f in zip(np.atleast_1d(axes),files):
        name=Path(f).name;bands={}
        for fam,ls in (('depth','-'),('reference','--')):
            r=load(Path(cones)/fam/name)
            paths=[np.r_[0.,p['trace']['return_bp']] for lab,ps in r['paths'].items() if not lab.startswith('baseline:') for p in ps if p['failure'] is None]
            x=np.arange(1+min(map(len,paths))-1);m=np.array([p[:len(x)] for p in paths])
            lo,hi=np.percentile(m,[5,95],0);c=COL[fam];bands[fam]=(lo,hi)
            if fam=='depth':
                ax.fill_between(x,lo,hi,color=c,alpha=.18,lw=0,label='DEX-Net 5–95%')
                q25,q75=np.percentile(m,[25,75],0);ax.fill_between(x,q25,q75,color=c,alpha=.3,lw=0,label='DEX-Net 25–75%')
            else:
                ax.plot(x,lo,ls,color=c,lw=.9,label='No profile 5–95%');ax.plot(x,hi,ls,color=c,lw=.9)
        real=np.r_[0.,r['actual_trace']['return_bp']];ax.plot(np.arange(len(real)),real,color='k',lw=1,label='Realized')
        # Axis from the depth-aware band and the realized path; reference beyond it is reported, not drawn off-scale.
        lim=1.3*max(np.abs(bands['depth'][0]).max(),np.abs(bands['depth'][1]).max(),np.abs(real).max())
        ax.set_ylim(-lim,lim);rlo,rhi=bands['reference']
        note=f"\nno profile to {rlo.min():+,.0f} / {rhi.max():+,.0f} bp" if rlo.min()<-lim or rhi.max()>lim else '\n'
        held='held-out' if r['split'].startswith('heldout') else 'seen'
        ax.set_title(f"{r['pool']} ({held})\nrow {r['row']}{note}",fontsize=6.3);ax.set_xlabel('events after anchor')
        ax.axhline(0,color='#bbbbbb',lw=.5,zorder=0)
    np.atleast_1d(axes)[0].set_ylabel('cumulative return (bp)')
    fig.legend(*np.atleast_1d(axes)[1].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,fontsize=6.5,bbox_to_anchor=(.5,-.2))
    fig.subplots_adjust(wspace=.35)
    fig.suptitle('Forecast cones at four randomly drawn test anchors (3 seeds × 32 paths per model)',fontsize=8,y=1.13)
    save(fig,'fig2_forecast_cones')


# ---------------------------------------------------------------- Figure 3
def fig3():
    data={f:load(FINAL/f'stylized-facts-{f}.json') for f in ('depth','reference')}
    hs=data['depth']['horizons']
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.2))
    for row,group in enumerate(('seen','heldout')):
        real=data['depth']['groups'][group]['real']
        # (a) volatility ratio by horizon
        ax=axes[row,0]
        for fam,d in data.items():
            g=d['groups'][group];ratios=np.array([[g[f'{fam}{s}']['volatility'][str(h)]['width_5_95']/real['volatility'][str(h)]['width_5_95'] for h in hs] for s in SEEDS])
            ax.plot(hs,ratios.mean(0),'o-',ms=2.5,color=COL[fam],label=NAME[fam]);ax.fill_between(hs,ratios.min(0),ratios.max(0),color=COL[fam],alpha=.2,lw=0)
        ax.axhline(1,color='k',lw=.6);ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_yticks([.8,1,1.2,1.4] if group=='seen' else [1,2,5,10]);ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_:f'{v:g}×'));ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax.set_ylabel(f'{"Seen" if group=="seen" else "Held-out"} pools\n5–95% width, sim / real')
        if row==0:ax.set_title('Return spread vs horizon');ax.legend(frameon=False)
        if row==1:ax.set_xlabel('events after anchor')
        # (b) volatility clustering
        ax=axes[row,1];lags=list(real['abs_return_acf'])
        ax.plot([int(l) for l in lags],[real['abs_return_acf'][l] for l in lags],'ks-',ms=3,lw=1,label='Realized')
        for fam,d in data.items():
            g=d['groups'][group];v=np.array([[g[f'{fam}{s}']['abs_return_acf'][l] for l in lags] for s in SEEDS])
            ax.plot([int(l) for l in lags],v.mean(0),'o-',ms=2.5,color=COL[fam],label=NAME[fam])
        if row==0:ax.set_title('Volatility clustering: ACF of |return|');ax.legend(frameon=False)
        if row==1:ax.set_xlabel('lag (events)')
        # (c) event mix and clock
        ax=axes[row,2];labels=['sell','buy','mint','burn','same\nblock'];w=.27
        vals={'real':real['mix']+[real['same_block']]}
        for fam,d in data.items():
            g=d['groups'][group];vals[fam]=list(np.mean([g[f'{fam}{s}']['mix']+[g[f'{fam}{s}']['same_block']] for s in SEEDS],0))
        for i,(k,v) in enumerate(vals.items()):
            ax.bar(np.arange(5)+(i-1)*w,v,w,color=COL[k],label='Realized' if k=='real' else NAME[k])
        ax.set_xticks(range(5));ax.set_xticklabels(labels)
        if row==0:ax.set_title('Event mix and same-block share');ax.legend(frameon=False,loc='upper right',fontsize=6,bbox_to_anchor=(1,1.02))
        ax.set_ylim(0,.62)
    fig.tight_layout();save(fig,'fig4_stylized_facts')



# ---------------------------------------------------------------- Figure 4
def fig4():
    d=load(FINAL/'jit.json')['groups'];fams=dict(real=['real'],depth=[f'depth{s}' for s in SEEDS],reference=[f'reference{s}' for s in SEEDS])
    label=dict(real='Realized',depth=NAME['depth'],reference=NAME['reference'])
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.4));w=.26
    def vals(g,f,fn):return np.array([v for v in (fn(d[g][l]['all']) for l in fams[f]) if v is not None])
    panels=((lambda s:s['pairs_per_1000_events'],'Strict JIT pairs per 1,000 events',False),
            (lambda s:(s['swap_deposit'] or {}).get('q0.5'),'Median JIT swap / deposit',True),
            (lambda s:s['removal']['unremoved_after_64'],'Large mints not removed\nafter 64 events',False))
    for ax,(fn,title,log) in zip(axes,panels):
        for i,f in enumerate(fams):
            for j,g in enumerate(('seen','heldout')):
                v=vals(g,f,fn)
                if not len(v):continue
                y=v.mean();x=j+(i-1)*w
                ax.bar(x,max(y,1e-5) if log else y,w,color=COL[f],label=label[f] if j==0 else None)
                if len(v)>1:ax.errorbar(x,y,yerr=[[y-v.min()],[v.max()-y]],color='k',lw=.7,capsize=1.5)
        ax.set_xticks([0,1]);ax.set_xticklabels(['seen','held-out']);ax.set_title(title)
        if log:ax.set_yscale('log')
    axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0));axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=6.5,bbox_to_anchor=(.5,-.08))
    fig.suptitle('JIT liquidity: the model learns the pattern but makes it too rare, too large relative to the deposit, and too sticky (bars: seed range)',fontsize=7.5,y=1.04)
    fig.tight_layout();save(fig,'fig3_jit')

# ---------------------------------------------------------------- Figure 5
def fig5():
    nll=load(FINAL/'nll.json')['families'];T=load(FINAL/'rollout-table-test.json')['cells']
    fig,axes=plt.subplots(1,2,figsize=(6.6,2.6))
    for ax,(group,ng,title) in zip(axes,(('seen','seen','Seen pools'),('heldout','heldout','Held-out pools'))):
        c=T[f'{group}:512']
        for f,fam in FAM.items():
            x=np.array([nll[fam][ng]['seeds'][str(s)] for s in SEEDS]);y=np.array([c['seed_energy'][fam][f'{fam}{s}'] for s in SEEDS])
            ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),fmt='o',color=COL[f],ms=5,capsize=2,label=NAME[f])
            ax.scatter(x,y,s=8,color=COL[f],alpha=.5,lw=0)
        for b in ('recent','expanding'):ax.axhline(c['energy'][b],color=COL[b],ls=':',lw=1,label=NAME[b])
        ax.set_title(f"{title} ({c['anchors']} anchors)");ax.set_xlabel('next-event NLL (nats/event)');ax.set_ylabel('512-event energy score')
    axes[0].legend(frameon=False,fontsize=6)
    fig.suptitle('Better next-event likelihood does not mean better simulation (test anchors; points are seeds)',fontsize=8,y=1.03)
    fig.tight_layout();save(fig,'fig5_nll_vs_rollout')


# ---------------------------------------------------------------- Figure 6
def fig6():
    d=load(RES/'economics'/'l2_fee_lvr.json')
    rows=[(*pool_label(k),v) for part in ('mainnet','l2_on_mainnet_days') for k,v in d[part].items()]
    rows.sort(key=lambda r:(r[0],float(r[1].split()[-1][:-1]),r[1]))
    chain_col={'Ethereum':'#555555','Arbitrum':'#0072B2','Base':'#D55E00'}
    fig,ax=plt.subplots(figsize=(4.6,3.0))
    for i,(chain,label,v) in enumerate(rows):
        lo,hi=v['ci95'];ax.errorbar(v['fee_over_lvr'],i,xerr=[[v['fee_over_lvr']-lo],[hi-v['fee_over_lvr']]],fmt='o',color=chain_col[chain],ms=4,capsize=2)
    ax.set_yticks(range(len(rows)));ax.set_yticklabels([f"{c}  {l}" for c,l,_ in rows])
    ax.axvline(1,color='k',lw=.7);ax.set_xlabel('fees ÷ LVR (CEX markouts, 95% CI over days)')
    ax.set_title(f"Fees roughly equal LVR across chains ({rows[0][2]['days']} common days)")
    ax.invert_yaxis();save(fig,'fig6_fees_vs_lvr')


# ---------------------------------------------------------------- Figure 7
def fig7():
    d=load(RES/'economics'/'ceiling.json');ints=['block','1min','5min','1h','1d']
    chain_col={'mainnet':'#555555','arbitrum':'#0072B2','base':'#D55E00'};ls={'100':':','500':'-','3000':'--'}
    fig,ax=plt.subplots(figsize=(5.6,3.0))
    for key,p in d['pools'].items():
        chain=key.split(':')[0];tier=key.split('-')[-1];y=[p['year'][i]['gas+swap']['bp_per_day'] for i in ints]
        c,l=pool_label(key);ax.plot(range(len(ints)),y,marker='o',ms=3,lw=1,color=chain_col[chain],ls=ls.get(tier,'-'),label=f'{c} {l}')
    ax.set_xticks(range(len(ints)));ax.set_xticklabels(['every\nblock','1 min','5 min','1 hour','1 day'])
    ax.set_yscale('log');ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_:f'{v:g}'))
    ax.set_ylabel('perfect-foresight value (bp/day)\nafter gas and rebalancing swap')
    ax.set_xlabel('decision interval');ax.set_title('The value of timing is concentrated at short intervals')
    ax.legend(frameon=False,fontsize=6,loc='center left',bbox_to_anchor=(1.01,.5));save(fig,'fig7_timing_ceiling')


# ---------------------------------------------------------------- Supplementary: PIT
def figS1():
    fig,axes=plt.subplots(2,3,figsize=(7.2,3.6),sharey=True)
    for col,h in enumerate(('32','128','512')):
        for row,group in enumerate(('seen','heldout')):
            ax=axes[row,col]
            for f,fam in (('depth','depth'),('reference','reference')):
                c=load(FINAL/f'calibration-{fam}.json')['cells'][f'{group}:{h}']['forecasters']
                hist=np.sum([c[f'{fam}{s}']['pit_histogram'] for s in SEEDS],0);hist=hist/hist.sum()*10
                ax.step(np.arange(11)/10,np.r_[hist,hist[-1]],where='post',color=COL[f],lw=1.1,label=NAME[f])
            ax.axhline(1,color='k',lw=.6);ax.set_ylim(0,2.2)
            if row==0:ax.set_title(f'{h} events')
            if col==0:ax.set_ylabel(f'{"Seen" if group=="seen" else "Held-out"}\ndensity')
            if row==1:ax.set_xlabel('PIT of realized return')
    axes[0,0].legend(frameon=False,fontsize=6);fig.suptitle('Calibration of the return forecast (PIT; uniform = calibrated; 3 seeds pooled)',fontsize=8,y=1.02)
    fig.tight_layout();save(fig,'figS1_pit')


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('--cones',type=Path);a=p.parse_args()
    fig1();fig3();fig4();fig5();figS1()
    if a.cones:fig2(a.cones)


if __name__=='__main__':main()
