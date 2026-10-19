"""Paper tables from paper/results: LaTeX (paper/tables/*.tex) and one Markdown file for review.

  python paper/tables/make_tables.py
"""
import json,re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];FINAL=ROOT/'results'/'final';OUT=ROOT/'tables'
SEEDS=(260930,260931,260932)
FAMILIES=(('depth','DEX-Net'),('reference','No profile'),('depth-small','DEX-Net-small'),('depth-hist8','DEX-Net, 8 events'),('depth-hist0','DEX-Net, no history'))
HEADS=(('wait','Clock'),('kind','Event type'),('swap_sell','Sell size'),('swap_buy','Buy size'),('mint_lower','Mint lower'),('mint_width','Mint width'),
       ('mint_amount','Mint amount'),('burn_lower','Burn lower'),('burn_width','Burn width'),('burn_amount','Burn amount'),('joint','Joint'))
md=[]


def load(p):return json.loads(Path(p).read_text())


def write(name,caption,header,rows,align=None):
    """One table as LaTeX (booktabs) and as Markdown."""
    align=align or 'l'+'r'*(len(header)-1)
    tex=['\\begin{table}[t]','\\centering','\\small',f'\\caption{{{caption}}}',f'\\label{{tab:{name}}}','\\begin{adjustbox}{max width=\\linewidth}',f'\\begin{{tabular}}{{{align}}}','\\toprule',
         ' & '.join(header)+' \\\\','\\midrule']
    for r in rows:tex.append('\\midrule' if r=='---' else ' & '.join(str(c) for c in r)+' \\\\')
    tex+=['\\bottomrule','\\end{tabular}','\\end{adjustbox}','\\end{table}']
    OUT.mkdir(exist_ok=True);(OUT/f'{name}.tex').write_text('\n'.join(tex).replace('±','$\\pm$').replace('×','$\\times$').replace('%','\\%').replace('−','$-$'))
    (OUT/f'{name}.tex').write_text(re.sub(r'\*\*(.+?)\*\*',r'\\textbf{\1}',(OUT/f'{name}.tex').read_text())+'\n')
    md.extend([f'### {caption}','','| '+' | '.join(header)+' |','|'+'---|'*len(header)]+['| '+' | '.join(str(c) for c in r)+' |' for r in rows if r!='---']+[''])
    print('wrote',name)


def per_seed(family,file,pool,key='joint'):
    return np.array([load(FINAL/'runs'/f'{family}-seed{s}'/'training'/file)[pool]['loss'][key] for s in SEEDS])


def fmt(x,d=3):return f'{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}'


def table1():
    d=load(FINAL/'dataset-table.json');rows=[]
    for name,v in sorted(d.items(),key=lambda t:(t[1]['held_out'],-t[1]['events'])):
        s=v['splits'];pair,fee=name.rsplit('-',1)
        rows.append([pair.replace('-','/'),f'{int(fee)/1e4:g}%','held out' if v['held_out'] else 'fit',f"{v['events']:,}",f"{v['sell']+v['buy']:,}",f"{v['mint']:,}",f"{v['burn']:,}",
                     f"{v['first']} – {v['last']}",f"{s['train']['rows']:,}" if 'train' in s else '—',f"{s['val']['rows']:,}",f"{s['test']['rows']:,}"])
    tot=[sum(v[k] for v in d.values()) for k in ('events','sell','buy','mint','burn')]
    rows+=['---',['Total','','',f'{tot[0]:,}',f'{tot[1]+tot[2]:,}',f'{tot[3]:,}',f'{tot[4]:,}','','','','']]
    write('dataset','Pools, event counts and chronological splits (model rows). Held-out pools contribute no training rows.',
          ['Pool','Fee','Role','Events','Swaps','Mints','Burns','Dates','Train','Val','Test'],rows,'lllrrrrlrrr')


def table2():
    pools=load(FINAL/'dataset-table.json');rows=[]
    for name,v in sorted(pools.items(),key=lambda t:(t[1]['held_out'],-t[1]['events'])):
        file='heldout_test.json' if v['held_out'] else 'test.json'
        cells=[]
        for fam,_ in FAMILIES:
            x=per_seed(fam,file,name);cells.append(f'{x.mean():.3f} ± {x.std(ddof=1):.3f}' if x.max()<100 else f'{x.mean():.1f} ± {x.std(ddof=1):.1f}')
        pair,fee=name.rsplit('-',1);rows.append([f"{pair.replace('-','/')} {int(fee)/1e4:g}%"+(' (held out)' if v['held_out'] else '')]+cells)
    nll=load(FINAL/'nll.json')['families']
    rows.append('---')
    for split,label in (('seen','Macro, seen pools'),('heldout','Macro, held-out pools')):
        rows.append([label]+[f"{nll[f][split]['mean']:.3f} ± {nll[f][split]['sd']:.3f}" for f,_ in FAMILIES])
    write('nll_by_pool','Next-event NLL (nats/event) on the test period, mean ± s.d. over 3 seeds; 32,768 fixed examples per pool.',
          ['Pool']+[n for _,n in FAMILIES],rows)


def table3():
    rows=[]
    for key,label in HEADS:
        row=[label]
        for split,file in (('seen','test.json'),('heldout','heldout_test.json')):
            for fam in ('depth','reference'):
                r=[load(FINAL/'runs'/f'{fam}-seed{s}'/'training'/file) for s in SEEDS]
                x=np.array([np.mean([v['loss'][key] for v in rr.values()]) for rr in r]);row.append(f'{x.mean():.3f}')
        rows.append(row)
    write('nll_by_head','Macro next-event NLL by model component (nats per event of that type; Clock, Event type and Joint per event), mean over 3 seeds.',
          ['Component','DEX-Net (seen)','No profile (seen)','DEX-Net (held out)','No profile (held out)'],rows)


def table4():
    rows=[]
    for split,label in (('test','Test'),('val','Validation')):
        T=load(FINAL/f'rollout-table-{split}.json')['cells']
        for group in ('heldout','seen'):
            for h in ('32','128','512'):
                c=T[f'{group}:{h}'];e=c['energy']
                vals=[e['depth'],e['reference'],e['depth-small'],e['recent'],e['expanding']];best=int(np.argmin(vals))
                rows.append([label,'held out' if group=='heldout' else 'seen',h,c['anchors']]+[f'**{v:.2f}**' if i==best else f'{v:.2f}' for i,v in enumerate(vals)])
            rows.append('---')
    write('rollout_energy','Rollout energy score (lower is better) at 32/128/512 events on all anchors; a seed’s forecast is the ensemble of its paths that reached the horizon (failed paths, at most 0.3%, are dropped); models are the mean of 3 seeds × 32 paths; best per row in bold.',
          ['Period','Pools','Events','Anchors','DEX-Net','No profile','DEX-Net-small','Recent','Expanding'],rows[:-1],'lllrrrrrr')
    # Paired differences, depth minus reference.
    rows=[]
    for split in ('test','val'):
        c=load(FINAL/f'compare-{split}-depth-vs-reference.json')['cells'];T=load(FINAL/f'rollout-table-{split}.json')['cells']
        for k,v in c.items():
            e=T[k]['differences']['depth-reference'];col=v['collapse'];sp=v['spread_ratio'];g,h=k.split(':')
            rows.append([split,'held out' if g=='heldout' else 'seen',h,f"{e['mean']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]",
                         f"{100*col['real']['share_lost_63pct']:.1f}%",f"{100*col['depth']['share_lost_63pct']:.1f}%",f"{100*col['reference']['share_lost_63pct']:.1f}%",
                         f"{sp['depth']:.2f}×",f"{sp['reference']:.2f}×"])
    write('depth_vs_reference','DEX-Net minus the no-profile model: paired energy difference with 95% bootstrap CI over the Table 4 anchors; share of paths losing >63% of active liquidity; 5–95% return-spread ratio (simulated / realised).',
          ['Period','Pools','Events','Δ energy [95% CI]','Collapse real','Collapse DEX-Net','Collapse no profile','Spread DEX-Net','Spread no profile'],rows,'lllrrrrrr')


def table5():
    h=load(FINAL/'hawkes.json');rows=[]
    names=(('unconditional','Unconditional (renewal + type frequencies)'),('hawkes','4-type Hawkes, 2 time scales'),('depth','DEX-Net'),('reference','No profile (neural)'))
    for key,label in names:
        row=[label]
        for g in ('seen','heldout'):
            m=h['macro'][g][key];row+=[f"{m['clock']:.3f}",f"{m['type']:.3f}",f"{m['clock_plus_type']:.3f}"]
        rows.append(row)
    write('point_process_baselines','Clock and event-type NLL (nats/event, macro over pools) on the same evaluation rows. Point-process baselines are fitted per seen pool; held-out pools use one fit on pooled seen-pool data. Neural rows are 3-seed means.',
          ['Model','Clock (seen)','Type (seen)','Sum (seen)','Clock (held out)','Type (held out)','Sum (held out)'],rows,'lrrrrrr')


def table6():
    rows=[]
    for split,label in (('test','Test'),('val','Validation')):
        c=load(FINAL/f'lp-{split}.json')['cells']
        for group in ('heldout','seen'):
            for h in ('32','128','512'):
                v=c[f'{group}:{h}:0.01'];sc=v['scores'];d=v['pnl_crps_difference']['depth-recent']
                vals=[sc[k]['crps_pnl_bp'] for k in ('depth','reference','depth-small','recent','expanding')];best=int(np.argmin(vals))
                rows.append([label,'held out' if group=='heldout' else 'seen',h,v['anchors'],f"{v['real_median']['fee_bp']:.1f}"]+[f'**{x:.2f}**' if i==best else f'{x:.2f}' for i,x in enumerate(vals)]
                            +[f"{d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]",f"{100*sc['depth']['pnl_coverage90']:.0f}%",f"{sc['depth']['crps_fee_bp']:.1f} / {sc['recent']['crps_fee_bp']:.1f}"])
            rows.append('---')
    write('lp_eval','LP-facing evaluation: a hypothetical marginal position over \u00b11% around the anchor price. CRPS (bp of position value, lower is better) of P&L versus holding; DEX-Net minus the recent baseline with 95% bootstrap CI; DEX-Net 90%-interval coverage of the realized P&L; fee-income CRPS for DEX-Net / recent baseline. Real median fee in bp. Gross of any protocol fee share.',
          ['Period','Pools','Events','Anchors','Real fee','DEX-Net','No profile','DEX-Net-small','Recent','Expanding','DEX-Net \u2212 recent','Coverage','Fee CRPS'],rows[:-1],'lllrrrrrrrrrr')


def main():
    table1();table2();table3();table4();table5();table6()
    (OUT/'TABLES.md').write_text('# Paper tables\n\nGenerated by `paper/tables/make_tables.py` from `paper/results/final/`.\n\n'+'\n'.join(md))


if __name__=='__main__':main()
