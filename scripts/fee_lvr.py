"""Fee income versus loss-versus-rebalancing (LVR) for marginal in-range liquidity.

For one unit of liquidity that stays in range, every historical swap moves the
pool from sqrt price a to b (human quote per base). The LP pays out / receives

    base  dx = 1/a - 1/b        quote dy = b - a

and earns fee/(1-fee)*(1-protocol share) of the input leg. Valued at a reference
price R, the swap's adverse-selection loss is dx*R - dy (>= 0 when R is the
post-swap pool price). Summed over swaps with R = block-close pool price this is
exactly the cost of rebalancing a delta hedge once per block, i.e. block-grid LVR.
Later pool-price markouts (R = close k blocks later) remove bid-ask bounce, and an
optional CEX price file gives LVR against the external market.

Read pool+0b with care: noise swaps that later revert inflate it, and a pool that
lags inside its no-arbitrage band deflates it. On synthetic arbitrage fixtures the
pool+10b..+300b markouts agree with CEX LVR (and with sigma^2/8 per day), so
pool+50b is the default no-CEX estimate used for the regime and hour tables.

Both fees and LVR scale with position liquidity, so fees/LVR is the same for every
range width while in range. Below 1, passive in-range liquidity loses to arbitrage
before any gas or rebalancing cost; widths and entry timing only change exposure.

Reads the prepared dataset (pools/<name>/decoded/*.parquet, metadata.json,
exposure.json, headers/timestamp.npy). CPU only, streams one decoded part at a time.

  python -m scripts.fee_lvr --data data/prepared --out results/fee-lvr
  python -m scripts.fee_lvr --data ... --pools USDC-WETH-500 --cex binance_ethusdt_1s.csv
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dexnet.lvr import load_swaps, protocol_share, swap_terms, load_cex
BLOCK_MARKOUTS = (0, 1, 10, 50, 300)      # pool close k blocks after the swap's block
CEX_MARKOUTS = (0, 12, 60, 300, 3600)     # seconds after the swap's block timestamp
HOUR = 3600
DAY = 86400


def analyse_pool(data, name, cex=None, protocol=None):
    folder = data/'pools'/name
    meta = json.loads((folder/'metadata.json').read_text())
    exposure = json.loads((folder/'exposure.json').read_text())
    header = json.loads((data/'headers'/'COMPLETE.json').read_text())
    clock = np.load(data/'headers'/'timestamp.npy', mmap_mode='r')
    blocks, logs, a, b = load_swaps(folder, meta)
    times = np.asarray(clock[blocks-header['first_block']], np.int64)
    share = protocol_share(protocol, meta, blocks, logs, a, b)
    fee = meta['fee']/1e6

    # Last swap of each block gives its close; later closes give the pool markouts.
    last = np.r_[blocks[1:] != blocks[:-1], True]
    close_block, close_price = blocks[last], b[last]**2
    terms = {}
    for k in BLOCK_MARKOUTS:
        ref = close_price[np.searchsorted(close_block, blocks+k, side='right')-1]
        fees, loss = swap_terms(a, b, fee, share, ref)
        terms[f'pool+{k}b'] = (fees, loss)
    if cex is not None:
        for s in CEX_MARKOUTS:
            ref = np.interp(times+s, cex[0], cex[1], left=np.nan, right=np.nan)
            fees, loss = swap_terms(a, b, fee, share, ref)
            terms[f'cex+{s}s'] = (fees, loss)

    # Hourly ledger. Normalise by the value of a full-range position with unit L
    # (2*sqrt P), so yields read like a v2 position; concentration scales both legs.
    hour = times//HOUR
    hours, index = np.unique(hour, return_inverse=True)
    value = np.bincount(index, 2*b, minlength=len(hours))/np.bincount(index, minlength=len(hours))
    split = np.full(len(hours), '', object)
    for s in exposure:
        lo, hi = s['first_timestamp']//HOUR, s['last_timestamp']//HOUR
        split[(hours >= lo) & (hours <= hi)] = s['split']
    table = dict(hour=hours, split=split.astype(str), swaps=np.bincount(index, minlength=len(hours)),
                 value=value, protocol_known=np.bincount(index, ~np.isnan(share), minlength=len(hours)))
    for key, (fees, loss) in terms.items():
        ok = np.isfinite(loss)
        table[f'fee:{key}'] = np.bincount(index[ok], fees[ok], minlength=len(hours))/value
        table[f'lvr:{key}'] = np.bincount(index[ok], loss[ok], minlength=len(hours))/value
    regime_labels(table)
    return meta, table


def regime_labels(table):
    """Causal volatility regime: trailing 24h block-grid LVR, terciles fixed on train hours."""
    hours, lvr = table['hour'], table['lvr:pool+0b']
    dense = np.zeros(hours[-1]-hours[0]+1); dense[hours-hours[0]] = lvr
    trailing = np.r_[0, np.cumsum(dense)]
    pos = hours-hours[0]
    past = (trailing[pos]-trailing[np.maximum(pos-24, 0)])  # previous 24 hours, excludes current
    train = (table['split'] == 'train') & (pos >= 24)
    cuts = np.quantile(past[train], [1/3, 2/3]) if train.any() else np.quantile(past[pos >= 24], [1/3, 2/3])
    regime = np.where(pos < 24, 'warmup', np.select([past <= cuts[0], past <= cuts[1]], ['low', 'mid'], 'high'))
    table['regime'] = regime.astype(str); table['trailing_lvr_24h'] = past


def summarise(table, mask, key, rng, draws):
    """Fees, LVR (bp/day of full-range value) and fees/LVR with a day-block bootstrap."""
    if not mask.any():
        return None
    fees, lvr = table[f'fee:{key}'][mask], table[f'lvr:{key}'][mask]
    days = table['hour'][mask]*HOUR//DAY
    unique, day_index = np.unique(days, return_inverse=True)
    f_day = np.bincount(day_index, fees, minlength=len(unique)); l_day = np.bincount(day_index, lvr, minlength=len(unique))
    n = len(unique)
    picks = rng.integers(0, n, size=(draws, n))
    ratio = f_day[picks].sum(1)/np.maximum(l_day[picks].sum(1), 1e-300)
    exposure_days = mask.sum()/24
    return dict(days=int(n), hours=int(mask.sum()), swaps=int(table['swaps'][mask].sum()),
                fee_bp_per_day=float(1e4*fees.sum()/exposure_days), lvr_bp_per_day=float(1e4*lvr.sum()/exposure_days),
                fee_over_lvr=float(fees.sum()/lvr.sum()) if lvr.sum() > 0 else None,
                ci95=[float(x) for x in np.quantile(ratio, [.025, .975])],
                share_days_fee_gt_lvr=float(np.mean(f_day > l_day)))


def report(name, meta, table, rng, draws):
    keys = [k[4:] for k in table if k.startswith('fee:')]
    splits = [s for s in ('train', 'val', 'test') if (table['split'] == s).any()]
    out = dict(pool=name, fee_tier=meta['fee']/1e6, base=meta['base'], quote=meta['quote'],
               protocol_fee_known_fraction=float(table['protocol_known'].sum()/table['swaps'].sum()), by_split={}, by_regime={}, by_hour_utc={}, by_year={})
    for s in splits:
        m = table['split'] == s
        out['by_split'][s] = {k: summarise(table, m, k, rng, draws) for k in keys}
        out['by_regime'][s] = {r: summarise(table, m & (table['regime'] == r), 'pool+50b', rng, draws) for r in ('low', 'mid', 'high')}
    utc = (table['hour'] % 24)
    out['by_hour_utc'] = {int(h): summarise(table, utc == h, 'pool+50b', rng, draws) for h in range(24)}
    year = (1970 + table['hour']*HOUR/(365.2425*DAY)).astype(int)
    out['by_year'] = {int(y): summarise(table, year == y, 'pool+50b', rng, draws) for y in np.unique(year)}
    return out


def ratio(x):
    return '>100' if x > 100 else f'{x:.2f}'


def print_report(r):
    print(f"\n== {r['pool']}  fee {r['fee_tier']*1e4:g} bp  {r['base']}/{r['quote']}  "
          f"protocol fee known for {100*r['protocol_fee_known_fraction']:.0f}% of swaps (gross fees elsewhere)")
    print(f"{'split':6s} {'reference':11s} {'days':>5s} {'fee bp/d':>9s} {'LVR bp/d':>9s} {'fee/LVR':>8s} {'95% CI':>15s} {'days fee>LVR':>12s}")
    for s, rows in r['by_split'].items():
        for k, v in rows.items():
            if v and v['fee_over_lvr'] is not None:
                print(f"{s:6s} {k:11s} {v['days']:5d} {v['fee_bp_per_day']:9.2f} {v['lvr_bp_per_day']:9.2f} {ratio(v['fee_over_lvr']):>8s} "
                      f"{'['+ratio(v['ci95'][0])+', '+ratio(v['ci95'][1])+']':>15s} {100*v['share_days_fee_gt_lvr']:11.0f}%")
    print('volatility regime (trailing 24h, pool+50b markout):')
    for s, rows in r['by_regime'].items():
        print('  '+s+': '+'  '.join(f"{g} {ratio(v['fee_over_lvr'])} [{ratio(v['ci95'][0])},{ratio(v['ci95'][1])}]" for g, v in rows.items() if v and v['fee_over_lvr']))
    hours = {h: v['fee_over_lvr'] for h, v in r['by_hour_utc'].items() if v and v['fee_over_lvr']}
    if hours:
        best = sorted(hours, key=hours.get)[-3:]
        print('hour UTC, best fee/LVR: '+', '.join(f'{h:02d}h {ratio(hours[h])}' for h in reversed(best)))
    print('by year: '+'  '.join(f"{y} {ratio(v['fee_over_lvr'])}" for y, v in r['by_year'].items() if v and v['fee_over_lvr']))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', type=Path, required=True, help='pooled dataset root (contains pools/ and headers/)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pools', default='', help='comma-separated; default all pools')
    p.add_argument('--protocol-dir', type=Path, help='folder of <pool>-protocol.json fee histories')
    p.add_argument('--cex', type=Path, help='external price file for the pools named in --cex-pools')
    p.add_argument('--cex-pools', default='USDC-WETH-500,USDC-WETH-3000')
    p.add_argument('--draws', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    names = a.pools.split(',') if a.pools else sorted(x.name for x in (a.data/'pools').iterdir() if (x/'metadata.json').exists())
    cex = load_cex(a.cex) if a.cex else None
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed); results = []
    for name in names:
        protocol = a.protocol_dir/f'{name}-protocol.json' if a.protocol_dir else None
        meta, table = analyse_pool(a.data, name, cex if name in a.cex_pools.split(',') else None,
                                   protocol if protocol and protocol.exists() else None)
        pq.write_table(pa.table(table), a.out/f'{name}-hourly.parquet', compression='zstd')
        r = report(name, meta, table, rng, a.draws); results.append(r); print_report(r)
    (a.out/'fee_lvr.json').write_text(json.dumps(dict(
        method=__doc__, block_markouts=BLOCK_MARKOUTS, cex_markouts=CEX_MARKOUTS if cex is not None else None,
        cex_file=str(a.cex) if a.cex else None, pools=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
