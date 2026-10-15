"""Fees / LVR for Arbitrum and Base WETH/USDC v3 pools, 2025-09-01 .. 2026-09-30, vs mainnet.

Inputs are the verified Swap logs from scripts.collect_l2_swaps and Binance ETHUSDT
1-second closes (scripts.fetch_binance); mainnet comparisons read the hourly ledgers
written by scripts.fee_lvr run with --cex. Per unit of in-range
liquidity, the same swap terms as scripts/fee_lvr.py. LVR references: Binance
close 0/12/60/300/3600 s after the swap's block timestamp. Fees use the pool fee
net of the protocol share read from slot0 (zero on both chains at the start).

Splits inside the year, fixed before any results were seen:
  train 2025-09-01..2026-02-28, val 2026-03-01..2026-05-31, test 2026-06-01..2026-09-30.
Hourly ledgers use the fee_lvr.py layout (hour, split, swaps, value, fee:<ref>,
lvr:<ref>) so scripts.hourly_gate can read them with --key cex+12s.

  python -m scripts.l2_fee_lvr --l2 data/l2 --cex data/cex/ethusdt_1s.parquet \
      --mainnet results/fee-lvr-cex --out results/l2
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dexnet.lvr import l2_swaps

CEX_LAGS = (0, 12, 60, 300, 3600)
HOUR, DAY = 3600, 86400
SPLITS = {'train': ('2025-09-01', '2026-03-01'), 'val': ('2026-03-01', '2026-06-01'), 'test': ('2026-06-01', '2026-10-01')}
MAINNET = ('USDC-WETH-500', 'USDC-WETH-3000', 'USDT-WETH-500', 'USDT-WETH-3000')


def ts(day):
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp())


def split_of(hours):
    out = np.full(len(hours), '', object)
    for s, (a, b) in SPLITS.items():
        out[(hours*HOUR >= ts(a)) & (hours*HOUR < ts(b))] = s
    return out.astype(str)


def pool_ledger(l2, chain, name, cex):
    blocks, times, a, b, fee, share = l2_swaps(l2/chain, name)
    rate = fee/(1-fee)*(1-share)
    dx, dy = 1/a - 1/b, b - a
    hour = times//HOUR
    hours, index = np.unique(hour, return_inverse=True)
    n = len(hours)
    value = np.bincount(index, 2*b, minlength=n)/np.bincount(index, minlength=n)
    table = dict(hour=hours, split=split_of(hours), swaps=np.bincount(index, minlength=n), value=value)
    for s in CEX_LAGS:
        R = np.interp(times+s, cex[0], cex[1], left=np.nan, right=np.nan)
        ok = np.isfinite(R)
        fees = np.where(dy > 0, dy, -dx*R)*rate
        table[f'fee:cex+{s}s'] = np.bincount(index[ok], fees[ok], minlength=n)/value
        table[f'lvr:cex+{s}s'] = np.bincount(index[ok], (dx*R - dy)[ok], minlength=n)/value
    return table, dict(swaps=int(len(b)), protocol_share=float(share[0]) if len(share) else 0.)


def summarise(table, mask, key, rng, draws=4000):
    if not mask.any():
        return None
    f, l = table[f'fee:{key}'][mask], table[f'lvr:{key}'][mask]
    day = table['hour'][mask]*HOUR//DAY
    u, i = np.unique(day, return_inverse=True)
    fd, ld = np.bincount(i, f, minlength=len(u)), np.bincount(i, l, minlength=len(u))
    pick = rng.integers(0, len(u), (draws, len(u)))
    boot = fd[pick].sum(1)/ld[pick].sum(1)
    exposure_days = mask.sum()/24
    return dict(days=int(len(u)), fee_bp_per_day=float(1e4*f.sum()/exposure_days), lvr_bp_per_day=float(1e4*l.sum()/exposure_days),
                fee_over_lvr=float(f.sum()/l.sum()), ci95=[float(x) for x in np.quantile(boot, [.025, .975])])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--l2', type=Path, required=True, help='output folder of scripts.collect_l2_swaps (one subfolder per chain)')
    p.add_argument('--cex', type=Path, required=True)
    p.add_argument('--mainnet', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    c = pq.read_table(a.cex); cex = (c['timestamp'].to_numpy(), c['price'].to_numpy())
    rng = np.random.default_rng(0)
    pools = {k: v for chain in ('arbitrum', 'base') if (a.l2/chain/'pools.json').exists() for k, v in json.loads((a.l2/chain/'pools.json').read_text()).items()}
    result = dict(method=__doc__, splits=SPLITS, pools={})
    key = 'cex+12s'
    print(f"{'pool':24s} {'swaps':>10s} {'fee bp/d':>9s} {'LVR bp/d':>9s}  fee/LVR [95%]   train | val | test   (vs Binance +12s)")
    tables = {}
    for name, v in pools.items():
        table, info = pool_ledger(a.l2, v['chain'], name, cex)
        tables[name] = table
        pq.write_table(pa.table(table), a.out/f'{name}-hourly.parquet', compression='zstd')
        allm = table['split'] != ''
        r = dict(info, all={k: summarise(table, allm, k, rng) for k in [f'cex+{s}s' for s in CEX_LAGS]},
                 by_split={s: summarise(table, table['split'] == s, key, rng) for s in SPLITS})
        utc = table['hour'] % 24
        r['by_hour_utc'] = {int(h): summarise(table, allm & (utc == h), key, rng, 500) for h in range(24)}
        result['pools'][name] = r
        x = r['all'][key]
        print(f"{name:24s} {info['swaps']:10d} {x['fee_bp_per_day']:9.2f} {x['lvr_bp_per_day']:9.2f}  {x['fee_over_lvr']:.2f} [{x['ci95'][0]:.2f},{x['ci95'][1]:.2f}]   "
              + ' | '.join(f"{r['by_split'][s]['fee_over_lvr']:.2f}" for s in SPLITS))
    # Mainnet over the hours it covers inside the same year, and L2 restricted to those same hours.
    result['mainnet'] = {}
    common = None
    for name in MAINNET:
        h = pq.read_table(a.mainnet/f'{name}-hourly.parquet').to_pandas()
        h = h[(h.hour*HOUR >= ts('2025-09-01')) & (h.hour*HOUR < ts('2026-10-01'))]
        common = set(h.hour) if common is None else common & set(h.hour)
        t = {k: h[k].to_numpy() for k in ('hour', f'fee:{key}', f'lvr:{key}')}
        x = summarise(t, np.ones(len(h), bool), key, rng)
        result['mainnet'][name] = x
        print(f"{'mainnet '+name:24s} {'':10s} {x['fee_bp_per_day']:9.2f} {x['lvr_bp_per_day']:9.2f}  {x['fee_over_lvr']:.2f} [{x['ci95'][0]:.2f},{x['ci95'][1]:.2f}]   {x['days']} days")
    result['l2_on_mainnet_days'] = {}
    last_common = max(common)
    for name, table in tables.items():
        m = (table['split'] != '') & (table['hour'] <= last_common)
        result['l2_on_mainnet_days'][name] = summarise(table, m, key, rng)
    (a.out/'l2_fee_lvr.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
