"""Perfect-foresight ceiling on LP in/out timing, per decision interval.

Every swap contributes y = fee - LVR for a position of half-width W (log) centred
on the Binance ETHUSDT close 12 s after its block: each swap's price move is
clipped to that range before applying the scripts/fee_lvr.py terms, so swaps far
outside the range (e.g. through empty liquidity) earn and lose nothing. y is in
bp of the position's value at the reference price. Swaps are
summed into decision intervals of length D (one block, 1 min, 5 min, 1 h, 1 day).
A dynamic programme then picks the best in/out schedule knowing every interval's
outcome in advance, paying an entry cost per cycle (exit free):

  out[k] = max(out[k-1], in[k-1])
  in[k]  = max(in[k-1], out[k-1] - cost) + Y[k]

No forecaster can beat this. Block-level timing implies builder/JIT-style access;
minute-level is reachable for an ordinary LP on an L2, not on mainnet.

  python -m scripts.timing_ceiling --data data/prepared --protocol-dir data/protocol --l2 data/l2 \
      --cex data/cex/ethusdt_1s.parquet --out results/ceiling.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from dexnet.lvr import mainnet_swaps, l2_swaps

START, END = 1756684800, 1790812800          # 2025-09-01 .. 2026-10-01 UTC (Binance coverage)
SPLIT_TEST = 1780272000                      # 2026-06-01 UTC
INTERVALS = {'block': None, '1min': 60, '5min': 300, '1h': 3600, '1d': 86400}
GAS = {'mainnet': 15., 'base': .10, 'arbitrum': .10}
WIDTH = .02


def per_swap(blocks, times, a, b, fee, share, cex):
    keep = (times >= START) & (times < END)
    blocks, times, a, b, share = blocks[keep], times[keep], a[keep], b[keep], share[keep]
    R = np.interp(times+12, cex[0], cex[1], left=np.nan, right=np.nan)
    ok = np.isfinite(R)
    blocks, times, a, b, share, R = blocks[ok], times[ok], a[ok], b[ok], share[ok], R[ok]
    lo, hi = np.sqrt(R*np.exp(-WIDTH)), np.sqrt(R*np.exp(WIDTH))
    a, b = np.clip(a, lo, hi), np.clip(b, lo, hi)
    dx, dy = 1/a - 1/b, b - a
    fees = np.where(dy > 0, dy, -dx*R)*fee/(1-fee)*(1-share)
    value = np.sqrt(R)*(2 - 2*np.exp(-WIDTH/2))          # unit-L position value at the range centre
    return blocks, times, 1e4*(fees - (dx*R - dy))/value


def ceiling(Y, cost):
    """Best total with perfect foresight; returns (total, intervals in, entries)."""
    out_v, in_v = 0., -np.inf
    out_from, in_from = [], []                 # back-pointers: True = came from the other state
    for y in Y.tolist():
        enter = out_v - cost
        new_in = (in_v if in_v >= enter else enter) + y
        in_from.append(in_v < enter)
        stay_out = out_v >= in_v
        out_from.append(not stay_out)
        out_v, in_v = (out_v if stay_out else in_v), new_in
    # Backtrack the schedule.
    state = in_v > out_v
    total = max(in_v, out_v)
    held, entries = 0, 0
    for k in range(len(Y)-1, -1, -1):
        if state:
            held += 1
            if in_from[k]:
                entries += 1; state = False
        else:
            if out_from[k]:
                state = True
    return total, held, entries


def analyse(blocks, times, y, width, gas, capital, tier):
    days = (times.max() - times.min())/86400
    scale = 1.                                 # y is already bp of position value
    out = {'always': dict(bp_per_day=float(y.sum()*scale/days))}
    for label, D in INTERVALS.items():
        key = blocks if D is None else times//D
        u, idx = np.unique(key, return_inverse=True)
        Y = np.bincount(idx, y)*scale
        row = {}
        for scen, cost in (('gas', gas/capital*1e4), ('gas+swap', gas/capital*1e4 + .5*tier*1e4)):
            total, held, entries = ceiling(Y, cost)
            span = (u[-1]-u[0]+1)
            row[scen] = dict(bp_per_day=float(total/days), intervals_held=int(held), active_intervals=int(len(u)),
                             entries_per_day=float(entries/days),
                             share_of_time_in=float(held/span) if D is not None else None)
        row['no_cost_bp_per_day'] = float(np.maximum(Y, 0).sum()/days)
        out[label] = row
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--width', type=float, default=.02)
    p.add_argument('--capital', type=float, default=100_000.)
    p.add_argument('--data', type=Path, required=True, help='prepared mainnet data (scripts.prepare_data)')
    p.add_argument('--protocol-dir', type=Path, help='folder of <pool>-protocol.json fee histories')
    p.add_argument('--l2', type=Path, required=True, help='output of scripts.collect_l2_swaps')
    p.add_argument('--cex', type=Path, required=True, help='Binance 1 s prices (scripts.fetch_binance)')
    a = p.parse_args()
    global WIDTH
    WIDTH = a.width
    c = pq.read_table(a.cex); cex = (c['timestamp'].to_numpy(), c['price'].to_numpy())
    pools = [('mainnet', n) for n in ('USDC-WETH-500', 'USDC-WETH-3000', 'USDT-WETH-500', 'USDT-WETH-3000')]
    pools += [(ch, f'{ch}-WETH-USDC-{f}') for ch in ('base', 'arbitrum') for f in (100, 500, 3000)]
    result = dict(method=__doc__, width=a.width, capital=a.capital, gas_usd=GAS, pools={})
    print(f'bp/day of a ±{a.width:.0%} ${a.capital:,.0f} position; perfect-foresight best schedule per decision interval')
    for chain, name in pools:
        raw = (mainnet_swaps(a.data, name, a.protocol_dir/f'{name}-protocol.json' if a.protocol_dir else None) if chain == 'mainnet'
               else l2_swaps(a.l2/chain, name))
        blocks, times, y = per_swap(*raw, cex)
        tier = raw[4]
        r = dict(swaps=int(len(y)))
        r['year'] = analyse(blocks, times, y, a.width, GAS[chain], a.capital, tier)
        t = times >= SPLIT_TEST
        r['test_jun_sep_2026'] = analyse(blocks[t], times[t], y[t], a.width, GAS[chain], a.capital, tier)
        result['pools'][f'{chain}:{name}'] = r
        yr = r['year']
        print(f"\n{chain}:{name}  swaps {len(y):,}  always-in {yr['always']['bp_per_day']:+.1f}")
        for label in INTERVALS:
            g, s = yr[label]['gas'], yr[label]['gas+swap']
            print(f"  {label:6s} gas {g['bp_per_day']:+9.1f} ({g['entries_per_day']:7.1f} entries/d)   gas+swap {s['bp_per_day']:+9.1f} ({s['entries_per_day']:6.1f}/d)   no-cost {yr[label]['no_cost_bp_per_day']:+9.1f}", flush=True)
        a.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
