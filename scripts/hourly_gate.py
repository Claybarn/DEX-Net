"""Can next-hour fees minus LVR be predicted well enough to sit out the bad hours?

Uses the hourly ledgers written by scripts/fee_lvr.py (net of protocol fee; LVR at
the pool+300b markout, which agrees with Binance-referenced LVR on the 2025-26 test
period). Each hour h is decided at its start from hours < h only.

Models (fitted per pool on train, choices made on val, test scored once):
  always   - stay in range every hour
  hour     - in range during hours of day whose train-mean net is positive
  ridge    - ridge regression on causal features; alpha and threshold chosen on val

Units: bp of position value, for a +-W log half-width position held in range
(concentration ~ 2/W times the full-range numbers in the ledgers). Range exits and
recentering are not modelled. Two switching-cost scenarios bracket the truth:
'gas' charges mint+burn gas per cycle only (re-entering with the tokens already
held); 'swap' adds a swap of half the position at the pool fee per cycle.

  python -m scripts.hourly_gate --ledgers results/fee-lvr --out results/gate
  python -m scripts.hourly_gate --ledgers results/l2 --key cex+12s --gas-usd 0.10 --pools base-WETH-USDC-500,...
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

POOLS = ['USDC-WETH-500', 'USDT-WETH-500', 'DAI-WETH-500', 'WBTC-WETH-500',
         'USDC-WETH-3000', 'USDT-WETH-3000', 'DAI-WETH-3000', 'WBTC-WETH-3000']
KEY = 'pool+300b'   # overridden by --key
SPLITS = ('train', 'val', 'test')


def dense(path):
    """Hourly ledger on a gap-free hour grid (hours without swaps are zeros)."""
    t = pq.read_table(path).to_pandas()
    hours = np.arange(t.hour.min(), t.hour.max()+1)
    pos = (t.hour - hours[0]).to_numpy()
    out = dict(hour=hours)
    for c in ('swaps', f'fee:{KEY}', f'lvr:{KEY}'):
        x = np.zeros(len(hours)); x[pos] = t[c].to_numpy(); out[c] = x
    split = np.full(len(hours), '', object); split[pos] = t.split.to_numpy()
    # Empty hours inherit the split of the surrounding hours.
    for s in SPLITS:
        idx = np.flatnonzero(split == s)
        if len(idx): split[idx[0]:idx[-1]+1] = np.where(split[idx[0]:idx[-1]+1] == '', s, split[idx[0]:idx[-1]+1])
    out['split'] = split.astype(str)
    return out


def trailing(x, n):
    """Sum of the n hours strictly before each hour."""
    c = np.r_[0., np.cumsum(x)]
    i = np.arange(len(x))
    return c[i] - c[np.maximum(i-n, 0)]


def features(d, train):
    fee, lvr, swaps = d[f'fee:{KEY}'], d[f'lvr:{KEY}'], d['swaps']
    scale = fee[train].mean() + 1e-12      # hourly LVR can be negative (markout noise): use asinh, not log
    hod = d['hour'] % 24
    dow = (d['hour']//24 + 3) % 7          # 1970-01-01 was a Thursday; 5,6 = Sat, Sun
    cols, names = [], []
    for h in range(1, 24):
        cols.append((hod == h).astype(float)); names.append(f'hod{h}')
    cols.append((dow >= 5).astype(float)); names.append('weekend')
    for n in (1, 6, 24, 168):
        f, l, s = trailing(fee, n), trailing(lvr, n), trailing(swaps, n)
        cols += [np.arcsinh(f/n/scale), np.arcsinh(l/n/scale), np.log1p(s/n), np.arcsinh((f-l)/n/scale)]
        names += [f'fee{n}', f'lvr{n}', f'logswaps{n}', f'net{n}']
    return np.column_stack(cols), names


def ridge(X, y, alpha):
    A = X.T @ X + alpha*np.eye(X.shape[1]); A[0, 0] -= alpha   # no penalty on intercept
    return np.linalg.solve(A, X.T @ y)


def score(y, active, entry_cost, days, draws=2000, seed=0):
    """Net bp/day after switching costs, with a 95% day-block bootstrap interval."""
    enter = np.r_[active[:1], active[1:] & ~active[:-1]]
    hourly = np.where(active, y, 0.) - enter*entry_cost
    n = len(hourly)//24
    daily = hourly[:n*24].reshape(n, 24).sum(1)
    boot = daily[np.random.default_rng(seed).integers(0, n, (draws, n))].mean(1)
    return dict(net_bp_per_day=float(hourly.sum()/days), ci95=[float(x) for x in np.quantile(boot, [.025, .975])],
                before_switching_bp_per_day=float(y[active].sum()/days),
                active_fraction=float(active.mean()), entries_per_day=float(enter.sum()/days))


def run_pool(path, width, capital, gas_usd, scenario):
    d = dense(path)
    X, names = features(d, (d['split'] == 'train') & (np.arange(len(d['hour'])) >= 168))
    conc = 2/width
    y = (d[f'fee:{KEY}'] - d[f'lvr:{KEY}'])*1e4*conc          # bp of position value per hour
    fee_ratio = d[f'fee:{KEY}'], d[f'lvr:{KEY}']
    tier = int(path.name.replace('-hourly.parquet', '').split('-')[-1])/1e6
    entry_cost = (gas_usd/capital + (0.5*tier if scenario == 'swap' else 0.))*1e4   # bp of position value per entry
    warm = np.arange(len(y)) >= 168
    m = {s: (d['split'] == s) & warm for s in SPLITS}
    days = {s: m[s].sum()/24 for s in SPLITS}

    # Standardise on train only.
    mu, sd = X[m['train']].mean(0), X[m['train']].std(0) + 1e-9
    Z = np.column_stack([np.ones(len(y)), (X - mu)/sd])

    out = dict(entry_cost_bp=entry_cost, days={s: float(days[s]) for s in SPLITS}, models={})
    # always on
    out['models']['always'] = {s: score(y[m[s]], np.ones(m[s].sum(), bool), entry_cost, days[s]) for s in SPLITS}
    # hour-of-day rule from train means
    hod = d['hour'] % 24
    good = np.array([y[m['train'] & (hod == h)].mean() > 0 for h in range(24)])
    out['models']['hour'] = {s: score(y[m[s]], good[hod[m[s]]], entry_cost, days[s]) for s in SPLITS}
    out['hour_window_utc'] = [int(h) for h in np.flatnonzero(good)]
    # ridge: alpha and threshold on val (net after switching costs)
    best = None
    for alpha in (1., 10., 100., 1000., 10000.):
        w = ridge(Z[m['train']], y[m['train']], alpha)
        pv = Z[m['val']] @ w
        for q in np.linspace(0, 1, 21):   # q=1 means never enter
            tau = np.quantile(pv, q)
            v = score(y[m['val']], pv > tau, entry_cost, days['val'])['net_bp_per_day']
            if best is None or v > best[0]:
                best = (v, alpha, float(q), float(tau), w)
    _, alpha, q, tau, w = best
    out['models']['ridge'] = {s: score(y[m[s]], (Z[m[s]] @ w) > tau, entry_cost, days[s]) for s in SPLITS}
    out['ridge_choice'] = dict(alpha=alpha, quantile=q, threshold=tau)
    # Test-period fee/LVR inside the ridge-active hours.
    act = (Z[m['test']] @ w) > tau
    f, l = fee_ratio[0][m['test']], fee_ratio[1][m['test']]
    out['test_fee_over_lvr'] = dict(always=float(f.sum()/l.sum()), ridge_active=float(f[act].sum()/max(l[act].sum(), 1e-300)) if act.any() else None)
    top = np.argsort(-np.abs(w[1:]))[:8]
    out['ridge_top_features'] = [(names[i], float(w[1+i])) for i in top]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--ledgers', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--width', type=float, default=.02, help='log half-width of the position')
    p.add_argument('--capital', type=float, default=100_000.)
    p.add_argument('--gas-usd', type=float, default=15., help='mint+burn gas per cycle, USD')
    p.add_argument('--key', default='pool+300b', help='LVR reference column in the ledgers')
    p.add_argument('--pools', default=','.join(POOLS))
    a = p.parse_args()
    global KEY
    KEY = a.key
    a.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for scenario in ('gas', 'swap'):
      results[scenario] = {}
      print(f'\n[{scenario}] bp/day of a ±{a.width:.0%} position held in range, ${a.capital:,.0f}, ${a.gas_usd:g} gas per cycle; net of switching costs')
      print(f"{'pool':15s} {'model':7s} {'train':>8s} {'val':>8s} {'test':>8s} {'active':>7s} {'entries/d':>9s}")
      for pool in a.pools.split(','):
        r = run_pool(a.ledgers/f'{pool}-hourly.parquet', a.width, a.capital, a.gas_usd, scenario); results[scenario][pool] = r
        for k, v in r['models'].items():
            print(f"{pool:15s} {k:7s} " + ' '.join(f"{v[s]['net_bp_per_day']:8.2f}" for s in SPLITS)
                  + f" {v['test']['active_fraction']:7.0%} {v['test']['entries_per_day']:9.2f}"
                  + f"   test 95% [{v['test']['ci95'][0]:.1f}, {v['test']['ci95'][1]:.1f}]")
        print(f"{'':15s} test fee/LVR always {r['test_fee_over_lvr']['always']:.2f}, ridge-active {r['test_fee_over_lvr']['ridge_active'] or float('nan'):.2f}; hour window {r['hour_window_utc']}")
    (a.out/'gate.json').write_text(json.dumps(dict(args={k: str(v) for k, v in vars(a).items()}, method=__doc__, pools=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
