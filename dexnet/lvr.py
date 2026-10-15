"""Per-swap fee income and loss-versus-rebalancing (LVR) for in-range liquidity.

For one unit of liquidity that stays in range, a swap moving the pool from sqrt
price a to b (human quote per base) pays the LP base dx = 1/a - 1/b and receives
quote dy = b - a, and earns fee/(1-fee)*(1-protocol share) of the input leg. Valued
at a reference price R, the swap's adverse-selection loss is dx*R - dy. With R the
post-swap pool price this sums to the cost of rebalancing a delta hedge once per
block (block-grid LVR); later pool or CEX prices give markout-based LVR.
"""
import json
from pathlib import Path
import numpy as np
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

COLUMNS = ['block', 'log_index', 'event', 'sqrt_before', 'sqrt_after']


def human_sqrt(raw, meta):
    """sqrtPriceX96 strings -> sqrt(human quote per base)."""
    native = np.asarray(raw, dtype=float)/float(2**96)
    if meta['base_is_token0']:
        return native*10.**((meta['decimals0']-meta['decimals1'])/2)
    return 10.**((meta['decimals1']-meta['decimals0'])/2)/native


def load_swaps(folder, meta):
    blocks, logs, a, b = [], [], [], []
    for path in sorted((folder/'decoded').glob('*.parquet')):
        t = pq.read_table(path, columns=COLUMNS, filters=[('event', '=', 'Swap')], use_threads=False)
        if not t.num_rows:
            continue
        blocks.append(t['block'].to_numpy()); logs.append(t['log_index'].to_numpy())
        a.append(human_sqrt(t['sqrt_before'].to_pylist(), meta)); b.append(human_sqrt(t['sqrt_after'].to_pylist(), meta))
    if not blocks:
        raise ValueError(f'No swaps in {folder}')
    blocks, logs, a, b = (np.concatenate(x) for x in (blocks, logs, a, b))
    order = np.lexsort((logs, blocks))
    blocks, logs, a, b = blocks[order], logs[order], a[order], b[order]
    if not (np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all()):
        raise ValueError('Invalid swap prices')
    return blocks.astype(np.int64), logs.astype(np.int64), a, b


def protocol_share(path, meta, blocks, logs, a, b):
    """Protocol share of each swap's fee; NaN where the history does not cover the swap."""
    share = np.full(len(blocks), np.nan)
    if path is None:
        return share
    history = json.loads(Path(path).read_text())
    if history['pool'].lower() != meta['address'].lower():
        raise ValueError(f'{path} is for another pool')
    covered = (blocks >= history['from_block']) & (blocks <= history['to_block'])
    code = np.full(len(blocks), history['initial'], np.int64)
    for change in history['changes']:
        later = (blocks > change['block']) | ((blocks == change['block']) & (logs > change['event_index']))
        code[later] = change['fee_protocol']
    # Native token0 is the input when native sqrt price falls.
    token0_in = (b < a) if meta['base_is_token0'] else (b > a)
    divisor = np.where(token0_in, code % 16, code//16)
    share[covered] = np.where(divisor[covered] > 0, 1/np.maximum(divisor[covered], 1), 0.)
    return share


def swap_terms(a, b, fee, share, reference):
    """Per unit liquidity, quote units: (fee income, LVR loss) at one reference price."""
    rate = fee/(1-fee)*(1-np.nan_to_num(share))
    dx = 1/a - 1/b                        # base paid out by the LP (negative when received)
    dy = b - a                            # quote received by the LP (negative when paid)
    fees = np.where(dy > 0, dy, -dx*reference)*rate
    return fees, dx*reference - dy


def load_cex(path):
    """CSV with header: unix timestamp seconds, price in the pool's quote per base."""
    t = pq.read_table(path) if str(path).endswith('.parquet') else pacsv.read_csv(path)
    ts, px = t.column(0).to_numpy().astype(float), t.column(1).to_numpy().astype(float)
    order = np.argsort(ts)
    return ts[order], px[order]


def mainnet_swaps(data, name, protocol_file=None):
    """Swaps of one prepared mainnet pool: blocks, timestamps, pre/post sqrt (human), fee, protocol share."""
    data = Path(data); folder = data/'pools'/name
    meta = json.loads((folder/'metadata.json').read_text())
    header = json.loads((data/'headers/COMPLETE.json').read_text())
    clock = np.load(data/'headers/timestamp.npy', mmap_mode='r')
    blocks, logs, a, b = load_swaps(folder, meta)
    times = np.asarray(clock[blocks-header['first_block']], np.int64)
    share = np.nan_to_num(protocol_share(protocol_file, meta, blocks, logs, a, b))
    return blocks, times, a, b, meta['fee']/1e6, share


def l2_swaps(folder, name):
    """Swaps collected by scripts.collect_l2_swaps for one pool, same tuple as mainnet_swaps."""
    import pyarrow.dataset as ds
    folder = Path(folder)
    pool = json.loads((folder/'pools.json').read_text())[name]
    state = json.loads((folder/'progress.json').read_text())['start_state'][name]
    files = sorted(str(p) for p in folder.glob('part-*.parquet'))
    t = ds.dataset(files, format='parquet').to_table(columns=['block', 'log_index', 'timestamp', 'sqrt_after'], filter=ds.field('pool') == name)
    o = np.lexsort((t['log_index'].to_numpy(), t['block'].to_numpy()))
    meta = dict(base_is_token0=pool['base_is_token0'], decimals0=18 if pool['base_is_token0'] else 6, decimals1=6 if pool['base_is_token0'] else 18)
    b = human_sqrt(np.asarray(t['sqrt_after'].to_pylist(), dtype=object)[o], meta)
    a = np.r_[human_sqrt([state['sqrt']], meta), b[:-1]]
    share = (1/(state['fee_protocol'] % 16)) if state['fee_protocol'] % 16 else 0.
    return (t['block'].to_numpy()[o].astype(np.int64), t['timestamp'].to_numpy()[o].astype(np.int64), a, b,
            pool['fee']/1e6, np.full(len(b), share))
