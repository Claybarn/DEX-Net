"""Causal depth and retained-observation inputs for rollout-only conditioning.

No future historical inventory is read: profiles use each path's current book.
The original observations are immutable; generated history is capped at 64.
"""
import math
import numpy as np
import torch
from .tick_mechanics import MIN_TICK, MAX_TICK, Q96

PRICE_OFFSETS = np.array([-.04, -.02, -.01, -.005, -.002, -.001, -.0005,
                          -.0001, 0., .0001, .0005, .001, .002, .005, .01, .02, .04])


def liquidity_profile(state, book, meta):
    """log1p(L_at_offset/L_now)/4 at economic log-price offsets, -4% to +4%."""
    if state.liquidity <= 0:
        raise ValueError('Depth normalization requires positive active liquidity')
    sign = 1 if meta['base_is_token0'] else -1
    tick = 2 * (math.log(state.sqrt) - math.log(Q96)) / math.log1p(.0001)
    targets = np.clip(np.floor(tick + sign * PRICE_OFFSETS / math.log1p(.0001)),
                      MIN_TICK, MAX_TICK - 1).astype(int)
    # At an exact boundary the stored tick can intentionally be one below sqrt's
    # tick after a leftward crossing. Center is the current active state itself.
    targets[len(PRICE_OFFSETS)//2] = state.tick
    values = np.zeros(len(targets))
    for left in [True, False]:
        ids = [i for i, t in enumerate(targets) if (t < state.tick) == left]
        ids.sort(key=lambda i: targets[i], reverse=left)
        cursor, liquidity = state.tick, state.liquidity
        for i in ids:
            target = int(targets[i])
            while cursor > target if left else cursor < target:
                boundary, initialized = book.next(cursor, left)
                if (boundary <= target if left else boundary > target):
                    break
                if initialized:
                    liquidity += (-1 if left else 1) * book.tick(boundary)[1]
                if liquidity < 0:
                    raise ValueError('Negative depth in reconstructed book')
                cursor = boundary - 1 if left else boundary
            values[i] = math.log1p(liquidity / state.liquidity) / 4.
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite liquidity profile')
    return values.astype(np.float32)


def encode_rollout(model, batch, states, books, meta, observed, steps):
    """Batched bridge used identically by LP sampling, score replay and evaluation."""
    depth = None
    if model.liquidity_context:
        depth = batch['x'].new_tensor(np.stack([
            liquidity_profile(s, book, meta) for s, book in zip(states, books)]))
    memory = observed if model.retain_observed else None
    return model.encode(batch['x'], batch['history_kind'], batch['history_mask'],
                        batch['metadata'], batch.get('pool_id'), depth=depth,
                        observed=memory, rollout_steps=steps)
