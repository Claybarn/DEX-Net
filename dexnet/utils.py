"""Small file and bookkeeping helpers."""
import json
import torch


def dump(path, value):
    """Atomically write JSON."""
    t = path.with_suffix('.tmp'); t.write_text(json.dumps(value, indent=2)+'\n'); t.replace(path)


def atomic_save(value, path):
    """Atomically write a torch checkpoint."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def range_update(book, lower, upper, delta):
    """Add delta liquidity to the aggregate inventory of one tick range."""
    level = book.setdefault(lower, {})
    new = level.get(upper, 0)+delta
    if new < 0: raise ValueError('Burn exceeds aggregate range inventory; incomplete liquidity history')
    if new: level[upper] = new
    else: level.pop(upper, None)
    if not level: book.pop(lower, None)
