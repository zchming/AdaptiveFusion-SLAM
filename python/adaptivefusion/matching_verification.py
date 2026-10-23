"""Depth-independent geometric masks and equal-cardinality input controls."""
import numpy as np


def checked_mask(model, mask, count):
    if model is None or mask is None:
        raise ValueError('no geometric model/mask returned')
    model = np.asarray(model, float); mask = np.asarray(mask).reshape(-1)
    if (model.shape != (3, 3) or not np.isfinite(model).all() or np.linalg.norm(model) == 0
            or mask.shape != (count,) or not np.isin(mask, [0, 1]).all()):
        raise ValueError('invalid geometric model/mask')
    return model, mask.astype(bool)


def filtered_rows(matches, mask, rows):
    """Map the full 2D mask onto original depth-eligible rows without renumbering."""
    matches = np.asarray(matches)
    if matches.ndim != 2 or matches.shape[1] != 2 or len(mask) != len(matches):
        raise ValueError('match-mask identity mismatch')
    lookup = {tuple(map(int, pair)):bool(ok) for pair, ok in zip(matches, mask)}
    if len(lookup) != len(matches): raise ValueError('duplicate match identity')
    selected = []
    for row in rows:
        key = row['source_feature'], row['target_feature']
        if key not in lookup: raise ValueError('original eligible row absent from matches')
        if lookup[key]: selected.append(row)
    return selected


def equal_count_rows(rows, count, seed):
    if not 0 <= count <= len(rows): raise ValueError('invalid subset cardinality')
    indices = np.sort(np.random.default_rng(seed).choice(len(rows), count, replace=False))
    return [rows[int(i)] for i in indices]
