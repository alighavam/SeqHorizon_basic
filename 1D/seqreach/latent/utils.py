"""Small helpers shared by the latent-analysis methods.

Ported from ``latent_analysis/utils.py`` by Mehrdad Kashefi (MIT licence, see
``LICENSE`` in this folder).  The spike-train tools (``get_stem``, ``get_fr``,
``align_fr`` ...) and the NWB / .mat readers are not ported: a GRU state needs no
binning or smoothing, and our trials come from ``seqreach.latent.data`` instead.

Array convention, as in the original: the unit axis is always *last*, e.g.
``(conditions, time, units)``.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- indicators
def get_indicator_matrix(data) -> np.ndarray:
    """One-hot (dummy) coding of a label vector: ``(n_samples, n_levels)``.

    Levels are ordered as ``np.unique`` orders them.  Labels may be anything
    ``np.unique`` can sort (ints, strings, rows of a 2-D array).
    """
    data = np.asarray(data)
    if data.ndim > 1 and data.shape[1] == 1:
        data = data[:, 0]
    axis = 0 if data.ndim > 1 else None
    _, inv = np.unique(data, return_inverse=True, axis=axis)
    inv = np.asarray(inv).reshape(-1)
    out = np.zeros((len(inv), inv.max() + 1))
    out[np.arange(len(inv)), inv] = 1.0
    return out


def get_condition_mean(D: np.ndarray, indicator) -> np.ndarray:
    """Average the rows of ``D`` (first axis = trials) within each condition.

    Returns ``(n_conditions, ...)`` with conditions in ``np.unique`` order.
    Unlike the original, the labels need not be ``0 .. K-1``, and NaNs (steps
    outside a trial, see ``data.align``) are ignored rather than propagated.
    """
    levels, inv = np.unique(np.asarray(indicator).reshape(-1), return_inverse=True)
    D = np.asarray(D, dtype=float)
    out = np.empty((len(levels),) + D.shape[1:])
    for k in range(len(levels)):
        with np.errstate(invalid="ignore"):
            out[k] = np.nanmean(D[inv == k], axis=0)
    return out


def collapse_cond_time(data: np.ndarray) -> np.ndarray:
    """``(n_cond, n_time, n_units)`` -> ``(n_cond * n_time, n_units)``, condition-major."""
    data = np.asarray(data)
    return data.reshape(-1, data.shape[-1])


# --------------------------------------------------------------- second moments
def get_G(data: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Second-moment matrix of condition patterns, ``G = H Y Y' H' / N``.

    ``data`` is ``(K conditions, N units)``; ``H`` removes the mean pattern.
    """
    data = np.asarray(data, dtype=float)
    K, N = data.shape
    H = np.eye(K) - np.ones((K, K)) / K
    G = (H @ data @ data.T @ H.T) / N
    if normalize:
        G = G / np.trace(G)
    return G


def dist_to_G(dist: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix -> second-moment matrix (double centring)."""
    n = dist.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * H @ (np.asarray(dist) ** 2) @ H


def cosine_similarity(D1: np.ndarray, D2: np.ndarray) -> float:
    """Cosine similarity between two matrices, compared as flattened vectors.

    Change from the original: it shifted each vector by ``- min - (max - min)``
    (i.e. by ``- max``) before the cosine, which looks like a typo for a min-max
    rescaling and makes the result depend on each matrix's maximum.  This is the
    plain cosine.
    """
    r1, r2 = np.ravel(D1).astype(float), np.ravel(D2).astype(float)
    return float(r1 @ r2 / np.sqrt((r1 @ r1) * (r2 @ r2)))


# --------------------------------------------------------------- normalisation
def soft_normalize(X: np.ndarray, soft_norm: float | None = None, center: str = "condition",
                   unit_range: np.ndarray | None = None):
    """Centre, and optionally soft-normalise, ``(conditions, time, units)`` data.

    ``center``: ``"condition"`` subtracts the across-condition mean at each time step
    (removes the condition-independent signal; what the original does for jPCA and
    the orthogonal-subspace PCA), ``"all"`` subtracts each unit's overall mean,
    ``"none"`` leaves the data alone.  For 2-D ``(samples, units)`` input both centring
    modes subtract the overall mean.

    ``soft_norm``: ``None`` (default) means no per-unit scaling; a number divides each
    unit by ``range + soft_norm``.  The original always used ``soft_norm = 5``, which is
    in spikes/s and assumes firing rates of tens of Hz.  GRU units live in (-1, 1), so 5
    would shrink every unit by roughly the same factor; and any per-unit scaling bends
    the geometry of the linear readout ``u = w.h + b``, so subspaces found on scaled data
    can no longer be compared with ``w``.  Hence the default.  If you want the original
    flavour of soft normalisation, something like 0.1-0.2 is the comparable constant.

    Returns ``(X_normalised, unit_range)``; pass ``unit_range`` back in to normalise new
    data with the ranges of the data a model was fitted on.
    """
    X = np.asarray(X, dtype=float)
    if center == "condition" and X.ndim >= 3:
        X = X - np.nanmean(X, axis=0, keepdims=True)
    elif center in ("condition", "all"):
        X = X - np.nanmean(X.reshape(-1, X.shape[-1]), axis=0)
    elif center != "none":
        raise ValueError(f"center must be 'condition', 'all' or 'none', not {center!r}")
    flat = X.reshape(-1, X.shape[-1])
    if unit_range is None:
        unit_range = np.nanmax(flat, axis=0) - np.nanmin(flat, axis=0)
    if soft_norm is not None:
        X = X / (unit_range + soft_norm)
    return X, unit_range


def subspace_variance(X: np.ndarray, W: np.ndarray) -> float:
    """Fraction of the variance of ``X`` (samples x units, or any shape with units last)
    that lies in the span of the columns of ``W``.  ``W`` need not be orthonormal."""
    Xf = np.asarray(X, dtype=float).reshape(-1, np.shape(X)[-1])
    Xf = Xf[np.isfinite(Xf).all(1)]
    Xf = Xf - Xf.mean(0)
    Q, _ = np.linalg.qr(np.asarray(W, dtype=float).reshape(Xf.shape[1], -1))
    return float(np.sum((Xf @ Q) ** 2) / np.sum(Xf ** 2))


# --------------------------------------------------------------- feature bases
class GaussianBasis1D:
    """Gaussian tuning-curve features for a 1-D variable (e.g. a target position).

    The 1-D counterpart of the original ``Kernel_Gaussian`` (which tiles a 2-D
    workspace), with the same rule for placing centres: ``n_kernel`` centres evenly
    spaced inside ``[lo, hi]``, inset by one spacing from each end.  Used to build
    feature models for ``stats.Model`` / ``stats.GDecompose``.  The original's
    ``Kernel_Cosine`` (for reach angles) has no use in a 1-D task and is not ported.
    """

    def __init__(self, lo: float, hi: float, n_kernel: int, width: float):
        step = (hi - lo) / (n_kernel + 1)
        self.centers = np.linspace(lo + step, hi - step, n_kernel)
        self.width = float(width)

    def encode(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return np.exp(-0.5 * ((x - self.centers[None, :]) / self.width) ** 2)
