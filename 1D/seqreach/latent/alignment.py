"""Comparing two populations (or one population in two epochs).

Ported from ``latent_analysis/alignment.py`` by Mehrdad Kashefi (MIT licence, see
``LICENSE`` in this folder).  In our setting "two populations" is typically two seeds
(do different agents find the same latent dynamics?) or two epochs of one agent (hold
vs movement, first vs second reach).

``top_pcs``, ``alignment_index`` and ``alignment_index_crossval`` are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from .utils import collapse_cond_time, get_condition_mean


def _kfold(n, n_splits, rng):
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_splits)
    for i in range(n_splits):
        yield np.concatenate(folds[:i] + folds[i + 1:]), folds[i]


# ===================================================================== CCA
class CCA_svd:
    """Canonical correlation analysis via QR + SVD.

    Change from the original: the data are centred in ``fit`` (and new data with the
    fitted means in ``transform``).  The original took the QR of uncentred data, so the
    mean pattern counted as a shared dimension and inflated the correlations -- large
    for GRU units, whose means are far from zero.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X, Y):
        self.mx_, self.my_ = X.mean(0), Y.mean(0)
        Q_x, R_x = np.linalg.qr(X - self.mx_)
        Q_y, R_y = np.linalg.qr(Y - self.my_)
        U, S, Vt = np.linalg.svd(Q_x.T @ Q_y)
        if len(S) < self.n_components:
            raise ValueError("n_components exceeds the smaller number of features")
        self._cc, self.cc = S, S[:self.n_components]
        self.Wx = np.linalg.pinv(R_x) @ U
        self.Wy = np.linalg.pinv(R_y) @ Vt.T
        return self

    def transform(self, X, Y):
        k = self.n_components
        return (X - self.mx_) @ self.Wx[:, :k], (Y - self.my_) @ self.Wy[:, :k]

    def score(self, X, Y):
        """R^2 between the canonical variates of X and Y (the original's score)."""
        X_c, Y_c = self.transform(X, Y)
        return 1 - np.sum((X_c - Y_c) ** 2) / np.sum((X_c - X_c.mean(0)) ** 2)


# ===================================================================== Procrustes
class Procrustes:
    """Procrustes analysis (translation, orthogonal rotation/reflection, scaling) of Y
    onto X, after Matlab's ``procrustes``.  Unchanged from the original."""

    def __init__(self, scaling: bool = True, reflection="best"):
        self.scaling, self.reflection = scaling, reflection

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        n, nx = X.shape
        ny = Y.shape[1]
        muX, muY = X.mean(0), Y.mean(0)
        X0, Y0 = X - muX, Y - muY
        ssX, ssY = (X0 ** 2).sum(), (Y0 ** 2).sum()
        normX, normY = np.sqrt(ssX), np.sqrt(ssY)
        X0, Y0 = X0 / normX, Y0 / normY
        if ny < nx:
            Y0 = np.concatenate((Y0, np.zeros((n, nx - ny))), axis=1)
        U, s, Vt = np.linalg.svd(X0.T @ Y0, full_matrices=False)
        V = Vt.T
        T = V @ U.T
        if self.reflection != "best":
            have_reflection = np.linalg.det(T) < 0
            if self.reflection != have_reflection:
                V[:, -1] *= -1
                s[-1] *= -1
                T = V @ U.T
        traceTA = s.sum()
        if self.scaling:
            b = traceTA * normX / normY
            d = 1 - traceTA ** 2
            Z = normX * traceTA * Y0 @ T + muX
        else:
            b = 1
            d = 1 + ssY / ssX - 2 * traceTA * normY / normX
            Z = normY * Y0 @ T + muX
        if ny < nx:
            T = T[:ny, :]
        c = muX - b * muY @ T
        self.tform = {"rotation": T, "scale": b, "translation": c}
        return d, Z, self.tform

    def transform(self, Y):
        return Y @ self.tform["rotation"] * self.tform["scale"] + self.tform["translation"]

    def score(self, X, Y):
        Z = self.transform(Y)
        return 1 - np.sum((X - Z) ** 2) / ((X - X.mean(0)) ** 2).sum()


# ===================================================================== cross-validated
def get_dissimilarity_cond_avr(X, X_conds, Y, Y_conds, n_folds: int = 2, n_times: int = 1,
                               method: str = "CCA_svd", n_components: int = 10, seed: int = 0):
    """Cross-validated similarity of two datasets' condition-averaged dynamics.

    ``X``, ``Y``: single trials ``(trials, time, units)``; ``*_conds``: a condition label
    per trial (both datasets must share the condition set).  Trials are split into
    folds; the alignment (CCA or Procrustes) is fitted on the condition averages of the
    training folds and scored on those of the held-out fold.

    Changes from the original: ``CCA_sklearn`` is not available; the number of
    canonical components (fixed at 10 in the original) is ``n_components``; and every
    fold must contain every condition (the original silently averaged empty
    conditions to NaN).  As with ``dPCA(regularizer="auto")``, the folds only differ if
    repeats differ, so use rollouts with motor noise.

    Returns ``(df_score, df_corr)`` as the original.
    """
    rng = np.random.default_rng(seed)
    data, conds = {"X": np.asarray(X, float), "Y": np.asarray(Y, float)}, \
        {"X": np.asarray(X_conds), "Y": np.asarray(Y_conds)}
    levels = np.unique(conds["X"])
    if not np.array_equal(levels, np.unique(conds["Y"])):
        raise ValueError("X and Y must contain the same conditions")
    rows, rows_score = [], []

    def cond_avg(name, idx):
        if not np.array_equal(np.unique(conds[name][idx]), levels):
            raise ValueError("a fold is missing a condition; use fewer folds")
        return collapse_cond_time(get_condition_mean(data[name][idx], conds[name][idx]))

    for t in range(n_times):
        folds = {name: list(_kfold(len(data[name]), n_folds, rng)) for name in data}
        for fold in range(n_folds):
            (trX, teX), (trY, teY) = folds["X"][fold], folds["Y"][fold]
            dXtr, dYtr = cond_avg("X", trX), cond_avg("Y", trY)
            dXte, dYte = cond_avg("X", teX), cond_avg("Y", teY)
            if method == "CCA_svd":
                cca = CCA_svd(n_components=n_components).fit(dXtr, dYtr)
                X_c, Y_c = cca.transform(dXte, dYte)
                rows_score.append({"Fold": fold, "Time": t, "Score": cca.score(dXte, dYte)})
                for cc in range(X_c.shape[1]):
                    rows.append({"Fold": fold, "Time": t, "CC": cc,
                                 "Corr": np.corrcoef(X_c[:, cc], Y_c[:, cc])[0, 1]})
            elif method == "Procrustes":
                pr = Procrustes(scaling=True, reflection="best")
                pr.fit(dXtr, dYtr)
                rows_score.append({"Fold": fold, "Time": t, "Score": pr.score(dXte, dYte)})
            else:
                raise ValueError("method must be 'CCA_svd' or 'Procrustes'")
    return pd.DataFrame(rows_score), pd.DataFrame(rows)


# ===================================================================== alignment index
def _flat(X):
    """(trials, time, units) -> (trials*time, units)."""
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, X.shape[-1])


def top_pcs(X, k):
    """Orthonormal basis (n_units, k) for the top-k PC subspace of X (samples x units)."""
    X0 = X - X.mean(axis=0)
    n, u = X0.shape
    if u <= n:
        _, V = eigh(X0.T @ X0, subset_by_index=(u - k, u - 1))
    else:
        _, W = eigh(X0 @ X0.T, subset_by_index=(n - k, n - 1))
        V = X0.T @ W
        V /= np.linalg.norm(V, axis=0, keepdims=True)
    return np.ascontiguousarray(V)


def alignment_index(A, B, n_dim: int = 10, pca=top_pcs):
    """Variance of A captured by B's top ``n_dim`` PCs, relative to A's own top PCs
    (Elsayed et al. 2016).  1 = same subspace, ~0 = orthogonal.  A and B are
    ``(..., units)``; they are centred on the mean of A and B together."""
    Amat, Bmat = _flat(A), _flat(B)
    mu = (Amat.sum(0) + Bmat.sum(0)) / (len(Amat) + len(Bmat))
    UA, UB = pca(Amat, n_dim), pca(Bmat, n_dim)
    Ac = Amat - mu
    PA, PB = Ac @ UA, Ac @ UB
    return float((PB.ravel() @ PB.ravel()) / (PA.ravel() @ PA.ravel()))


class _Half:
    __slots__ = ("n", "mean", "S", "U")

    def __init__(self, X, n_dim, pca):
        M = _flat(X)
        self.n = M.shape[0]
        self.mean = M.mean(axis=0)
        X0 = M - self.mean
        self.S = X0.T @ X0
        self.U = pca(M, n_dim)


def _quad(h, d, U):
    v = d @ U
    return np.sum(U * (h.S @ U)) + h.n * (v @ v)


def _ai_pair(X, Y):
    d = (Y.n / (X.n + Y.n)) * (Y.mean - X.mean)
    return _quad(X, d, Y.U) / _quad(X, d, X.U)


def _split(X, parts, rng=None):
    rng = np.random if rng is None else rng
    idx = rng.permutation(X.shape[0])
    k = X.shape[0] // parts
    return [X[idx[i * k:(i + 1) * k]] for i in range(parts)]


def alignment_index_crossval(A, B, n_dim: int = 10, pca=top_pcs, same=None,
                             match_diag: bool = False, seed=None):
    """Cross-validated alignment index: subspaces and variances come from disjoint
    halves of the trials (first axis), normalised by the within-condition split-half
    value, averaged over both directions (A in B, B in A).

    ``same``: True when A and B are the same condition (then four disjoint quarters).
    ``match_diag``: subsample off-diagonal halves to quarter size too.
    ``seed`` (added) makes the random split reproducible.  Needs trial-to-trial
    variability to be meaningful: use rollouts with motor noise.
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    if same is None:
        same = A is B
    if same:
        a1, a2, b1, b2 = _split(A, 4, rng)
    elif match_diag:
        a1, a2 = _split(A, 4, rng)[:2]
        b1, b2 = _split(B, 4, rng)[:2]
    else:
        a1, a2 = _split(A, 2, rng)
        b1, b2 = _split(B, 2, rng)
    A1, A2 = _Half(a1, n_dim, pca), _Half(a2, n_dim, pca)
    B1, B2 = _Half(b1, n_dim, pca), _Half(b2, n_dim, pca)
    a = (_ai_pair(A1, B1) + _ai_pair(A2, B2) + _ai_pair(A1, B2) + _ai_pair(A2, B1)) \
        / (4 * (_ai_pair(A1, A2) + _ai_pair(A2, A1)))
    b = (_ai_pair(B1, A1) + _ai_pair(B2, A2) + _ai_pair(B1, A2) + _ai_pair(B2, A1)) \
        / (4 * (_ai_pair(B1, B2) + _ai_pair(B2, B1)))
    return a + b
