"""Linear dimensionality reduction for population activity (here: GRU states).

Ported from ``latent_analysis/transform.py`` by Mehrdad Kashefi (MIT licence, see
``LICENSE`` in this folder).  Same classes and the same objectives; the changes are
listed in each docstring.  Three apply throughout:

* **No per-unit scaling by default.**  The original divides each unit by
  ``range + 5`` (spikes/s).  Here ``soft_norm=None`` means centring only; see
  ``utils.soft_normalize`` for why, and pass a number to get soft normalisation back.
* **No extra dependencies.**  ``sklearn.FactorAnalysis``, the ``dPCA`` package and
  ``pymanopt`` are replaced by short numpy implementations of the same estimators.
* Arrays keep the original convention: units on the *last* axis, typically
  ``(conditions, time, units)``.
"""

from __future__ import annotations

import itertools

import numpy as np

from .utils import soft_normalize


def _flat(X):
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, X.shape[-1])


# ===================================================================== PCA
class PCA:
    """Principal component analysis.

    Change from the original: ``variance`` is divided by ``n_samples - 1`` (the
    original divided by ``n_units - 1``).  ``variance_explained`` is unchanged.
    """

    def __init__(self, num_latent: int):
        self.num_latent = num_latent

    def fit(self, X):
        Xf = _flat(X)
        self.method = "PCA"
        self.mean_ = Xf.mean(0)
        _, S, Vt = np.linalg.svd(Xf - self.mean_, full_matrices=False)
        self.components_ = Vt.T[:, :self.num_latent]          # (units, k)
        self.variance_explained = S ** 2 / np.sum(S ** 2)
        self.variance = S ** 2 / (Xf.shape[0] - 1)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Z = (_flat(X) - self.mean_) @ self.components_
        return Z.reshape(X.shape[:-1] + (self.components_.shape[1],))


# ===================================================================== FA
class FA:
    """Factor analysis, fitted by EM (Rubin & Thayer 1982).

    Replaces ``sklearn.decomposition.FactorAnalysis`` (same model, same maximum-
    likelihood estimate up to a rotation of the factors).  ``components_`` is
    ``(k, units)`` as in sklearn.

    Bug fixed: with ``ensure_orthogonality=True`` the original projected onto
    ``U[:k, :]`` -- the first k *rows* of the left singular vectors -- instead of the
    first k columns, so it did not project onto the factor subspace.  Here the
    projection is onto an orthonormal basis of the loadings, ``U[:, :k]``.
    """

    def __init__(self, num_latent: int, max_iter: int = 2000, tol: float = 1e-8):
        self.num_latent, self.max_iter, self.tol = num_latent, max_iter, tol

    def fit(self, X):
        Xf = _flat(X)
        n, p = Xf.shape
        k = self.num_latent
        self.method = "FA"
        self.mean_ = Xf.mean(0)
        Xc = Xf - self.mean_
        S = Xc.T @ Xc / n
        floor = 1e-6 * max(np.trace(S) / p, 1e-12)
        ev, V = np.linalg.eigh(S)
        ev, V = ev[::-1][:k], V[:, ::-1][:, :k]
        W = V * np.sqrt(np.maximum(ev, 0))
        psi = np.maximum(np.diag(S) - np.sum(W ** 2, 1), floor)
        self.loglik_ = []
        for _ in range(self.max_iter):
            Wp = W / psi[:, None]
            M = np.eye(k) + W.T @ Wp
            beta = np.linalg.solve(M, Wp.T)                        # (k, p)
            Ezz = np.eye(k) - beta @ W + beta @ S @ beta.T
            W = S @ beta.T @ np.linalg.inv(Ezz)
            psi = np.maximum(np.diag(S) - np.sum(W * (S @ beta.T), 1), floor)
            C = W @ W.T + np.diag(psi)
            _, logdet = np.linalg.slogdet(C)
            ll = -0.5 * n * (p * np.log(2 * np.pi) + logdet + np.trace(np.linalg.solve(C, S)))
            self.loglik_.append(ll)
            if len(self.loglik_) > 1 and abs(ll - self.loglik_[-2]) < self.tol * abs(ll):
                break
        self.loadings_ = W                                         # (p, k)
        self.noise_variance_ = psi
        self.components_ = W.T
        return self

    def transform(self, X, ensure_orthogonality: bool = False):
        X = np.asarray(X, dtype=float)
        Xc = _flat(X) - self.mean_
        if ensure_orthogonality:
            U, _, _ = np.linalg.svd(self.loadings_, full_matrices=False)
            Z = Xc @ U[:, :self.num_latent]
        else:  # posterior mean of the factors, as sklearn's transform
            W, psi = self.loadings_, self.noise_variance_
            Wp = W / psi[:, None]
            M = np.eye(self.num_latent) + W.T @ Wp
            Z = Xc @ Wp @ np.linalg.inv(M)
        return Z.reshape(X.shape[:-1] + (self.num_latent,))


# ===================================================================== jPCA
class jPCA:
    """jPCA (Churchland et al. 2012): the plane of strongest rotational dynamics.

    ``X`` is ``(conditions, time, units)``.  Steps, as in the original: centre on the
    across-condition mean at each time step (optionally soft-normalise), reduce to
    ``num_latent`` PCs, fit ``dX = X M`` with ``M`` skew-symmetric, and take the plane
    of ``M``'s fastest rotation.

    Changes from the original:

    * The skew-symmetric fit is solved exactly as a linear least-squares problem over
      the ``k(k-1)/2`` free entries (the original used conjugate gradients from a
      least-squares start; same optimum).
    * Pairs are ``(x_t, x_{t+1} - x_t)``, as in Churchland's code (the original paired
      the difference with ``x_{t+1}``).
    * **Bug fixed:** the original took the first eigenvalue pair that ``np.linalg.eig``
      returned, which is not sorted, so the plane was not reliably the fastest one.
      Here planes are sorted by rotation frequency; ``planes_`` holds all of them.
    * **Bug fixed:** the original's counter-clockwise check flipped the second axis
      whenever the cross product was non-zero, i.e. almost always.  Here the basis is
      built so the fitted rotation is counter-clockwise, and the optional alignment of
      the initial states with the x axis is a proper rotation that keeps it so.
    * ``transform`` reuses the PCA and unit ranges of ``fit`` (the original refitted the
      PCA on whatever data it was given).
    * ``soft_norm`` defaults to ``None`` (the original hard-coded ``range + 5``).
    """

    def __init__(self, num_latent: int = 6, force_skewness: bool = True,
                 align_x_axis: bool = True, soft_norm: float | None = None):
        self.num_comp_pc = num_latent
        self.force_skewness = force_skewness
        self.align_x_axis = align_x_axis
        self.soft_norm = soft_norm

    @staticmethod
    def _skew_basis(k):
        idx = list(itertools.combinations(range(k), 2))
        B = np.zeros((len(idx), k, k))
        for n, (i, j) in enumerate(idx):
            B[n, i, j], B[n, j, i] = 1.0, -1.0
        return B

    def skew_sym_regress(self, X, X_dot):
        """Least-squares ``M`` (skew-symmetric) for ``X_dot ~ X @ M``."""
        B = self._skew_basis(X.shape[1])
        A = np.stack([(X @ b).ravel() for b in B], axis=1)
        h = np.linalg.lstsq(A, X_dot.ravel(), rcond=None)[0]
        return np.tensordot(h, B, axes=1)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        Xn, self.unit_range_ = soft_normalize(X, self.soft_norm, center="condition")
        self.pca_ = PCA(self.num_comp_pc).fit(Xn)
        self.variance_explained_pca = float(np.sum(self.pca_.variance_explained[:self.num_comp_pc]))
        Z = self.pca_.transform(Xn)                                # (cond, time, k)
        Xs = Z[:, :-1, :].reshape(-1, Z.shape[-1])
        dXs = np.diff(Z, axis=1).reshape(-1, Z.shape[-1])
        if self.force_skewness:
            M = self.skew_sym_regress(Xs, dXs)
        else:
            M = np.linalg.lstsq(Xs, dXs, rcond=None)[0]
        self.M = M
        self.variance_explained_rot_fit = float(self.R2(dXs, Xs @ M))
        # Column-vector dynamics are dx/dt = A x with A = M'.  For A's eigenvalue
        # i*w (w > 0) with eigenvector a + i b, the basis (a, -b) turns counter-clockwise.
        L, V = np.linalg.eig(M.T)
        order = [i for i in np.argsort(-L.imag) if L[i].imag > 1e-12]
        self.planes_, self.frequencies_ = [], []
        for i in order:
            a, b = V[:, i].real, V[:, i].imag
            P = np.stack([a / np.linalg.norm(a), -b / np.linalg.norm(b)], axis=1)
            self.planes_.append(P)
            self.frequencies_.append(float(L[i].imag))
        if not self.planes_:
            raise ValueError("the fitted dynamics have no rotational component")
        self.jpca_w = self.planes_[0]
        if self.align_x_axis:
            Y0 = Z[:, 0, :] @ self.jpca_w
            _, _, Vt = np.linalg.svd(Y0 - Y0.mean(0), full_matrices=False)
            th = np.arctan2(Vt[0, 1], Vt[0, 0])
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            self.jpca_w = self.jpca_w @ R                          # rotation: still CCW
        self.components_units_ = self.pca_.components_ @ self.jpca_w   # (units, 2)
        Zj = Z @ self.jpca_w
        self.variance_explained_jpcs = float(np.sum(np.var(Zj.reshape(-1, 2), 0))
                                             / np.sum(np.var(Z.reshape(-1, Z.shape[-1]), 0)))
        self._Xn = Xn
        return self

    def transform(self, X=None):
        """Project onto the (first) jPC plane: ``(conditions, time, 2)``."""
        if X is None:
            Xn = self._Xn
        else:
            Xn, _ = soft_normalize(X, self.soft_norm, center="condition",
                                   unit_range=self.unit_range_)
        return self.pca_.transform(Xn) @ self.jpca_w

    @staticmethod
    def R2(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true, 0)) ** 2)


# ===================================================================== dPCA
def _subsets(labels):
    out = []
    for r in range(1, len(labels) + 1):
        out += ["".join(c) for c in itertools.combinations(labels, r)]
    return out


def marginalize(X, labels):
    """dPCA marginalisations of centred ``X`` (``labels`` names its non-unit axes).

    Returns ``{subset: X_subset}``, each broadcast to ``X``'s shape; they sum to ``X``
    and are mutually orthogonal (Kobak et al. 2016, eq. 5).
    """
    X = np.asarray(X, dtype=float)
    out = {}
    for s in _subsets(labels):
        axes = tuple(i for i, c in enumerate(labels) if c not in s)
        m = X.mean(axis=axes, keepdims=True) if axes else X.copy()
        for s2, v in out.items():
            if set(s2) < set(s):
                m = m - v
        out[s] = np.broadcast_to(m, X.shape).copy()
    return out


class dPCA:
    """Demixed PCA (Kobak et al. 2016), for any number of task factors.

    ``X`` holds condition averages, units last, e.g. ``(T1 direction, T2 direction,
    time, units)`` with ``labels="abt"``.  The original wrapped the ``dPCA`` package
    for one condition factor (``labels="ct"``); this is a numpy implementation of the
    same estimator (reduced-rank ridge regression per marginalisation), not a
    bit-for-bit copy of the package.

    ``join`` merges marginalisations, e.g. ``{"c": ["c", "ct"], "t": ["t"]}``.
    ``regularizer``: ``None`` (none), a number lambda (penalty ``lambda * ||X||^2``, as
    in the paper), or ``"auto"`` to pick lambda by cross-validation -- which needs
    ``trials`` in ``fit`` (this is what the original ``CVdPCA`` class did).  With the
    noiseless plant every repeat of a condition is identical, so ``"auto"`` needs
    rollouts with motor noise.

    After ``fit``: ``encoders_[key]`` (units x q), ``decoders_[key]`` (q x units),
    ``explained_variance_ratio_[key]`` (per component, the paper's definition) and
    ``marginal_purity_[key]`` (for each component, the share of its projected variance
    that comes from each marginalisation).
    """

    def __init__(self, labels: str = "ct", n_components=3, regularizer=None, join=None,
                 soft_norm: float | None = None, n_cv: int = 10, seed: int = 0):
        self.labels, self.n_components = labels, n_components
        self.regularizer, self.join = regularizer, join
        self.soft_norm, self.n_cv, self.seed = soft_norm, n_cv, seed

    # ------------------------------------------------------------ pieces
    def _prep(self, X, unit_range=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != len(self.labels) + 1:
            raise ValueError(f"X has {X.ndim - 1} non-unit axes but labels={self.labels!r}")
        mean = X.reshape(-1, X.shape[-1]).mean(0)
        Xc = X - mean
        if unit_range is None:
            f = Xc.reshape(-1, X.shape[-1])
            unit_range = f.max(0) - f.min(0)
        if self.soft_norm is not None:
            Xc = Xc / (unit_range + self.soft_norm)
        return Xc, mean, unit_range

    def _margs(self, Xc):
        m = marginalize(Xc, self.labels)
        if self.join is None:
            return m
        out = {}
        for key, parts in self.join.items():
            out[key] = sum(m[p] for p in parts)
        missing = set(m) - {p for parts in self.join.values() for p in parts}
        if missing:
            raise ValueError(f"join leaves marginalisations out: {sorted(missing)}")
        return out

    def _q(self, key):
        return self.n_components[key] if isinstance(self.n_components, dict) else self.n_components

    def _solve(self, Xc, margs, lam):
        U_ = Xc.shape[-1]
        Xr = Xc.reshape(-1, U_).T                                  # (units, samples)
        mu = (lam or 0.0) * np.sum(Xr ** 2)
        XXt = Xr @ Xr.T + mu * np.eye(U_)
        XXt_inv = np.linalg.pinv(XXt)
        enc, dec = {}, {}
        for key, Xm in margs.items():
            Xmr = Xm.reshape(-1, U_).T
            B = Xmr @ Xr.T @ XXt_inv                               # ridge OLS
            fitted = B @ Xr
            if mu > 0:                                             # augmented data
                fitted = np.concatenate([fitted, np.sqrt(mu) * B], axis=1)
            Uq = np.linalg.svd(fitted, full_matrices=False)[0][:, :self._q(key)]
            enc[key], dec[key] = Uq, Uq.T @ B
        return enc, dec

    def _cv_lambda(self, trials):
        rng = np.random.default_rng(self.seed)
        grid = np.logspace(-7, 0, 22)
        R = trials.shape[0]
        if R < 2:
            raise ValueError("regularizer='auto' needs at least 2 trials per condition")
        err = np.zeros(len(grid))
        shape = trials.shape[1:-1]                                 # the labelled axes
        tpos = self.labels.find("t")
        pick_shape = tuple(1 if i == tpos else s for i, s in enumerate(shape))
        rshape = (R,) + (1,) * len(shape)
        for _ in range(self.n_cv):
            # leave one random trial per condition out (the same trial at every time step)
            pick = rng.integers(0, R, size=pick_shape)
            mask = np.arange(R).reshape(rshape) == pick[None]
            mask = np.broadcast_to(mask[..., None], trials.shape)
            test = np.where(mask, trials, 0).sum(0)
            train = np.where(mask, 0, trials).sum(0) / (R - 1)
            Xtr, _, rng_tr = self._prep(train)
            Xte, _, _ = self._prep(test, unit_range=rng_tr)
            mtr, mte = self._margs(Xtr), self._margs(Xte)
            Xte_r = Xte.reshape(-1, Xte.shape[-1]).T
            for g, lam in enumerate(grid):
                enc, dec = self._solve(Xtr, mtr, lam)
                num = den = 0.0
                for key in mte:
                    T = mte[key].reshape(-1, Xte.shape[-1]).T
                    num += np.sum((T - enc[key] @ dec[key] @ Xte_r) ** 2)
                    den += np.sum(T ** 2)
                err[g] += num / den
        self.cv_grid_, self.cv_error_ = grid, err / self.n_cv
        return float(grid[np.argmin(err)])

    # ------------------------------------------------------------ API
    def fit(self, X, trials=None):
        """``X``: condition averages.  ``trials``: ``(repeats, *X.shape)``, needed only
        for ``regularizer="auto"`` (balanced: the same number of repeats per condition;
        see ``data.balanced_trials``)."""
        if self.regularizer == "auto":
            if trials is None:
                raise ValueError("regularizer='auto' needs trials")
            self.lambda_ = self._cv_lambda(np.asarray(trials, dtype=float))
        else:
            self.lambda_ = self.regularizer or 0.0
        Xc, self.mean_, self.unit_range_ = self._prep(X)
        margs = self._margs(Xc)
        self.encoders_, self.decoders_ = self._solve(Xc, margs, self.lambda_)
        Xr = Xc.reshape(-1, Xc.shape[-1]).T
        total = np.sum(Xr ** 2)
        self.total_variance_ = total
        self.marginal_variance_ratio_ = {k: float(np.sum(v ** 2) / total) for k, v in margs.items()}
        self.explained_variance_ratio_, self.marginal_purity_ = {}, {}
        for key in margs:
            F, D = self.encoders_[key], self.decoders_[key]
            ev, pur = [], []
            for i in range(F.shape[1]):
                resid = Xr - np.outer(F[:, i], D[i] @ Xr)
                ev.append(float((total - np.sum(resid ** 2)) / total))
                parts = {k2: float(np.sum((D[i] @ m.reshape(-1, Xc.shape[-1]).T) ** 2))
                         for k2, m in margs.items()}
                s = sum(parts.values())
                pur.append({k2: v / s for k2, v in parts.items()})
            self.explained_variance_ratio_[key], self.marginal_purity_[key] = ev, pur
        self._Xc = Xc
        return self

    def transform(self, X=None):
        """``{key: (*condition axes, q)}`` -- each marginalisation's components."""
        if X is None:
            Xc = self._Xc
        else:
            Xc = np.asarray(X, dtype=float) - self.mean_
            if self.soft_norm is not None:
                Xc = Xc / (self.unit_range_ + self.soft_norm)
        return {k: Xc @ D.T for k, D in self.decoders_.items()}


class CVdPCA(dPCA):
    """The original's cross-validated dPCA: ``dPCA`` with ``regularizer="auto"``.

    Original signature: ``fit(data_aligned, conds_idx)`` with single trials
    ``(trials, time, units)`` and one condition label per trial; trials are resampled
    (with repeats) to equal counts per condition, as the original did.
    """

    def __init__(self, n_components, seed: int = 0, **kwargs):
        super().__init__(labels="ct", n_components=n_components, regularizer="auto",
                         seed=seed, **kwargs)

    def _arrange(self, data_aligned, conds_idx):
        from .data import balanced_trials
        trials, _ = balanced_trials(data_aligned, conds_idx, seed=self.seed)
        return trials.mean(0), trials

    def fit(self, data_aligned, conds_idx):
        X, trials = self._arrange(data_aligned, conds_idx)
        return super().fit(X, trials)

    def transform(self, data_aligned=None, conds_idx=None):
        if data_aligned is None:
            return super().transform()
        X, _ = self._arrange(data_aligned, conds_idx)
        return super().transform(X)


# ===================================================================== orthogonal subspaces
def _stiefel_max(Cs, ks, W0, max_iter=20000, tol=1e-13):
    """Maximise ``mean_i tr(W_i' C_i W_i)`` over ``[W_1 .. W_m]`` with orthonormal columns.

    Each ``C_i`` is a (normalised) covariance, so the objective is convex in W and the
    generalised power iteration ``W <- polar(grad)`` increases it monotonically to a
    stationary point (Nie et al. 2017; Journee et al. 2010).  Replaces pymanopt's
    trust-region solver: same problem, same constraint set.
    """
    edges = np.cumsum([0] + list(ks))
    W = W0.copy()
    f_old = -np.inf
    for _ in range(max_iter):
        G = np.concatenate([Cs[i] @ W[:, edges[i]:edges[i + 1]] for i in range(len(ks))], axis=1)
        f = np.sum(W * G) / len(ks)
        U, _, Vt = np.linalg.svd(G, full_matrices=False)
        W = U @ Vt
        if f - f_old < tol * max(abs(f), 1.0):
            break
        f_old = f
    G = np.concatenate([Cs[i] @ W[:, edges[i]:edges[i + 1]] for i in range(len(ks))], axis=1)
    f = np.sum(W * G) / len(ks)
    # Riemannian gradient on the Stiefel manifold, for the stationarity check
    rg = G - W @ (0.5 * (W.T @ G + G.T @ W))
    return W, float(f), float(np.linalg.norm(rg))


class OrthogonalSubspaces:
    """Mutually orthogonal subspaces, each capturing as much of one epoch's variance as
    possible (Elsayed et al. 2016; the original ``OrthogonalPCA`` family).

    ``fit(*epochs)`` takes one array per epoch, each ``(conditions, time, units)`` (or
    ``(samples, units)``), and maximises
    ``mean_i tr(W_i' C_i W_i) / tr(C_i)`` subject to all ``W_i`` being orthonormal and
    mutually orthogonal.  Preprocessing as the original: centre each epoch on its
    across-condition mean at each time step, one set of unit ranges shared by all
    epochs (only used if ``soft_norm`` is a number).

    After ``fit``: ``W_`` (list of units x k bases, each rotated so its first column
    carries the most variance, as the original's final PCA step),
    ``var_exp_fit_[i]`` = share of epoch i's variance in its own subspace, and
    ``variance_fractions_[i, j]`` = share of epoch i's variance in subspace j (the
    numbers to report: e.g. how much hold-period variance lies in the movement
    subspace).  ``grad_norm_`` is the stationarity check (should be ~0).
    """

    def __init__(self, n_components=5, soft_norm: float | None = None, center: str = "condition",
                 n_restarts: int = 4, seed: int = 0, verbose: int = 0):
        self.n_components, self.soft_norm, self.center = n_components, soft_norm, center
        self.n_restarts, self.seed, self.verbose = n_restarts, seed, verbose

    def pre_process(self, X):
        return soft_normalize(X, self.soft_norm, center=self.center, unit_range=self.units_range)[0]

    def fit(self, *epochs):
        m = len(epochs)
        ks = (list(self.n_components) if np.ndim(self.n_components)
              else [int(self.n_components)] * m)
        flat = np.concatenate([_flat(e) for e in epochs], axis=0)
        self.units_range = np.nanmax(flat, 0) - np.nanmin(flat, 0)
        data = [self.pre_process(e) for e in epochs]
        n_units = flat.shape[1]
        if sum(ks) > n_units:
            raise ValueError("more dimensions requested than there are units")
        Cs, Craw = [], []
        for d in data:
            N = _flat(d)
            N = N - N.mean(0)
            C = N.T @ N / (N.shape[0] - 1)
            Craw.append(C)
            Cs.append(C / np.trace(C))
        # initialisation: each epoch's own top PCs, orthonormalised together
        init = np.concatenate([np.linalg.eigh(C)[1][:, ::-1][:, :k] for C, k in zip(Cs, ks)], 1)
        starts = [np.linalg.qr(init)[0]]
        rng = np.random.default_rng(self.seed)
        starts += [np.linalg.qr(rng.standard_normal((n_units, sum(ks))))[0]
                   for _ in range(self.n_restarts)]
        best = None
        for W0 in starts:
            W, f, g = _stiefel_max(Cs, ks, W0)
            if best is None or f > best[1] + 1e-12:
                best = (W, f, g)
        W, self.objective_, self.grad_norm_ = best
        edges = np.cumsum([0] + ks)
        self.W_, self.var_exp_ = [], []
        for i, d in enumerate(data):
            Wi = W[:, edges[i]:edges[i + 1]]
            pca = PCA(Wi.shape[1]).fit(d @ Wi)                    # sort, as the original
            self.W_.append(Wi @ pca.components_)
            self.var_exp_.append(pca.variance_explained)
        self.variance_fractions_ = np.array([[np.trace(Wj.T @ C @ Wj) / np.trace(C)
                                              for Wj in self.W_] for C in Craw])
        self.var_exp_fit_ = np.diag(self.variance_fractions_).copy()
        Wall = np.concatenate(self.W_, axis=1)
        self.constraints_satisfied = {
            "orthonormal": bool(np.allclose(Wall.T @ Wall, np.eye(Wall.shape[1]), atol=1e-8))}
        if self.verbose:
            print("variance in own subspace:", np.round(self.var_exp_fit_, 4),
                  "| stationarity:", f"{self.grad_norm_:.1e}")
        return self.W_


class OrthogonalPCA(OrthogonalSubspaces):
    """Original API: ``W_p, W_e = OrthogonalPCA(n_components=k).fit(data_prep, data_exe)``.

    Changes from the original: ``soft_norm`` defaults to ``None``; ``method`` is ignored
    (one exact solver instead of ``maniopt`` / ``scipy_minimize``)."""

    def __init__(self, n_components=20, soft_norm=None, method=None, verbose=1, **kw):
        super().__init__(n_components=n_components, soft_norm=soft_norm, verbose=verbose, **kw)

    def fit(self, data_prep, data_exe):
        W_p, W_e = super().fit(data_prep, data_exe)
        self.var_exp = {"plan": self.var_exp_[0], "exe": self.var_exp_[1]}
        self.var_exp_fit = {"plan": self.var_exp_fit_[0], "exe": self.var_exp_fit_[1]}
        return W_p, W_e


class OrthogonalPCA3(OrthogonalSubspaces):
    """Preparation-only, execution-only and shared prep+exe subspaces.

    The shared epoch is the preparation and execution data concatenated along time.
    **Bugs fixed:** the original never set ``units_range`` here (so ``fit`` failed on
    a fresh object), normalised the concatenated data a second time, and printed the
    shared subspace's variance against the execution covariance."""

    def __init__(self, n_components=5, soft_norm=None, verbose=1, **kw):
        super().__init__(n_components=n_components, soft_norm=soft_norm, verbose=verbose, **kw)

    def fit(self, data_prep, data_exe):
        both = np.concatenate([np.asarray(data_prep, float), np.asarray(data_exe, float)], axis=1)
        return tuple(super().fit(data_prep, data_exe, both))


class OrthogonalPCA3D(OrthogonalSubspaces):
    """Preparation, execution and posture (e.g. the hold before the cue) subspaces.
    Same fix as ``OrthogonalPCA3`` for ``units_range``."""

    def __init__(self, n_components=5, soft_norm=None, verbose=1, **kw):
        super().__init__(n_components=n_components, soft_norm=soft_norm, verbose=verbose, **kw)

    def fit(self, data_prep, data_exe, data_pos):
        return tuple(super().fit(data_prep, data_exe, data_pos))
