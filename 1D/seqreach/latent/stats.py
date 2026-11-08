"""Statistical analyses of population activity over time.

Ported from ``latent_analysis/stats.py`` by Mehrdad Kashefi (MIT licence, see
``LICENSE`` in this folder).  ``GDecompose`` and ``VarDecompose`` are unchanged apart
from plotting; ``TimePointClassifier`` and ``Model`` replace scikit-learn's ridge
estimators with exact numpy equivalents.

Data convention, as the original: ``(conditions or trials, time, units)``.

A caution specific to our agents: the targets are *in the observation* during the
hold, so decoding T1 or T2 from the GRU state is a pass-through check, not evidence of
planning.  Decode what the input does not contain (e.g. the upcoming peak speed), or
compare conditions where the input is identical (T2 hidden vs visible).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from . import utils


# ===================================================================== G decomposition
class GDecompose:
    """Decompose the condition second-moment matrix G (at each time step) into a
    non-negative combination of model G's built from features (RSA/PCM-style)."""

    def __init__(self):
        self.models_name, self.models_G, self.models_RDM = [], [], []

    def add_model_from_features(self, features, feature_names, normalize: bool = True):
        """``features``: (n_conditions, n_features)."""
        from scipy.spatial.distance import pdist, squareform
        features = np.asarray(features, dtype=float).reshape(len(features), -1)
        self.models_G.append(utils.get_G(features, normalize=normalize))
        self.models_name.append(feature_names)
        self.models_RDM.append(squareform(pdist(features)))

    def plot_models(self, what: str = "G", cmap: str = "inferno"):
        from .plot import RDM
        mats = self.models_G if what == "G" else self.models_RDM
        for name, m in zip(self.models_name, mats):
            RDM(m, cmap=cmap, title=name)

    def fit(self, data, normalize_G_emp: bool = True):
        """``data``: (conditions, [time,] units), or a list of two such arrays (two
        independent halves of the trials) for a cross-validated G.
        Returns ``(df_beta, df_fit)`` as the original."""
        if isinstance(data, list):
            data = [d[:, None, :] if d.ndim == 2 else d for d in data]
            n_t = data[0].shape[1]
        else:
            data = data[:, None, :] if data.ndim == 2 else data
            n_t = data.shape[1]
        X = np.stack([g.ravel() for g in self.models_G], axis=1)
        self.rows_beta, self.rows_fit = [], []
        for t in range(n_t):
            if isinstance(data, list):
                A, B = data[0][:, t, :], data[1][:, t, :]
                K, N = A.shape
                H = np.eye(K) - np.ones((K, K)) / K
                G = (H @ A @ B.T @ H.T) / N
                if normalize_G_emp:
                    G = G / np.trace(G)
            else:
                G = utils.get_G(data[:, t, :], normalize=normalize_G_emp)
            beta = nnls(X, G.ravel())[0]
            fss = [np.trace(gm * b) for gm, b in zip(self.models_G, beta)]
            for name, b in zip(self.models_name, beta):
                self.rows_beta.append({"feature": name, "beta": b, "time": t})
            tss = np.trace(G)
            self.rows_fit.append({"fss": np.sum(fss), "tss": tss, "r2": np.sum(fss) / tss,
                                  "time": t})
        return pd.DataFrame(self.rows_beta), pd.DataFrame(self.rows_fit)


# ===================================================================== variance decomposition
class VarDecompose:
    """Split the across-condition variance of the state, at each time step, among task
    factors (and their interactions) given as indicator vectors.

    ``Indicators``: ``{name: labels}`` with one label per condition; a key ``"a:b"``
    is an interaction (supply the combined label), orthogonalised against the main
    effects ``a`` and ``b`` when ``ortho_interaction`` is set.  ``fit(Y)`` with ``Y``
    ``(conditions, time, units)`` returns ``(tss, fss)``.
    """

    class G_maker:
        def __init__(self):
            self.G, self.name, self.num_models = [], [], 0

        def add(self, G, name):
            self.G.append(G)
            self.name.append(name)
            self.num_models += 1

        def plot(self):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, len(self.G), figsize=(2.2 * len(self.G), 2.4))
            for a, g, n in zip(np.atleast_1d(ax), self.G, self.name):
                a.imshow(g)
                a.set_title(n)
                a.set_xticks([])
                a.set_yticks([])
            return fig

    def __init__(self, Indicators: dict, ortho_interaction: bool = True, verbose: bool = False,
                 **kw):
        ortho_interaction = kw.pop("ortho_ineraction", ortho_interaction)  # original spelling
        self.G_m = self.G_maker()
        for key, vec in Indicators.items():
            D = utils.get_indicator_matrix(vec)
            if ortho_interaction and ":" in key:
                a, b = key.split(":")
                Xm = np.c_[utils.get_indicator_matrix(Indicators[a]),
                           utils.get_indicator_matrix(Indicators[b])]
                D = D - Xm @ np.linalg.pinv(Xm) @ D
            K = D.shape[0]
            H = np.eye(K) - np.ones((K, K)) / K
            g = H @ D @ D.T @ H.T
            self.G_m.add(g / np.trace(g), key)
        if verbose:
            self.G_m.plot()

    def IndicatorMatrix(self, what, vec):
        return utils.get_indicator_matrix(vec)

    def fit(self, Y):
        Y = np.asarray(Y, dtype=float)
        self.num_cond = Y.shape[0]
        X = np.stack([g.ravel() for g in self.G_m.G], axis=1)
        n_t = Y.shape[1]
        self.beta = np.zeros((self.G_m.num_models, n_t))
        self.tss = np.zeros((n_t, 1))
        self.fss = np.zeros((self.G_m.num_models, n_t))
        K = self.num_cond
        H = np.eye(K) - np.ones((K, K)) / K
        for t in range(n_t):
            y = Y[:, t, :]
            G_emp = H @ y @ y.T @ H.T
            self.beta[:, t] = nnls(X, G_emp.ravel())[0]
            self.tss[t] = np.trace(G_emp)
            for f, gm in enumerate(self.G_m.G):
                self.fss[f, t] = np.trace(gm) * self.beta[f, t]
        return self.tss, self.fss

    def plot(self, times=None, ax=None):
        import matplotlib.pyplot as plt
        t = np.arange(self.fss.shape[1]) if times is None else times
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].plot(t, self.tss, color="k", label="total")
        ax[0].plot(t, self.fss.sum(0), color="k", ls="--", label="explained")
        for f, n in enumerate(self.G_m.name):
            ax[0].plot(t, self.fss[f], label=n)
        ax[0].set_ylabel("variance")
        ax[0].legend(fontsize=8)
        share = self.fss / (self.fss.sum(0, keepdims=True) + np.finfo(float).eps)
        for f, n in enumerate(self.G_m.name):
            ax[1].plot(t, share[f], label=n)
        ax[1].set_ylabel("share of explained variance")
        return ax


# ===================================================================== ridge helpers
def _ridge_loo_alpha(X, Y, alphas):
    """Exact leave-one-out squared error of ridge (unpenalised intercept) per alpha.

    With X centred, the hat matrix is ``J/n + Xc (Xc'Xc + a I)^-1 Xc'``, so the LOO
    residual is ``e_i / (1 - h_ii)`` exactly -- what scikit-learn's ``RidgeCV`` /
    ``RidgeClassifierCV`` (default ``cv=None``) minimise.
    """
    n = X.shape[0]
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    UtY = U.T @ Yc
    errs = []
    for a in alphas:
        d = s ** 2 / (s ** 2 + a)
        fit = U @ (d[:, None] * UtY)
        h = 1.0 / n + np.sum((U ** 2) * d[None, :], axis=1)
        errs.append(np.mean(((Yc - fit) / (1 - h)[:, None]) ** 2))
    return float(alphas[int(np.argmin(errs))])


def _ridge_fit(X, Y, alpha, fit_intercept=True):
    if fit_intercept:
        mx, my = X.mean(0), Y.mean(0)
    else:
        mx, my = np.zeros(X.shape[1]), np.zeros(Y.shape[1])
    Xc, Yc = X - mx, Y - my
    p = X.shape[1]
    if p <= X.shape[0]:
        B = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ Yc)
    else:
        B = Xc.T @ np.linalg.solve(Xc @ Xc.T + alpha * np.eye(X.shape[0]), Yc)
    return B, my - mx @ B


# ===================================================================== decoding over time
class TimePointClassifier:
    """Cross-validated decoding of a condition label at every time step, with a
    shuffled-label chance level.

    Same procedure as the original: at each time step, ``num_sampling_rep`` repeats of
    shuffled ``num_fold``-fold CV with a ridge classifier whose penalty is chosen from
    ``(0.1, 10, 100, 1000)`` by leave-one-out on the training folds, and the same with
    permuted labels.  The ridge classifier is a numpy re-implementation of
    scikit-learn's ``RidgeClassifierCV`` (targets coded +-1, intercept unpenalised,
    exact LOO), so the multiprocessing pool is not needed.

    ``X``: (trials, time, variables); ``y``: (trials,).  Returns ``(acc, acc_chance)``.
    """

    def __init__(self, num_fold: int = 5, num_sampling_rep: int = 30,
                 alphas=(0.1, 10.0, 100.0, 1000.0), seed: int = 0, num_core=None):
        self.num_fold, self.num_sampling_rep = num_fold, num_sampling_rep
        self.alphas, self.seed = np.asarray(alphas, float), seed

    def _code(self, y):
        self.classes_, inv = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if K == 2:
            return (2.0 * inv - 1.0)[:, None], inv
        T = -np.ones((len(y), K))
        T[np.arange(len(y)), inv] = 1.0
        return T, inv

    def classifier(self, X, y_inv, T, rng, randomize_y=False):
        if randomize_y:
            p = rng.permutation(len(y_inv))
            y_inv, T = y_inv[p], T[p]
        idx = rng.permutation(len(y_inv))
        folds = np.array_split(idx, self.num_fold)
        score = []
        for i in range(self.num_fold):
            te = folds[i]
            tr = np.concatenate(folds[:i] + folds[i + 1:])
            a = _ridge_loo_alpha(X[tr], T[tr], self.alphas)
            B, b0 = _ridge_fit(X[tr], T[tr], a)
            pred = X[te] @ B + b0
            lab = (pred[:, 0] > 0).astype(int) if T.shape[1] == 1 else pred.argmax(1)
            score.append(np.mean(lab == y_inv[te]))
        return float(np.mean(score))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        T, inv = self._code(np.asarray(y))
        rng = np.random.default_rng(self.seed)
        acc = np.zeros((X.shape[1], 2))
        for t in range(X.shape[1]):
            Xt = X[:, t, :]
            acc[t, 0] = np.mean([self.classifier(Xt, inv, T, rng) for _ in range(self.num_sampling_rep)])
            acc[t, 1] = np.mean([self.classifier(Xt, inv, T, rng, randomize_y=True)
                                 for _ in range(self.num_sampling_rep)])
        self.acc, self.acc_chance = acc[:, 0], acc[:, 1]
        return self.acc, self.acc_chance

    def plot(self, times=None, ax=None):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        t = np.arange(len(self.acc)) if times is None else times
        ax.plot(t, self.acc, color="k", label="accuracy")
        ax.plot(t, self.acc_chance, color="r", lw=0.8, label="chance (shuffled)")
        ax.set_ylabel("accuracy")
        ax.legend()
        return ax


# ===================================================================== encoding models
def R2(Y_true, Y_pred, unit_eval: bool = False, centered: bool = True):
    """Coefficient of determination.

    Change from the original (which flagged it with "subtract the mean?"): by default
    the total sum of squares is taken around the mean of ``Y_true``.  Uncentred, a
    GRU unit's non-zero mean counts as explained variance.  ``centered=False`` gives
    the original.
    """
    Y_true, Y_pred = np.asarray(Y_true, float), np.asarray(Y_pred, float)
    ref = Y_true.mean(0) if centered else 0.0
    if unit_eval:
        return 1 - np.sum((Y_true - Y_pred) ** 2, 0) / np.sum((Y_true - ref) ** 2, 0)
    return 1 - np.sum((Y_true - Y_pred) ** 2) / np.sum((Y_true - ref) ** 2)


def r(Y_true, Y_pred, unit_eval: bool = False, centered: bool = True):
    """Correlation between data and prediction.  Change from the original: Pearson
    (centred) by default; ``centered=False`` gives the original's uncentred cosine."""
    Y_true, Y_pred = np.asarray(Y_true, float), np.asarray(Y_pred, float)
    if centered:
        Y_true, Y_pred = Y_true - Y_true.mean(0), Y_pred - Y_pred.mean(0)
    ax = 0 if unit_eval else None
    return np.sum(Y_true * Y_pred, ax) / np.sqrt(np.sum(Y_true ** 2, ax) * np.sum(Y_pred ** 2, ax))


class Model:
    """Encoding model fitted at every time step: ``Y_t ~ M B_t`` (ridge), evaluated by
    k-fold CV (the original's "RMD-like" models).

    ``M``: (conditions, features).  ``fit(Y, "RidgeCV")`` with ``Y`` (conditions, time,
    units).  The penalty is chosen from ``lambda_list`` by an inner k-fold on the
    training conditions, scored with ``fit_score`` ('r' or 'r2') on all units pooled,
    which is what scikit-learn's ``RidgeCV(cv=k, scoring=...)`` did in the original.
    The ``"PCM"`` method needs the PcmPy toolbox and is not ported.
    After ``fit``: ``self.r`` and ``self.R2`` (time x 1, or time x units with
    ``unit_eval``), averaged over outer folds.
    """

    def __init__(self, name, M, fit_intercept: bool = True, **kwargs):
        self.name, self.M = name, np.asarray(M, dtype=float)
        self.num_sample, self.num_feature = self.M.shape
        self.fit_intercept = fit_intercept
        self.feature_indicator = kwargs.get("feature_indicator",
                                            np.zeros(self.M.shape[1], dtype=np.int8))

    def fit(self, Y, method: str = "RidgeCV", **kwargs):
        if method != "RidgeCV":
            raise NotImplementedError("only 'RidgeCV' is ported (PCM needs PcmPy)")
        self.method = method
        self.n_kfold = kwargs.get("n_kfold", 4)
        self.unit_eval = kwargs.get("unit_eval", False)
        self.n_kfold_in = kwargs.get("n_kfold_in", 2)
        self.lambda_list = kwargs.get("lambda_list", [1e-2, 1e-1, 1, 1e1, 1e2, 1e3])
        score_fn = {"r": r, "r2": R2}[kwargs.get("fit_score", "r")]
        rng = np.random.default_rng(kwargs.get("seed", 0))
        Y = np.asarray(Y, dtype=float)
        n_t, n_u = Y.shape[1], Y.shape[-1]
        width = n_u if self.unit_eval else 1
        self.r = np.zeros((self.n_kfold, n_t, width))
        self.R2 = np.zeros((self.n_kfold, n_t, width))

        def splits(n, k):
            idx = rng.permutation(n)
            f = np.array_split(idx, k)
            return [(np.concatenate(f[:i] + f[i + 1:]), f[i]) for i in range(k)]

        for T in range(n_t):
            YY = Y[:, T, :]
            for fold, (tr, te) in enumerate(splits(len(YY), self.n_kfold)):
                best, best_s = None, -np.inf
                for lam in self.lambda_list:
                    s = []
                    for tr2, te2 in splits(len(tr), self.n_kfold_in):
                        B, b0 = _ridge_fit(self.M[tr][tr2], YY[tr][tr2], lam, self.fit_intercept)
                        s.append(score_fn(YY[tr][te2], self.M[tr][te2] @ B + b0))
                    if np.mean(s) > best_s:
                        best, best_s = lam, np.mean(s)
                B, b0 = _ridge_fit(self.M[tr], YY[tr], best, self.fit_intercept)
                pred = self.M[te] @ B + b0
                self.r[fold, T, :] = r(YY[te], pred, unit_eval=self.unit_eval)
                self.R2[fold, T, :] = R2(YY[te], pred, unit_eval=self.unit_eval)
        self.r, self.R2 = self.r.mean(0), self.R2.mean(0)
        return self
