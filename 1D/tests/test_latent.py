"""Correctness of the latent-analysis port (``seqreach.latent``).

The original toolbox leaned on scikit-learn, the ``dPCA`` package and pymanopt; the port
replaces them with numpy.  Each method is pinned here against data built to a known
answer, and the bugs fixed in the port each get a test so they cannot come back.
The last tests check the rollout adapter against a real agent (skipped if absent).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seqreach.latent import alignment, data, stats, transform, utils  # noqa: E402

RUN = Path(__file__).resolve().parents[1] / "runs" / "v2_catch_s1"


def rot_data(n_cond=12, n_t=40, n_units=30, omega=0.25, noise=0.0, seed=0, plane=None):
    """Conditions start at different angles and rotate in one plane of unit space."""
    rng = np.random.default_rng(seed)
    if plane is None:
        plane = np.linalg.qr(rng.standard_normal((n_units, 2)))[0]
    ang0 = np.linspace(0, 2 * np.pi, n_cond, endpoint=False)
    t = np.arange(n_t)
    z = np.stack([np.cos(ang0[:, None] + omega * t), np.sin(ang0[:, None] + omega * t)], -1)
    X = z @ plane.T + noise * rng.standard_normal((n_cond, n_t, n_units))
    return X, plane


def principal_cos(A, B):
    """Cosines of the principal angles between span(A) and span(B)."""
    qa, qb = np.linalg.qr(A)[0], np.linalg.qr(B)[0]
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


# ===================================================================== utils
def test_condition_mean_and_indicators():
    D = np.arange(12.0).reshape(6, 2)
    lab = np.array(["b", "a", "b", "a", "c", "c"])
    m = utils.get_condition_mean(D, lab)
    np.testing.assert_allclose(m, [[4, 5], [2, 3], [9, 10]])   # a, b, c
    I = utils.get_indicator_matrix(lab)
    assert I.shape == (6, 3) and np.all(I.sum(1) == 1)
    D[0, 0] = np.nan                                              # NaNs are skipped
    assert utils.get_condition_mean(D, lab)[1, 0] == 4


def test_G_and_distances():
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((5, 20))
    G = utils.get_G(Y)
    dist = np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1) / 20)
    np.testing.assert_allclose(utils.dist_to_G(dist), G, atol=1e-10)
    assert abs(utils.cosine_similarity(G, 3 * G) - 1) < 1e-12


def test_soft_normalize_default_keeps_scale():
    X = np.random.default_rng(2).standard_normal((4, 10, 6))
    Xn, rng_ = utils.soft_normalize(X)
    np.testing.assert_allclose(Xn, X - X.mean(0, keepdims=True))
    Xs, _ = utils.soft_normalize(X, soft_norm=0.1)
    np.testing.assert_allclose(Xs, Xn / (np.ptp(Xn.reshape(-1, 6), 0) + 0.1))


# ===================================================================== PCA / FA
def test_pca_matches_svd():
    X = np.random.default_rng(3).standard_normal((200, 8)) * np.arange(1, 9)
    p = transform.PCA(3).fit(X)
    ev = np.linalg.eigvalsh(np.cov(X.T))[::-1]
    np.testing.assert_allclose(p.variance[:8], ev, rtol=1e-10)
    assert abs(p.variance_explained.sum() - 1) < 1e-12
    assert transform.PCA(3).fit(X.reshape(20, 10, 8)).transform(X.reshape(20, 10, 8)).shape == (20, 10, 3)


def test_fa_recovers_loadings_and_orthogonal_projection():
    rng = np.random.default_rng(4)
    W = rng.standard_normal((20, 3))
    Z = rng.standard_normal((3000, 3))
    psi = rng.uniform(0.2, 1.0, 20)
    X = Z @ W.T + rng.standard_normal((3000, 20)) * np.sqrt(psi) + 5.0
    fa = transform.FA(3).fit(X)
    assert np.all(np.diff(fa.loglik_) > -1e-6)                   # EM never goes down
    assert principal_cos(fa.loadings_, W).min() > 0.99
    np.testing.assert_allclose(fa.noise_variance_, psi, rtol=0.15)
    # the fixed projection: onto an orthonormal basis of the loadings' span
    L = fa.transform(X, ensure_orthogonality=True)
    Q = np.linalg.svd(fa.loadings_, full_matrices=False)[0]
    np.testing.assert_allclose(L, (X - X.mean(0)) @ Q, atol=1e-8)


# ===================================================================== jPCA
def test_jpca_finds_rotation_plane_counterclockwise():
    X, plane = rot_data(noise=0.01)
    X = X + 0.3 * np.random.default_rng(5).standard_normal((1, 1, 30))  # cross-cond. mean
    jp = transform.jPCA(num_latent=4).fit(X)
    assert principal_cos(jp.components_units_, plane).min() > 0.999
    assert abs(jp.frequencies_[0] - np.sin(0.25)) < 0.02          # one-step rotation
    assert jp.variance_explained_rot_fit > 0.95
    Y = jp.transform()
    cross = Y[:, :-1, 0] * Y[:, 1:, 1] - Y[:, :-1, 1] * Y[:, 1:, 0]
    assert np.all(cross > 0)                                      # counter-clockwise
    np.testing.assert_allclose(jp.M, -jp.M.T)


def test_jpca_picks_fastest_plane():
    rng = np.random.default_rng(6)
    P = np.linalg.qr(rng.standard_normal((30, 4)))[0]
    slow, _ = rot_data(omega=0.05, plane=P[:, :2])
    fast, _ = rot_data(omega=0.30, plane=P[:, 2:])
    jp = transform.jPCA(num_latent=4).fit(slow + 0.8 * fast)
    assert principal_cos(jp.components_units_, P[:, 2:]).min() > 0.99
    assert jp.frequencies_[0] > jp.frequencies_[1]


# ===================================================================== dPCA
def dpca_data(seed=7, n_units=25, noise=0.0, reps=0):
    """time-only signal on one axis, condition-only on another, interaction on a third."""
    rng = np.random.default_rng(seed)
    E = np.linalg.qr(rng.standard_normal((n_units, 3)))[0]
    t = np.linspace(0, 1, 30)
    c = np.linspace(-1, 1, 6)
    sig_t = np.sin(2 * np.pi * t)[None, :] * np.ones((6, 1))
    sig_c = c[:, None] * np.ones((1, 30))
    sig_ct = np.outer(c, t - t.mean()) * 2
    sig_ct -= sig_ct.mean(0, keepdims=True) + sig_ct.mean(1, keepdims=True) - sig_ct.mean()
    X = (sig_t[..., None] * E[:, 0] + 0.7 * sig_c[..., None] * E[:, 1]
         + 0.5 * sig_ct[..., None] * E[:, 2]) + 2.0
    if reps:
        trials = X[None] + noise * rng.standard_normal((reps,) + X.shape)
        return X, E, trials
    return X, E


def test_marginalizations_sum_and_are_orthogonal():
    X = np.random.default_rng(8).standard_normal((3, 4, 5, 6))
    Xc = X - X.reshape(-1, 6).mean(0)
    m = transform.marginalize(Xc, "abt")
    np.testing.assert_allclose(sum(m.values()), Xc, atol=1e-12)
    keys = list(m)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert abs(np.sum(m[keys[i]] * m[keys[j]])) < 1e-9


def test_dpca_demixes_known_components():
    X, E = dpca_data()
    dp = transform.dPCA(labels="ct", n_components=1).fit(X)
    for key, axis in (("t", 0), ("c", 1), ("ct", 2)):
        d = dp.decoders_[key][0]
        assert abs(d @ E[:, axis]) / np.linalg.norm(d) > 0.999
        assert dp.marginal_purity_[key][0][key] > 0.999
    ev = sum(v[0] for v in dp.explained_variance_ratio_.values())
    assert abs(ev - 1) < 1e-8                                      # three rank-1 parts
    Z = dp.transform()
    assert Z["t"].shape == (6, 30, 1)


def test_dpca_auto_regularization_and_cvdpca():
    X, E, trials = dpca_data(noise=0.3, reps=8)
    dp = transform.dPCA(labels="ct", n_components=1, regularizer="auto", n_cv=3)
    dp.fit(trials.mean(0), trials)
    assert dp.lambda_ in dp.cv_grid_
    assert abs(dp.decoders_["c"][0] @ E[:, 1]) / np.linalg.norm(dp.decoders_["c"][0]) > 0.95
    # CVdPCA takes single trials + labels, as the original
    single = trials.reshape(-1, 30, 25)                            # (reps*cond, t, u)
    labels = np.tile(np.arange(6), 8)
    cv = transform.CVdPCA(n_components=1, n_cv=2).fit(single, labels)
    assert set(cv.decoders_) == {"c", "t", "ct"}


# ===================================================================== orthogonal subspaces
def test_orthogonal_pca_disjoint_supports():
    rng = np.random.default_rng(9)
    B = np.linalg.qr(rng.standard_normal((20, 6)))[0]
    prep = rng.standard_normal((8, 15, 3)) * [3, 2, 1] @ B[:, :3].T
    exe = rng.standard_normal((8, 15, 3)) * [3, 2, 1] @ B[:, 3:].T
    op = transform.OrthogonalPCA(n_components=3, verbose=0)
    W_p, W_e = op.fit(prep, exe)
    assert principal_cos(W_p, B[:, :3]).min() > 0.999
    assert principal_cos(W_e, B[:, 3:]).min() > 0.999
    np.testing.assert_allclose(np.diag(op.variance_fractions_), 1, atol=1e-8)
    assert op.variance_fractions_[0, 1] < 1e-8 and op.constraints_satisfied["orthonormal"]


def test_orthogonal_pca_overlapping_is_stationary_optimum():
    rng = np.random.default_rng(10)
    A = rng.standard_normal((20, 20))
    prep = rng.standard_normal((6, 20, 20)) @ A
    exe = rng.standard_normal((6, 20, 20)) @ (A + rng.standard_normal((20, 20)))
    op = transform.OrthogonalSubspaces(n_components=3, n_restarts=6)
    W_p, W_e = op.fit(prep, exe)
    assert op.grad_norm_ < 1e-5
    # no better than each epoch's own unconstrained PCA, no worse than the PCA start
    assert np.all(op.var_exp_fit_ <= [transform.PCA(3).fit(prep - prep.mean(0)).variance_explained[:3].sum() + 1e-9,
                                      transform.PCA(3).fit(exe - exe.mean(0)).variance_explained[:3].sum() + 1e-9])
    W = np.concatenate([W_p, W_e], 1)
    np.testing.assert_allclose(W.T @ W, np.eye(6), atol=1e-8)


def test_orthogonal_pca3_runs_on_fresh_object():
    rng = np.random.default_rng(11)
    prep, exe = rng.standard_normal((5, 10, 15)), rng.standard_normal((5, 12, 15))
    W = transform.OrthogonalPCA3(n_components=2, verbose=0).fit(prep, exe)
    assert len(W) == 3 and all(w.shape == (15, 2) for w in W)
    W = transform.OrthogonalPCA3D(n_components=2, verbose=0).fit(prep, exe, prep[:, :4])
    assert len(W) == 3


# ===================================================================== alignment
def test_cca_and_procrustes_recover_a_linear_map():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((300, 5))
    R = np.linalg.qr(rng.standard_normal((5, 5)))[0]
    Y = 2.5 * X @ R + 7.0
    cca = alignment.CCA_svd(n_components=5).fit(X + 3.0, Y)
    np.testing.assert_allclose(cca.cc, 1, atol=1e-10)
    assert cca.score(X + 3.0, Y) > 1 - 1e-10
    d, Z, tf = alignment.Procrustes().fit(X, Y)
    assert d < 1e-12
    np.testing.assert_allclose(Z, X, atol=1e-10)


def test_cca_is_centred():
    rng = np.random.default_rng(13)
    X, Y = rng.standard_normal((200, 3)), rng.standard_normal((200, 3))
    shift = alignment.CCA_svd(3).fit(X + 50, Y - 50).cc
    np.testing.assert_allclose(shift, alignment.CCA_svd(3).fit(X, Y).cc, atol=1e-10)


def test_dissimilarity_cond_avr():
    rng = np.random.default_rng(14)
    base, _ = rot_data(n_cond=6, n_t=20, n_units=12)
    labels = np.repeat(np.arange(6), 10)
    Xtr = base[labels] + 0.05 * rng.standard_normal((60, 20, 12))
    R = np.linalg.qr(rng.standard_normal((12, 12)))[0]
    Ytr = base[labels] @ R + 0.05 * rng.standard_normal((60, 20, 12))
    sc, _ = alignment.get_dissimilarity_cond_avr(Xtr, labels, Ytr, labels,
                                                 method="Procrustes", n_folds=2)
    assert sc.Score.mean() > 0.95


def test_alignment_index():
    rng = np.random.default_rng(15)
    B = np.linalg.qr(rng.standard_normal((20, 6)))[0]
    A = rng.standard_normal((400, 3)) @ B[:, :3].T
    C = rng.standard_normal((400, 3)) @ B[:, 3:].T
    assert abs(alignment.alignment_index(A, A, n_dim=3) - 1) < 1e-10
    # not exactly 0: both are centred on their *joint* mean (as the original), and the
    # sample means of A and C differ slightly
    assert alignment.alignment_index(A, C, n_dim=3) < 1e-3
    noisy = lambda M: M + 0.05 * rng.standard_normal(M.shape)
    A3, C3 = noisy(A).reshape(40, 10, 20), noisy(C).reshape(40, 10, 20)
    # the average of the two directions, each normalised by its split-half value: ~1
    assert abs(alignment.alignment_index_crossval(A3, A3, n_dim=3, seed=0) - 1) < 0.05
    assert alignment.alignment_index_crossval(A3, C3, n_dim=3, seed=0) < 0.1


# ===================================================================== stats
def test_var_decompose_recovers_factor_variances():
    a = np.repeat([0, 1], 4)
    b = np.tile(np.repeat([0, 1], 2), 2)
    ab = a * 2 + b
    rng = np.random.default_rng(16)
    ua, ub = rng.standard_normal(30), rng.standard_normal(30)
    Y = (3 * (a - 0.5))[:, None] * ua + (1 * (b - 0.5))[:, None] * ub
    Y = np.repeat(Y[:, None, :], 5, axis=1)
    vd = stats.VarDecompose({"a": a, "b": b, "a:b": ab})
    tss, fss = vd.fit(Y)
    share = fss[:, 0] / tss[0, 0]
    ratio = share[0] / share[1]
    assert abs(share.sum() - 1) < 0.02 and abs(ratio - 9 * (ua @ ua) / (ub @ ub)) < 0.5 * ratio
    assert share[2] < 0.02


def test_gdecompose_weights():
    rng = np.random.default_rng(17)
    f1, f2 = rng.standard_normal((8, 2)), rng.standard_normal((8, 3))
    gd = stats.GDecompose()
    gd.add_model_from_features(f1, "f1")
    gd.add_model_from_features(f2, "f2")
    Y = np.concatenate([f1 * 2, f2 * 0.0], axis=1)
    betas, fit = gd.fit(Y)
    b = betas.set_index("feature").beta
    assert b["f1"] > 0.99 and b["f2"] < 1e-6 and fit.r2.iloc[0] > 0.99


def test_ridge_loo_matches_brute_force():
    rng = np.random.default_rng(18)
    X, Y = rng.standard_normal((25, 6)), rng.standard_normal((25, 2))
    alphas = np.array([0.1, 1.0, 10.0, 100.0])
    errs = []
    for a in alphas:
        e = 0.0
        for i in range(25):
            m = np.arange(25) != i
            B, b0 = stats._ridge_fit(X[m], Y[m], a)
            e += np.sum((Y[i] - X[i] @ B - b0) ** 2)
        errs.append(e)
    assert stats._ridge_loo_alpha(X, Y, alphas) == alphas[int(np.argmin(errs))]


def test_timepoint_classifier():
    rng = np.random.default_rng(19)
    y = np.repeat([0, 1], 40)
    X = rng.standard_normal((80, 3, 10))
    X[:, 1, 0] += 4 * (y - 0.5)                                  # decodable at t = 1 only
    tc = stats.TimePointClassifier(num_sampling_rep=3)
    acc, chance = tc.fit(X, y)
    assert acc[1] > 0.95 and acc[0] < 0.75 and np.all(np.abs(chance - 0.5) < 0.15)


def test_encoding_model_and_centred_r2():
    rng = np.random.default_rng(20)
    M = rng.standard_normal((60, 3))
    Y = (M @ rng.standard_normal((3, 8)))[:, None, :] + 10.0     # big unit means
    Y = Y + 0.1 * rng.standard_normal((60, 4, 8))
    mod = stats.Model("lin", M).fit(Y, "RidgeCV")
    assert np.all(mod.R2 > 0.9)
    # a constant prediction at the right mean explains nothing once centred
    yt = Y[:, 0, :]
    assert abs(stats.R2(yt, np.broadcast_to(yt.mean(0), yt.shape))) < 1e-12
    assert stats.R2(yt, np.broadcast_to(yt.mean(0), yt.shape), centered=False) > 0.9


# ===================================================================== rollout adapter
def fake_traj():
    T, N, H = 30, 3, 4
    h = np.arange(T * N * H, dtype=float).reshape(T, N, H)
    return {"h": h, "go_step": np.array([5, 10, 2]), "end_step": np.array([20, 12, 29]),
            "entry_step": np.array([[8, -1, -1], [-1, -1, -1], [6, 9, -1]]),
            "acq_step": np.full((N, 3), -1), "u": h[..., 0], "cue_delay": 4, "dt": 0.02}


def test_align_indexes_events_and_masks_outside_trial():
    tr = fake_traj()
    X, lags = data.align(tr, "cue", window=(3, 6))
    assert X.shape == (3, 10, 4) and lags[0] == -3
    np.testing.assert_array_equal(X[0, 3], tr["h"][9, 0])        # go 5 + cue delay 4
    # trial 1: cue at 14, ends at 12, so only lags -3, -2 (steps 11, 12) are inside it
    assert np.isnan(X[1, 2:]).all() and not np.isnan(X[1, :2]).any()
    E, _ = data.align(tr, "entry1", window=(0, 2))
    assert np.isnan(E[1]).all()                                   # never entered T1
    np.testing.assert_array_equal(E[2, 0], tr["h"][6, 2])
    u, _ = data.align(tr, "go", window=(5, 0), field="u")
    assert u.shape == (3, 6) and np.isnan(u[2, :3]).all()        # before step 0


def test_balanced_trials():
    X = np.arange(7.0)[:, None]
    lab = np.array([0, 0, 0, 1, 1, 2, 2])
    out, lev = data.balanced_trials(X, lab)
    assert out.shape == (3, 3, 1)
    assert set(out[:, 1, 0]) <= {3.0, 4.0} and set(out[:, 1, 0]) == {3.0, 4.0}


@pytest.mark.skipif(not (RUN / "checkpoint.pt").exists(), reason="trained run not present")
def test_record_readout_reproduces_command():
    """The recorded state, through the recorded readout, gives the command the agent
    actually sent: the alignment of h and u in ``probes.rollout`` is right."""
    n = 6
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-0.4, 0.4, n)
    tg = np.stack([x0 + 0.3, x0 - 0.2, x0 + 0.2], 1)
    spec = {"n_targets": np.full(n, 2), "preview": np.full(n, 2), "start_x": x0,
            "targets": tg, "delay_steps": np.full(n, 20), "catch": np.zeros(n, bool)}
    tr = data.record(RUN, spec)
    assert tr["h"].shape[1:] == (n, 128)
    for i in range(n):
        k = np.arange(tr["end_step"][i] + 1)
        u_hat = np.clip(tr["h"][k, i] @ tr["readout_w"] + tr["readout_b"], -tr["u_max"], tr["u_max"])
        np.testing.assert_allclose(u_hat, tr["u"][k, i], atol=1e-4)
    X, lags = data.align(tr, "cue", window=(5, 10))
    assert X.shape == (n, 16, 128) and not np.isnan(X).any()
