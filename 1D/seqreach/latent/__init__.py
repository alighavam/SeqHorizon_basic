"""Latent analysis of the GRU state, adapted from Mehrdad Kashefi's toolbox.

Source: https://github.com/mehrdadkashefi/latent_analysis (commit ``980d8e4``,
2026-09-25), MIT licence -- the copyright notice is in ``LICENSE`` in this folder.
The module layout follows the original:

* ``transform`` -- PCA, FA, jPCA, dPCA / CVdPCA, orthogonal subspaces (OrthogonalPCA,
  OrthogonalPCA3, OrthogonalPCA3D)
* ``alignment`` -- CCA, Procrustes, cross-validated dissimilarity, alignment index
* ``stats`` -- G and variance decomposition, decoding over time, encoding models
* ``utils`` -- indicator matrices, condition means, G matrices, normalisation
* ``plot`` -- matplotlib versions of the original's plotly plots
* ``data`` -- new: from our rollouts to (trials, time, units) arrays

What changed, and why
---------------------
Adapted to a GRU instead of spiking neurons:

* No per-unit soft normalisation by default (the original divides by ``range + 5``
  spikes/s).  GRU units share one scale, and per-unit scaling would bend the geometry
  of the linear readout ``u = w.h + b``.  ``soft_norm=<number>`` restores it.
* Spike handling and the NWB / .mat readers are replaced by ``data`` (recording,
  event alignment, the readout direction).
* No new dependencies: scikit-learn's FA and ridge estimators, the ``dPCA`` package
  and ``pymanopt`` are replaced by numpy implementations of the same estimators,
  checked against known answers in ``tests/test_latent.py``.

Bugs fixed, each confirmed by running the original code on a case with a known
answer:

* ``FA.transform(ensure_orthogonality=True)`` projects onto ``U[:k, :].T`` (the
  first k *rows* of the left singular vectors) instead of ``U[:, :k]``, so not onto
  the factor subspace.
* ``jPCA``: the counter-clockwise check tests the cross product of two orthonormal
  vectors, which is always +-1, so the axis is always flipped and the rotation
  direction ends up depending on SVD signs (CCW in 11 of 40 test datasets).
  ``transform`` refits the PCA on the data it is given, so it is only right for the
  exact data passed to ``fit``.  The plane is the first eigenvalue pair
  ``np.linalg.eig`` returns, which is not sorted; usually the fastest, but not always
  (2-7 % of random skew matrices; with an odd ``num_latent`` it can be the zero
  eigenvalue).
* ``OrthogonalPCA3`` / ``OrthogonalPCA3D`` use ``units_range`` before setting it
  (AttributeError on a fresh object); ``OrthogonalPCA3`` also normalises the shared
  epoch a second time and prints its variance against the execution covariance.
* ``CCA_svd`` does not centre the data: unrelated data with non-zero means give a
  first canonical correlation of 0.999.
* ``alignment_index_crossval(same=None)`` does not infer ``same`` as its docstring
  says, so passing the same array twice uses overlapping halves.
* ``PCA`` stores the eigenvalue variances as ``varianve`` (typo), divided by
  ``n_units - 1`` instead of ``n_samples - 1``.

Conventions changed (not bugs): ``stats.R2`` / ``stats.r`` are centred by default
(the original is uncentred and marks it as an open question; ``centered=False`` gives
the original), and no soft normalisation by default.

Not ported: ``Model.fit(method="PCM")`` (needs PcmPy), ``Kernel_Cosine`` (reach angles
have no meaning in 1-D; ``utils.GaussianBasis1D`` replaces ``Kernel_Gaussian``), and
``CCA_sklearn``.
"""

from . import alignment, data, plot, stats, transform, utils  # noqa: F401
