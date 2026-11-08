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

Bugs fixed (worth telling the author):

* ``FA.transform(ensure_orthogonality=True)`` used rows instead of columns of the
  singular vectors, so it did not project onto the factor subspace.
* ``jPCA`` took the first eigenvalue pair ``np.linalg.eig`` returned (unsorted), not
  the fastest rotation; its counter-clockwise check flipped the axis whenever the
  cross product was non-zero; ``transform`` refitted the PCA on new data.
* ``OrthogonalPCA3`` / ``OrthogonalPCA3D`` used ``units_range`` before setting it
  (fails on a fresh object); ``OrthogonalPCA3`` normalised the shared epoch twice.
* ``CCA_svd`` did not centre the data, so the mean pattern counted as a shared
  dimension.
* ``alignment_index_crossval(same=None)`` did not infer ``same`` as documented.
* ``PCA.variance`` divided by ``n_units - 1`` instead of ``n_samples - 1``.
* ``cosine_similarity`` shifted each matrix by its maximum before the cosine.
* ``stats.R2`` / ``stats.r`` were uncentred (flagged in the original as a question);
  they are centred by default now (``centered=False`` gives the original).

Not ported: ``Model.fit(method="PCM")`` (needs PcmPy), ``Kernel_Cosine`` (reach angles
have no meaning in 1-D; ``utils.GaussianBasis1D`` replaces ``Kernel_Gaussian``), and
``CCA_sklearn``.
"""

from . import alignment, data, plot, stats, transform, utils  # noqa: F401
