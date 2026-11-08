"""Plots for latent analyses.

The original (``latent_analysis/plot.py`` by Mehrdad Kashefi, MIT licence) draws with
plotly; these are matplotlib equivalents of ``RDM``, ``line_1d``, ``line_2d`` and
``line_3d`` so they render in our notebooks and figure files without plotly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def RDM(x_dis, ax=None, cmap: str = "inferno", cond_label=None, colorbar: bool = False,
        title: str = ""):
    """Show a condition-by-condition matrix (RDM or G)."""
    ax = ax or plt.subplots(figsize=(3, 3))[1]
    im = ax.imshow(x_dis, cmap=cmap)
    if cond_label is not None:
        ax.set_xticks(range(len(cond_label)), cond_label, rotation=90)
        ax.set_yticks(range(len(cond_label)), cond_label)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if colorbar:
        plt.colorbar(im, ax=ax)
    ax.set_title(title)
    return ax


def _colors(n, colors):
    if colors is None:
        return ["k"] * n
    if isinstance(colors, str):
        return [colors] * n
    return list(colors)


def line_1d(data, t=None, ax=None, colors=None, alpha: float = 1.0, labels=None):
    """``data``: (conditions, time) -- one line per condition."""
    data = np.asarray(data)
    ax = ax or plt.gca()
    t = np.arange(data.shape[1]) if t is None else t
    for i, c in enumerate(_colors(len(data), colors)):
        ax.plot(t, data[i], color=c, alpha=alpha,
                label=None if labels is None else labels[i])
    return ax


def line_2d(data, ax=None, colors=None, alpha: float = 1.0, mark_start: bool = True,
            equal: bool = False):
    """``data``: (conditions, time, 2) -- trajectories in a plane (e.g. jPCs)."""
    data = np.asarray(data)
    ax = ax or plt.gca()
    for i, c in enumerate(_colors(len(data), colors)):
        ax.plot(data[i, :, 0], data[i, :, 1], color=c, alpha=alpha)
        if mark_start:
            ax.plot(data[i, 0, 0], data[i, 0, 1], "o", color=c, ms=3)
    if equal:
        ax.set_aspect("equal")
    return ax


def line_3d(data, ax=None, colors=None, alpha: float = 1.0, mark_start: bool = True):
    """``data``: (conditions, time, 3)."""
    data = np.asarray(data)
    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")
    for i, c in enumerate(_colors(len(data), colors)):
        ax.plot(data[i, :, 0], data[i, :, 1], data[i, :, 2], color=c, alpha=alpha)
        if mark_start:
            ax.scatter(*data[i, 0, :3], color=c, s=8)
    return ax
