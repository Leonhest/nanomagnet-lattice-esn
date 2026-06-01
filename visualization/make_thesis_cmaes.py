"""Original illustration of one CMA-ES iteration over two generations.

Layout (2 rows x 3 cols), mirroring the standard pedagogical figure:
  columns: Sampling -> Selection -> Updated distribution
  rows:    Generation 1 (top), Generation 2 (bottom)
Over the two generations the search distribution adapts from an isotropic
circle to an ellipse aligned with the objective's valley. Original figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

np.random.seed(4)


def rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# --- objective: rotated anisotropic quadratic, optimum at origin, valley ~30 deg ---
H = rot(30) @ np.diag([0.22, 1.0]) @ rot(30).T


def f(P):
    return np.einsum('...i,ij,...j->...', P, H, P)


lim = 5.0
g = np.linspace(-lim, lim, 320)
XX, YY = np.meshgrid(g, g)
ZZ = f(np.stack([XX, YY], -1))


def cov_ell(C, m, ns, **kw):
    v, V = np.linalg.eigh(C)
    o = v.argsort()[::-1]
    v, V = v[o], V[:, o]
    ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
    w, h = 2 * ns * np.sqrt(v)
    return Ellipse(m, w, h, angle=ang, **kw)


def contours(ax):
    ax.contour(XX, YY, ZZ, levels=8, colors="0.78", linewidths=0.7, zorder=0)


# --- hand-set per-generation state so the adaptation reads clearly ---
# means walking down the valley toward the optimum
means = [np.array([3.0, 2.6]), np.array([1.95, 1.55]), np.array([1.05, 0.8])]
# covariances: isotropic -> tilted -> aligned with the 30-degree valley
covs = [
    np.eye(2) * 0.5,
    rot(20) @ np.diag([0.85, 0.30]) @ rot(20).T,
    rot(30) @ np.diag([1.15, 0.20]) @ rot(30).T,
]

lam, nsel = 40, 14

fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), constrained_layout=True)
col_titles = ["Sampling", "Selection", "Updated distribution"]

for gen in range(2):
    mu, C = means[gen], covs[gen]
    new_mu, new_C = means[gen + 1], covs[gen + 1]

    samp = np.random.multivariate_normal(mu, C, size=lam)
    order = np.argsort(f(samp))
    sel, rej = samp[order[:nsel]], samp[order[nsel:]]

    # 1) Sampling
    ax = axes[gen, 0]
    contours(ax)
    ax.add_patch(cov_ell(C, mu, 2, fill=False, ec="C0", lw=1.5, ls="--", zorder=4))
    ax.scatter(samp[:, 0], samp[:, 1], s=15, c="0.35", zorder=3)
    ax.plot(*mu, "+", c="C3", ms=13, mew=2.2, zorder=5)

    # 2) Selection
    ax = axes[gen, 1]
    contours(ax)
    for s in sel:
        ax.plot([mu[0], s[0]], [mu[1], s[1]], c="0.6", lw=0.6, zorder=2)
    ax.scatter(rej[:, 0], rej[:, 1], s=13, facecolors="none",
               edgecolors="0.6", zorder=3)
    ax.scatter(sel[:, 0], sel[:, 1], s=20, c="C0", zorder=4)
    ax.plot(*mu, "+", c="C3", ms=13, mew=2.2, zorder=5)

    # 3) Updated distribution (new solid vs previous dashed)
    ax = axes[gen, 2]
    contours(ax)
    ax.add_patch(cov_ell(C, mu, 2, fill=False, ec="0.6", lw=1.2, ls="--", zorder=3))
    ax.add_patch(cov_ell(new_C, new_mu, 2, fill=False, ec="C0", lw=1.9, zorder=4))
    ax.plot(*mu, "+", c="0.6", ms=11, mew=1.8, zorder=4)
    ax.plot(*new_mu, "+", c="C3", ms=13, mew=2.2, zorder=5)

for j, t in enumerate(col_titles):
    axes[0, j].set_title(t, fontsize=13)
for gen in range(2):
    axes[gen, 0].set_ylabel(f"Generation {gen + 1}", fontsize=12)
for ax in axes.ravel():
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(0, 0, "*", c="black", ms=12, zorder=1)  # optimum

fig.savefig("Figures/background/cmaes.png", dpi=200, bbox_inches="tight")
print("wrote Figures/background/cmaes.png")
