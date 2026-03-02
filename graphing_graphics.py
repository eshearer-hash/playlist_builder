from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd

def _generate_distinct_colors(n: int, seed: int = 42) -> list[str]:
    """
    Generate *n* visually distinct colours by spacing hues evenly around
    the HSL wheel with a random offset, then jittering saturation/lightness.
    """
    rng = np.random.RandomState(seed)
    offset = rng.uniform(0, 1)
    colors = []
    for i in range(n):
        hue = (i / n + offset) % 1.0
        sat = rng.uniform(0.55, 0.85)
        lit = rng.uniform(0.45, 0.62)
        r, g, b = colorsys.hls_to_rgb(hue, lit, sat)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def _draw_cluster_hull(ax: plt.Axes, points: np.ndarray, color: str,
                       alpha_fill: float = 0.08, alpha_edge: float = 0.45):
    """Draw a convex-hull boundary around a set of 2-D points."""
    if len(points) < 3:
        return
    try:
        hull = ConvexHull(points)
    except Exception:
        return
    vertices = np.append(hull.vertices, hull.vertices[0])  # close the polygon
    ax.fill(
        points[vertices, 0], points[vertices, 1],
        color=color, alpha=alpha_fill,
    )
    ax.plot(
        points[vertices, 0], points[vertices, 1],
        color=color, alpha=alpha_edge, linewidth=1.8, linestyle="--",
    )


def plot_clusters_2d(
    df_norm: pd.DataFrame,
    clusters: pd.Series,
    km: KMeans,
    title: str,
    ax: plt.Axes,
    color_seed: int = 42,
):
    """
    Reduce *df_norm* to 2-D with PCA, scatter-plot coloured by cluster,
    draw convex-hull borders, and mark centroids with ×.
    """
    filled = df_norm.fillna(df_norm.median())
    pca = PCA(n_components=2)
    coords = pca.fit_transform(filled.values)
    var = pca.explained_variance_ratio_

    centres_2d = pca.transform(km.cluster_centers_)
    colors = _generate_distinct_colors(km.n_clusters, seed=color_seed)

    # ── per-cluster: hull + scatter ──────────────────────────────
    for k in range(km.n_clusters):
        mask = clusters.values == k
        pts = coords[mask]
        c = colors[k]

        # convex-hull boundary
        _draw_cluster_hull(ax, pts, color=c)

        # data points
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=c, label=f"Cluster {k}",
            s=50, alpha=0.80,
            edgecolors="white", linewidths=0.5,
            zorder=3,
        )

    # ── centroids ────────────────────────────────────────────────
    ax.scatter(
        centres_2d[:, 0], centres_2d[:, 1],
        c="black", marker="X", s=200,
        edgecolors="white", linewidths=1.2,
        zorder=6, label="Centroid",
    )

    # ── axes polish ──────────────────────────────────────────────
    ax.set_xlabel(f"PC 1  ({var[0]*100:.1f}% var)", fontsize=14)
    ax.set_ylabel(f"PC 2  ({var[1]*100:.1f}% var)", fontsize=14)
    ax.set_title(title, fontsize=20, fontweight="bold", pad=14)
    ax.tick_params(labelsize=12)
    ax.legend(
        fontsize=11, loc="best", framealpha=0.9,
        edgecolor="#cccccc", fancybox=True,
    )
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
