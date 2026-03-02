import ast
import colorsys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ── Shared constants ────────────────────────────────────────────────

W, H, DPI = 1920 / 150, 1080 / 150, 150

PATH_CMAP = LinearSegmentedColormap.from_list(
    "path", ["#2c3e50", "#3498db", "#1abc9c", "#f39c12", "#e74c3c"]
)

_STROKE = [pe.withStroke(linewidth=2.5, foreground="white")]
_NUM_STROKE = [pe.withStroke(linewidth=1.5, foreground="black")]


# ── Helpers ─────────────────────────────────────────────────────────

def primary_artist(raw) -> str:
    """Parse ``"['A', 'B']"`` -> ``"A"`` (first / primary artist)."""
    try:
        return ast.literal_eval(raw)[0]
    except Exception:
        return str(raw)


def _generate_distinct_colors(n: int, seed: int = 42) -> list[str]:
    """Generate *n* visually distinct colours spaced around the HSL wheel."""
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
    vertices = np.append(hull.vertices, hull.vertices[0])
    ax.fill(points[vertices, 0], points[vertices, 1],
            color=color, alpha=alpha_fill)
    ax.plot(points[vertices, 0], points[vertices, 1],
            color=color, alpha=alpha_edge, linewidth=1.8, linestyle="--")


# ── Building blocks ─────────────────────────────────────────────────

def _pca_coords(features: pd.DataFrame) -> tuple[pd.DataFrame, PCA]:
    """Fit PCA(2) on *features* and return (coords_df, pca_object)."""
    pca = PCA(n_components=2)
    coords = pd.DataFrame(
        pca.fit_transform(features.fillna(features.median())),
        index=features.index, columns=["PC1", "PC2"],
    )
    return coords, pca


def _draw_path(ax: plt.Axes, coords: pd.DataFrame, playlist: list,
               cmap=PATH_CMAP):
    """Draw a gradient LineCollection along *playlist* order."""
    pc = coords.loc[playlist]
    n = len(playlist)
    segments = np.array(
        [[pc.iloc[i].values, pc.iloc[i + 1].values] for i in range(n - 1)]
    )
    lc = LineCollection(segments, cmap=cmap, norm=Normalize(0, n - 2),
                        linewidths=3, alpha=0.85, zorder=6, capstyle="round")
    lc.set_array(np.arange(n - 1))
    ax.add_collection(lc)


def _label_offset_centroid(x, y, centroid, radius=18):
    """Push label away from *centroid*."""
    dx, dy = x - centroid[0], y - centroid[1]
    mag = max(np.hypot(dx, dy), 1e-6)
    return radius * dx / mag, radius * dy / mag


def _label_offset_repulsion(x, y, all_xy, idx, radius=22):
    """Push label away from all other playlist nodes + their centroid."""
    n = len(all_xy)
    dx, dy = 0.0, 0.0
    for j in range(n):
        if j == idx:
            continue
        rx, ry = x - all_xy[j, 0], y - all_xy[j, 1]
        d = max(np.hypot(rx, ry), 0.5)
        dx += rx / (d ** 2)
        dy += ry / (d ** 2)
    cx, cy = all_xy.mean(axis=0)
    dx += (x - cx) * 0.3
    dy += (y - cy) * 0.3
    mag = max(np.hypot(dx, dy), 1e-6)
    return radius * dx / mag, radius * dy / mag


def _polish_axes(ax: plt.Axes, pca: PCA, title: str, *,
                 legend_ncol: int = 1):
    """Apply shared scientific axis styling."""
    ax.set_xlabel(f"PC 1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
                  fontsize=13)
    ax.set_ylabel(f"PC 2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)",
                  fontsize=13)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="#ccc", fancybox=True, ncol=legend_ncol)
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# ── Top-level plot functions ────────────────────────────────────────

def plot_smooth_playlist(
    merged_features: pd.DataFrame,
    playlist: list,
    artist_series: pd.Series,
    song_name_series: pd.Series,
    artist_colors: dict[str, str],
    title: str = "Smooth-Transition Playlist Path (PCA)",
    save_path: str | None = "smooth_playlist_path.png",
):
    """PCA scatter coloured by artist with a gradient playlist path."""
    coords, pca = _pca_coords(merged_features)
    parsed = artist_series.map(primary_artist)
    default_color = "#b0b8c0"

    fig, ax = plt.subplots(figsize=(W, H))

    # Background library per artist
    bg_idx = coords.index.difference(playlist)
    for artist, color in artist_colors.items():
        mask = parsed.loc[bg_idx] == artist
        if mask.any():
            ax.scatter(coords.loc[bg_idx[mask], "PC1"],
                       coords.loc[bg_idx[mask], "PC2"],
                       c=color, s=25, alpha=0.35, edgecolors="none",
                       zorder=1, label=artist)
    other_mask = ~parsed.loc[bg_idx].isin(artist_colors)
    ax.scatter(coords.loc[bg_idx[other_mask], "PC1"],
               coords.loc[bg_idx[other_mask], "PC2"],
               c=default_color, s=20, alpha=0.2, edgecolors="none",
               zorder=1, label="Other")

    # Path
    _draw_path(ax, coords, playlist)

    # Nodes + labels (centroid-push)
    pc = coords.loc[playlist]
    n = len(playlist)
    centroid = pc.mean().values

    for i, sid in enumerate(playlist):
        x, y = coords.loc[sid, "PC1"], coords.loc[sid, "PC2"]
        color = PATH_CMAP(i / max(n - 1, 1))
        ax.scatter(x, y, s=220, c=[color], edgecolors="white",
                   linewidths=1.3, zorder=6)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=7,
                path_effects=_NUM_STROKE)

        ox, oy = _label_offset_centroid(x, y, centroid)
        ha_align = "left" if ox > 0 else "right"
        artist = primary_artist(artist_series[sid])
        name = song_name_series[sid][:22]
        ax.annotate(f"{artist} — {name}", (x, y),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=7, ha=ha_align, va="center", color="#2c3e50",
                    path_effects=_STROKE,
                    arrowprops=dict(arrowstyle="-", color="#bdc3c7", lw=0.6),
                    zorder=8)

    _polish_axes(ax, pca, title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_thematic_playlist(
    merged_features: pd.DataFrame,
    playlist: list,
    clusters: pd.Series,
    km: KMeans,
    artist_series: pd.Series,
    song_name_series: pd.Series,
    title: str = "Thematic Structure Playlist (k={k} Clusters)",
    save_path: str | None = "thematic_playlist_path.png",
    color_seed: int = 42,
):
    """PCA scatter with cluster hulls, centroids, and a gradient playlist path."""
    filled = merged_features.fillna(merged_features.median())
    pca = PCA(n_components=2)
    coords = pd.DataFrame(
        pca.fit_transform(filled.values),
        index=merged_features.index, columns=["PC1", "PC2"],
    )

    k = km.n_clusters
    cluster_colors = _generate_distinct_colors(k, seed=color_seed)

    fig, ax = plt.subplots(figsize=(W, H))

    # Cluster hulls + scatter
    for cid in range(k):
        mask = clusters == cid
        members = coords[mask]
        c = cluster_colors[cid]
        _draw_cluster_hull(ax, members.values, color=c,
                           alpha_fill=0.08, alpha_edge=0.15)
        ax.scatter(members["PC1"], members["PC2"], c=c, s=35, alpha=0.45,
                   edgecolors="white", linewidths=0.3, zorder=2,
                   label=f"Cluster {cid} ({mask.sum()} songs)")

    # Centroids
    centres_2d = pca.transform(km.cluster_centers_)
    ax.scatter(centres_2d[:, 0], centres_2d[:, 1], c="black", marker="X",
               s=140, edgecolors="white", linewidths=1, zorder=4,
               label="Centroid")

    # Path
    _draw_path(ax, coords, playlist)

    # Nodes + labels (repulsion-based)
    pc = coords.loc[playlist]
    n = len(playlist)
    playlist_xy = pc.values

    for i, sid in enumerate(playlist):
        x, y = coords.loc[sid, "PC1"], coords.loc[sid, "PC2"]
        color = PATH_CMAP(i / max(n - 1, 1))
        cid = clusters[sid]

        # Cluster-colored ring + path-colored fill
        ax.scatter(x, y, s=280, c=cluster_colors[cid],
                   edgecolors="white", linewidths=2, zorder=8)
        ax.scatter(x, y, s=120, c=[color], edgecolors="none", zorder=9)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=10,
                path_effects=_NUM_STROKE)

        ox, oy = _label_offset_repulsion(x, y, playlist_xy, i)
        ha_align = "left" if ox > 0 else "right"
        artist = primary_artist(artist_series[sid])
        name = song_name_series[sid][:22]
        ax.annotate(f"{artist} — {name}", (x, y),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=7, ha=ha_align, va="center", color="#2c3e50",
                    path_effects=_STROKE,
                    arrowprops=dict(arrowstyle="-", color="#bdc3c7", lw=0.6),
                    zorder=11)

    _polish_axes(ax, pca, title.format(k=k), legend_ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_clusters_2d(
    df_norm: pd.DataFrame,
    clusters: pd.Series,
    km: KMeans,
    title: str,
    ax: plt.Axes,
    color_seed: int = 42,
):
    """Reduce *df_norm* to 2-D with PCA, scatter by cluster, draw hulls."""
    filled = df_norm.fillna(df_norm.median())
    pca = PCA(n_components=2)
    coords = pca.fit_transform(filled.values)
    var = pca.explained_variance_ratio_

    centres_2d = pca.transform(km.cluster_centers_)
    colors = _generate_distinct_colors(km.n_clusters, seed=color_seed)

    for k_id in range(km.n_clusters):
        mask = clusters.values == k_id
        pts = coords[mask]
        c = colors[k_id]
        _draw_cluster_hull(ax, pts, color=c)
        ax.scatter(pts[:, 0], pts[:, 1], c=c, label=f"Cluster {k_id}",
                   s=50, alpha=0.80, edgecolors="white", linewidths=0.5,
                   zorder=3)

    ax.scatter(centres_2d[:, 0], centres_2d[:, 1], c="black", marker="X",
               s=200, edgecolors="white", linewidths=1.2, zorder=6,
               label="Centroid")

    ax.set_xlabel(f"PC 1  ({var[0]*100:.1f}% var)", fontsize=14)
    ax.set_ylabel(f"PC 2  ({var[1]*100:.1f}% var)", fontsize=14)
    ax.set_title(title, fontsize=20, fontweight="bold", pad=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="best", framealpha=0.9,
              edgecolor="#cccccc", fancybox=True)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
