"""
Plot heatmaps of alpha/beta grid search results.

Usage:
  python plot_alpha_beta_heatmap.py --json_path path/to/alpha_beta_grid_search_lut.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_results(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_grid(data, metric_key: str, stage_filter=None):
    """
    从 result list 构造一个 alpha × beta 的 2D 网格，用于画热力图。
    若 stage_filter="fine"，则只用细搜索结果；若 None，则 coarse+fine 全用。
    """
    entries = [
        r for r in data
        if (stage_filter is None or r.get("stage") == stage_filter)
    ]

    alphas = sorted(set(r["alpha"] for r in entries))
    betas = sorted(set(r["beta"] for r in entries))

    A = len(alphas)
    B = len(betas)
    grid = np.full((A, B), np.nan, dtype=float)

    alpha_to_idx = {a: i for i, a in enumerate(alphas)}
    beta_to_idx = {b: j for j, b in enumerate(betas)}

    for r in entries:
        i = alpha_to_idx[r["alpha"]]
        j = beta_to_idx[r["beta"]]
        grid[i, j] = r[metric_key]

    return np.array(alphas), np.array(betas), grid


def plot_heatmap(alphas, betas, grid, title, cmap="viridis", invert_colormap=False):
    fig, ax = plt.subplots(figsize=(6, 5))

    data = grid.copy()
    if invert_colormap:
        cmap = cmap + "_r"  # 小 trick：小值颜色深时用

    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[
            betas.min() - 0.5 * (betas[1] - betas[0]) if len(betas) > 1 else betas[0] - 0.5,
            betas.max() + 0.5 * (betas[1] - betas[0]) if len(betas) > 1 else betas[0] + 0.5,
            alphas.min() - 0.5 * (alphas[1] - alphas[0]) if len(alphas) > 1 else alphas[0] - 0.5,
            alphas.max() + 0.5 * (alphas[1] - alphas[0]) if len(alphas) > 1 else alphas[0] + 0.5,
        ],
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(title)

    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    ax.set_title(title)

    # 在格子中心标出数值
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(
                    b,
                    a,
                    f"{val:.4f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

    plt.tight_layout()
    return fig, ax


def main(json_path):
    json_path = Path(json_path)
    data = load_results(json_path)
    # 三个指标画三张图
    for metric_key, title in [
        ("rmse_all", "RMSE_all (LUT val)"),
        ("sam", "SAM (LUT val)"),
        ("rmse_band", "Band RMSE (LUT val)"),
    ]:
        alphas, betas, grid = build_grid(data, metric_key)
        fig, ax = plot_heatmap(alphas, betas, grid, title)
        # 自动保存
        out_path = json_path.with_name(json_path.stem + f"_{metric_key}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    path='search/alpha_beta_grid_search_lut_cpu.json'
    main(path)
