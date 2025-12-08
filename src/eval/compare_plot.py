"""
Compare spectra: LUT-pretrained vs finetuned vs ASD ground truth.

Usage example:
  python -m m2h_swir.tools.compare_spectra_pretrain_finetuned \
      --pretrained m2h_swir_lut.pt \
      --finetuned m2h_swir_finetuned.pt \
      --num_samples 5 \
      --save_dir figures/compare_spectra

说明：
- 从 REAL_DIR 读取 X_uav_ms, Y_asd_hs, wavelength
- 用 WL_MIN/WL_MAX/WL_STEP 把 ASD 光谱插到完整栅格
- 使用同一批样本画出：ASD 实测 vs 预训练 vs 微调 后的光谱对比
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# === 路径与导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/m2h_swir/tools
parent_dir = os.path.dirname(current_dir)  # .../src/m2h_swir
sys.path.insert(0, parent_dir)
from config import (  # type: ignore
    REAL_DIR,
    MODEL_DIR,
    WL_MIN,
    WL_MAX,
    WL_STEP,
    N_SENSITIVE_WAVELENGTHS,  # 如果没有可删掉
)
from model.models import build_m2h_swir_model  # type: ignore


def build_full_grid_labels(Y_compact, wavelength_real):
    """
    把 ASD 的紧凑光谱 (N,1670) 扩展到 LUT 同样的 400–2500 nm 栅格 (N,2101)。
    """
    num_wl = int((WL_MAX - WL_MIN) / WL_STEP) + 1
    wl_full = WL_MIN + np.arange(num_wl) * WL_STEP

    valid_mask = np.isin(wl_full, wavelength_real)
    if valid_mask.sum() != wavelength_real.shape[0]:
        raise ValueError(
            f"波段对齐错误: valid_mask True 数={valid_mask.sum()} "
            f"!= ASD 波长数={wavelength_real.shape[0]}"
        )

    Y_full = np.zeros((Y_compact.shape[0], num_wl), dtype=Y_compact.dtype)
    Y_full[:, valid_mask] = Y_compact
    return Y_full, wl_full, valid_mask


def load_model_from_ckpt(ckpt_path: Path, input_dim: int, output_dim: int, device: torch.device):
    """
    用与训练时相同的结构构建模型并加载 state_dict。
    如果你训练时改过模型超参，这里要同步改。
    """
    model, _, _ = build_m2h_swir_model(
        input_shape=(input_dim,),
        output_dim=output_dim,
    )
    model.to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="m2h_data_driven_new_head.pt",
                        help="LUT 预训练模型 (.pt)，位于 MODEL_DIR 下")
    parser.add_argument("--finetuned", type=str, default="m2h_swir_lut_new_head.pt",
                        help="在 REAL 上微调后的模型 (.pt)，位于 MODEL_DIR 下")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="随机抽取多少个样本绘图")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="显式指定样本索引（覆盖 num_samples）")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="若指定则把图保存到该目录，否则直接显示")
    parsed = parser.parse_args(args=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 1) 读取 REAL 数据 ===
    X_real = np.load(REAL_DIR / "X_uav_ms.npy")            # (N, 10)
    Y_real_compact = np.load(REAL_DIR / "Y_asd_hs.npy")    # (N, 1670)
    wavelength_real = np.load(REAL_DIR / "wavelength.npy") # (1670,)

    N, input_dim = X_real.shape
    print(f"Loaded REAL data: {N} samples, {input_dim} MS bands")

    # === 2) 把 ASD 光谱扩展到完整 LUT 栅格 ===
    Y_real_full, wl_full, valid_mask = build_full_grid_labels(Y_real_compact, wavelength_real)
    output_dim = Y_real_full.shape[1]
    print(f"Full spectral grid: {output_dim} bands from {wl_full[0]} to {wl_full[-1]} nm")

    # === 3) 构建并加载两个模型 ===
    pretrained_path = MODEL_DIR / parsed.pretrained
    finetuned_path = MODEL_DIR / parsed.finetuned
    assert pretrained_path.exists(), f"预训练模型不存在: {pretrained_path}"
    assert finetuned_path.exists(), f"微调模型不存在: {finetuned_path}"

    print(f"Loading pretrained model from {pretrained_path}")
    model_pre = load_model_from_ckpt(pretrained_path, input_dim, output_dim, device)

    print(f"Loading finetuned model from {finetuned_path}")
    model_ft = load_model_from_ckpt(finetuned_path, input_dim, output_dim, device)

    # === 4) 选择要绘制的样本 ===
    if parsed.indices is not None and len(parsed.indices) > 0:
        indices = [i for i in parsed.indices if 0 <= i < N]
    else:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(N, size=min(parsed.num_samples, N), replace=False).tolist()

    print(f"Plotting {len(indices)} samples, indices = {indices}")

    # === 5) 推理并绘图 ===
    if parsed.save_dir is not None:
        save_dir = Path(parsed.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    # 转成 torch.Tensor 备用
    X_real_tensor = torch.from_numpy(X_real).float().to(device)

    for idx in indices:
        x = X_real_tensor[idx:idx+1]               # (1, input_dim)
        y_true_full = Y_real_full[idx]             # (output_dim,)

        with torch.no_grad():
            y_pre = model_pre(x).cpu().numpy().reshape(-1)  # (output_dim,)
            y_ft = model_ft(x).cpu().numpy().reshape(-1)    # (output_dim,)

        # === 画图 ===
        plt.figure(figsize=(10, 5))
        plt.plot(wl_full, y_true_full, label="ASD ground truth", linewidth=2)
        plt.plot(wl_full, y_pre, label="Pretrained (LUT)", linestyle="--")
        plt.plot(wl_full, y_ft, label="Finetuned (REAL)", linestyle="-.")

        # 标出氮敏感波段（如果 config 里有 N_SENSITIVE_WAVELENGTHS）
        try:
            for w in N_SENSITIVE_WAVELENGTHS:
                if wl_full[0] <= w <= wl_full[-1]:
                    plt.axvline(x=w, linestyle=":", alpha=0.5)
        except NameError:
            # 没定义就跳过
            pass

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title(f"Sample #{idx}: ASD vs Pretrained vs Finetuned")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir is not None:
            out_path = save_dir / f"spectrum_compare_sample_{idx}.png"
            plt.savefig(out_path, dpi=200)
            print(f"Saved figure: {out_path}")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    main()
