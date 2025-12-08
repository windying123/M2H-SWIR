import argparse
import os
import sys
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ==== 路径设置 ====
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 导入项目配置与模型构建器
from config import REAL_DIR, MODEL_DIR, WL_MIN, WL_MAX, WL_STEP  # type: ignore
from model.models import build_m2h_swir_model, build_simple_cnn  # type: ignore


# ============================================================
# 1. 构造完整波长轴 (full-grid)
# ============================================================
def build_full_grid_wavelengths():
    output_dim = int((WL_MAX - WL_MIN) / WL_STEP) + 1
    wavelengths_full = WL_MIN + np.arange(output_dim) * WL_STEP
    return wavelengths_full.astype(int)


# ============================================================
# 2. 加载模型
# ============================================================
def load_model_from_checkpoint(model_path, input_dim, output_dim, model_type, device):
    if model_type == "m2h":
        model, _, _ = build_m2h_swir_model(
            input_shape=(input_dim,),
            output_dim=output_dim,
        )
    elif model_type == "simple_cnn":
        model, _, _ = build_simple_cnn(
            input_shape=(input_dim,),
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 加载权重
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded model: {model_path}")
    return model


# ============================================================
# 3. 导出按生育时期的预测光谱 CSV
# ============================================================
def export_predictions_by_stage(model, X_real, stages, wavelengths_full, output_dir, device):
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_stages = np.unique(stages)

    print("\n=== Exporting predictions by stage ===")
    print(f"Output directory: {output_dir}")

    for stage in unique_stages:
        mask = stages == stage
        X_stage = X_real[mask]
        n_samples = X_stage.shape[0]

        if n_samples == 0:
            continue

        print(f"Stage: {stage} --> {n_samples} samples")

        with torch.no_grad():
            X_tensor = torch.from_numpy(X_stage).float().to(device)
            Y_pred_tensor = model(X_tensor)
            Y_pred = Y_pred_tensor.cpu().numpy()

        csv_path = output_dir / f"pred_{stage}.csv"

        # 写入 CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            header = ["sample_id"] + wavelengths_full.tolist()
            writer.writerow(header)

            for i, row in enumerate(Y_pred):
                writer.writerow([f"sample_{i+1:03d}"] + row.tolist())

        print(f"Saved: {csv_path}")

    print("\n=== CSV export completed ===\n")


# ============================================================
# 4. CLI 入口
# ============================================================
def main(args=None):
    parser = argparse.ArgumentParser(description="Export model spectral predictions by stage (CSV).")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of checkpoint file in MODEL_DIR (e.g., m2h_data_driven_new_head.pt)",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["m2h", "simple_cnn"],
        help="Architecture of the model",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="pred_csv",
        help="Folder name inside MODEL_DIR for saving CSV files",
    )

    parsed = parser.parse_args(args=args)

    # ==== 加载 REAL 数据 ====
    X_real = np.load(REAL_DIR / "X_uav_ms.npy")
    stages = np.load(REAL_DIR / "stages.npy")
    wavelength_real = np.load(REAL_DIR / "wavelength.npy")

    # full-grid 构造
    wavelengths_full = build_full_grid_wavelengths()
    output_dim = len(wavelengths_full)
    input_dim = X_real.shape[1]

    # ==== 加载模型 ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODEL_DIR / parsed.model_name
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = load_model_from_checkpoint(
        model_path=model_path,
        input_dim=input_dim,
        output_dim=output_dim,
        model_type=parsed.model_type,
        device=device,
    )

    # ==== 导出 CSV ====
    save_dir = MODEL_DIR / parsed.save_dir / model_path.stem
    export_predictions_by_stage(model, X_real, stages, wavelengths_full, save_dir, device)


if __name__ == "__main__":
    # LUT + finetuned on real
    print("\n=== m2h_swir_finetuned_new_head (M2H finetuned) ===")
    main([
        "--model_name", "m2h_swir_finetuned_new_head.pt",
        "--model_type", "m2h",
    ])
