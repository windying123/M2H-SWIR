"""Evaluate trained PyTorch model on real UAV–ASD pairs (global or per growth stage),
including nitrogen-sensitive vegetation index errors.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# === 路径与导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))   # ./src/m2h_swir/eval 或类似
parent_dir = os.path.dirname(current_dir)                  # ./src/m2h_swir
sys.path.insert(0, parent_dir)

from config import REAL_DIR, MODEL_DIR, WL_MIN, WL_MAX, WL_STEP  # type: ignore
from model.losses import (
    spectral_rmse,
    spectral_angle_mapper,
    band_rmse,
)  # type: ignore
from model.models import build_m2h_swir_model, build_simple_cnn
# ----------------------------------------------------------------------
# 1. 通用数据准备：构造 full 维度标签 + valid_mask
# ----------------------------------------------------------------------
def _prepare_real_full_and_mask():
    """
    统一的准备过程（PyTorch 版）：
    - 读取紧凑版 ASD 标签 Y_real_compact (N, n_valid)
    - 用 WL_MIN / WL_MAX / WL_STEP 构造 LUT 波长轴
    - 用 wavelength.npy 生成 valid_mask (哪些 LUT 波段在 ASD 里存在)
    - 构造 full 维度标签 Y_real_full (N, output_dim)
    返回：
      X_real, Y_real_full, wavelength_real, valid_mask, output_dim, wavelengths_full
    """
    X_real = np.load(REAL_DIR / "X_uav_ms.npy")
    Y_real_compact = np.load(REAL_DIR / "Y_asd_hs.npy")      # (N, n_valid)
    wavelength_real = np.load(REAL_DIR / "wavelength.npy")   # (n_valid,)

    # 用 WL_MIN / WL_MAX / WL_STEP 构造完整光谱轴（和 LUT & 训练时保持一致）
    # 例如 400–2500 nm, step=1 → 2101 维
    output_dim = int((WL_MAX - WL_MIN) / WL_STEP) + 1
    wavelengths_full = WL_MIN + np.arange(output_dim) * WL_STEP

    # 哪些 LUT 波段在 ASD 里有对应值
    valid_mask = np.isin(wavelengths_full, wavelength_real)
    if valid_mask.sum() != wavelength_real.shape[0]:
        raise ValueError(
            f"波段数不一致：valid_mask 中 True={valid_mask.sum()}，"
            f"wavelength_real 长度={wavelength_real.shape[0]}"
        )

    # 构造 full 维度标签：只有 valid_mask=True 的波段被填入 ASD 值
    Y_real_full = np.zeros((Y_real_compact.shape[0], output_dim), dtype=Y_real_compact.dtype)
    Y_real_full[:, valid_mask] = Y_real_compact

    return X_real, Y_real_full, wavelength_real, valid_mask, output_dim, wavelengths_full


# ----------------------------------------------------------------------
# 2. 基于 valid_mask 的 masked RMSE / SAM（PyTorch）
# ----------------------------------------------------------------------
def _build_masked_metrics_torch(valid_mask: np.ndarray, device: torch.device):
    """
    基于 valid_mask 构造 masked RMSE / SAM，用逐元素 mask。
    返回：
      masked_spectral_rmse_fn, masked_sam_fn
    """
    valid_mask_torch = torch.from_numpy(valid_mask.astype("float32")).to(device)
    valid_mask_torch = valid_mask_torch.view(1, -1)  # (1, output_dim)

    def masked_spectral_rmse_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true_sel = y_true * valid_mask_torch
        y_pred_sel = y_pred * valid_mask_torch
        return spectral_rmse(y_true_sel, y_pred_sel)

    def masked_sam_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true_sel = y_true * valid_mask_torch
        y_pred_sel = y_pred * valid_mask_torch
        return spectral_angle_mapper(y_true_sel, y_pred_sel)

    return masked_spectral_rmse_fn, masked_sam_fn


# ----------------------------------------------------------------------
# 3. 基于光谱计算氮敏感植被指数
# ----------------------------------------------------------------------
VI_WAVELENGTHS = {
    "NDVI":      (850, 670),
    "NDRE":      (850, 730),
    "GNDVI":     (850, 550),   # 采用 Green NDVI 定义
    "OSAVI1510": (850, 1510),
    "N870_1450": (870, 1450),
    "N850_1510": (850, 1510),
}

def _build_band_index_map(wavelengths_full: np.ndarray):
    """
    生成 {波长: 在 full 向量中的索引} 的字典。
    """
    return {int(w): idx for idx, w in enumerate(wavelengths_full.astype(int))}


def _compute_vi_from_spectra(Y: np.ndarray,
                             wavelengths_full: np.ndarray) -> dict:
    """
    给定 full 波段光谱 Y (N, D) 和对应波长 wavelengths_full (D,),
    计算指定的 VI，返回:
      {"NDVI": (N,), "NDRE": (N,), ...}
    """
    band_idx_map = _build_band_index_map(wavelengths_full)
    eps = 1e-8

    def get_band(R, wl):
        if wl not in band_idx_map:
            raise ValueError(f"目标波长 {wl} nm 不在 wavelengths_full 里，无法计算对应 VI。")
        return R[..., band_idx_map[wl]]

    VIs = {}

    # NDVI
    r850 = get_band(Y, 850)
    r670 = get_band(Y, 670)
    VIs["NDVI"] = (r850 - r670) / (r850 + r670 + eps)

    # NDRE
    r730 = get_band(Y, 730)
    VIs["NDRE"] = (r850 - r730) / (r850 + r730 + eps)

    # GNDVI (850, 550)
    r550 = get_band(Y, 550)
    VIs["GNDVI"] = (r850 - r550) / (r850 + r550 + eps)

    # OSAVI1510 (1+0.16) * (R850 - R1510) / (R850 + R1510 + 0.16)
    r1510 = get_band(Y, 1510)
    VIs["OSAVI1510"] = (1.0 + 0.16) * (r850 - r1510) / (r850 + r1510 + 0.16 + eps)

    # N1645_1715
    r1645 = get_band(Y, 1645)
    r1715 = get_band(Y, 1715)
    VIs["N1645_1715"] = (r1645 - r1715) / (r1645 + r1715 + eps)

    # N850_1510
    VIs["N850_1510"] = (r850 - r1510) / (r850 + r1510 + eps)

    return VIs


def _vi_error_metrics(VI_true: dict, VI_pred: dict) -> dict:
    """
    对每一个 VI，计算 RMSE 和 R^2。
    输入:
      VI_true: {"NDVI": (N,), ...}
      VI_pred: 同结构
    返回:
      {"NDVI": {"rmse": ..., "r2": ...}, ...}
    """
    vi_metrics = {}
    for name in VI_true.keys():
        y = VI_true[name].reshape(-1)
        y_hat = VI_pred[name].reshape(-1)

        diff = y_hat - y
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        # R^2
        ss_res = float(np.sum(diff ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot

        vi_metrics[name] = {
            "rmse": rmse,
            "r2": r2,
        }
    return vi_metrics


# ----------------------------------------------------------------------
# 4. 构建并加载 PyTorch 模型（支持 m2h / simple_cnn）
# ----------------------------------------------------------------------
def _load_pytorch_model(model_path: Path,
                        input_dim: int,
                        output_dim: int,
                        device: torch.device,
                        model_type: str = "m2h") -> nn.Module:
    """
    使用 build_m2h_swir_model 或 build_simple_cnn 重建架构，然后加载 state_dict。
    model_type:
      - "m2h"        -> M2H_SWIR_Model
      - "simple_cnn" -> SimpleCNN baseline
    """
    if model_type == "m2h":
        model, _, _ = build_m2h_swir_model(
            input_shape=(input_dim,),
            output_dim=output_dim,
        )
    elif model_type == "simple_cnn":
        model, _, _ =build_simple_cnn(
             input_shape=(input_dim,),
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 安全加载：优先尝试 weights_only=True（新版本 PyTorch），不行就回退
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------
# 5. 全部样本整体评估（global）
# ----------------------------------------------------------------------
def evaluate_model_on_real_global(model_path: Path, model_type: str = "m2h"):
    """在全部真实 UAV–ASD 配对上评估（不分时期），包含 VI 指标。"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_real, Y_real_full, wavelength_real, valid_mask, output_dim, wavelengths_full = (
        _prepare_real_full_and_mask()
    )

    input_dim = X_real.shape[1]
    model = _load_pytorch_model(model_path, input_dim, output_dim, device, model_type=model_type)

    # 预测
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_real).float().to(device)
        Y_pred_tensor = model(X_tensor)  # (N, output_dim)

    # 转成 tensor 做指标计算
    y_true = torch.from_numpy(Y_real_full).float().to(device)
    y_pred = Y_pred_tensor

    masked_rmse, masked_sam = _build_masked_metrics_torch(valid_mask, device)

    # 全谱重建指标（仅在 ASD 有效波段上）
    rmse_all = float(masked_rmse(y_true, y_pred).item())
    sam = float(masked_sam(y_true, y_pred).item())
    rmse_band_val = float(band_rmse(y_true, y_pred).item())

    # 氮敏感 VI 指标（这里用 numpy 计算）
    Y_pred = Y_pred_tensor.cpu().numpy()
    VI_true = _compute_vi_from_spectra(Y_real_full, wavelengths_full)
    VI_pred = _compute_vi_from_spectra(Y_pred, wavelengths_full)
    vi_metrics = _vi_error_metrics(VI_true, VI_pred)

    metrics = {
        "scope": "global",
        "model": model_path.name,
        "n_samples": int(X_real.shape[0]),
        "rmse_all": rmse_all,
        "sam": sam,
        "rmse_band": rmse_band_val,
        "vi_metrics": vi_metrics,  # NDVI / NDRE / GNDVI / OSAVI1510 / N870_1450 / N850_1510 ...
    }
    return metrics


# ----------------------------------------------------------------------
# 6. 按生育时期评估（by_stage）
# ----------------------------------------------------------------------
def evaluate_model_on_real_by_stage(model_path: Path, model_type: str = "m2h"):
    """
    按生育时期评估：
    要求 REAL_DIR 下有 stages.npy，shape=(N_real,)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_real, Y_real_full, wavelength_real, valid_mask, output_dim, wavelengths_full = (
        _prepare_real_full_and_mask()
    )
    stage_labels = np.load(REAL_DIR / "stages.npy")  # 比如 ['Tillering', 'Jointing', ...]

    if stage_labels.shape[0] != X_real.shape[0]:
        raise ValueError(
            f"stage_labels 长度={stage_labels.shape[0]} 与 X_real 样本数={X_real.shape[0]} 不一致"
        )

    input_dim = X_real.shape[1]
    model = _load_pytorch_model(model_path, input_dim, output_dim, device, model_type=model_type)

    masked_rmse, masked_sam = _build_masked_metrics_torch(valid_mask, device)

    stages = np.unique(stage_labels)
    stage_results = []

    for stage in stages:
        mask = stage_labels == stage
        if mask.sum() == 0:
            continue

        X_stage = X_real[mask]
        Y_stage_full = Y_real_full[mask]

        with torch.no_grad():
            X_stage_tensor = torch.from_numpy(X_stage).float().to(device)
            Y_pred_stage_tensor = model(X_stage_tensor)

        y_true = torch.from_numpy(Y_stage_full).float().to(device)
        y_pred = Y_pred_stage_tensor

        rmse_all = float(masked_rmse(y_true, y_pred).item())
        sam = float(masked_sam(y_true, y_pred).item())
        rmse_band_val = float(band_rmse(y_true, y_pred).item())

        # VI 指标（该时期）
        Y_pred_stage = Y_pred_stage_tensor.cpu().numpy()
        VI_true_stage = _compute_vi_from_spectra(Y_stage_full, wavelengths_full)
        VI_pred_stage = _compute_vi_from_spectra(Y_pred_stage, wavelengths_full)
        vi_metrics_stage = _vi_error_metrics(VI_true_stage, VI_pred_stage)

        stage_results.append(
            {
                "scope": "stage",
                "stage": str(stage),
                "model": model_path.name,
                "n_samples": int(X_stage.shape[0]),
                "rmse_all": rmse_all,
                "sam": sam,
                "rmse_band": rmse_band_val,
                "vi_metrics": vi_metrics_stage,
            }
        )

    return stage_results


# ----------------------------------------------------------------------
# 7. CLI 入口（PyTorch 版）
# ----------------------------------------------------------------------
def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="m2h_swir_finetuned_new_head.pt",
        help="PyTorch state_dict file under MODEL_DIR, e.g. m2h_swir_finetuned_new_head.pt",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="m2h",
        choices=["m2h","simple_cnn"],
        help="Architecture type of the model checkpoint.",
    )
    parser.add_argument(
        "--by_stage",
        action="store_true",
        help="If set, evaluate metrics per growth stage using REAL_DIR/stages.npy",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="If set, save metrics to MODEL_DIR/metrics/<model_name>_real[_by_stage].json",
    )
    parsed = parser.parse_args(args=args)

    model_path = MODEL_DIR / parsed.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if parsed.by_stage:
        # -------- 按时期评估 --------
        stage_results = evaluate_model_on_real_by_stage(model_path, model_type=parsed.model_type)

        print("=== Evaluation on REAL UAV–ASD pairs (per stage, PyTorch) ===")
        for res in stage_results:
            print(
                f"Stage: {res['stage']:<10} | "
                f"N={res['n_samples']:3d} | "
                f"RMSE_all={res['rmse_all']:.6f} | "
                f"SAM={res['sam']:.6f} | "
                f"RMSE_band={res['rmse_band']:.6f}"
            )
            vi = res["vi_metrics"]
            print(
                f"    NDVI_RMSE={vi['NDVI']['rmse']:.4f}, "
                f"NDRE_RMSE={vi['NDRE']['rmse']:.4f}, "
                f"GNDVI_RMSE={vi['GNDVI']['rmse']:.4f}"
            )

        if parsed.save_json:
            out_dir = MODEL_DIR / "metrics"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{model_path.stem}_real_by_stage.json"
            with open(out_path, "w") as f:
                json.dump(stage_results, f, indent=2)
            print(f"Stage-wise metrics saved to {out_path}")

    else:
        # -------- 全部样本整体评估 --------
        metrics = evaluate_model_on_real_global(model_path, model_type=parsed.model_type)

        print("=== Evaluation on REAL UAV–ASD pairs (global, PyTorch) ===")
        print(f"Model      : {metrics['model']}")
        print(f"#Samples   : {metrics['n_samples']}")
        print(f"RMSE_all   : {metrics['rmse_all']:.6f}")
        print(f"SAM (rad)  : {metrics['sam']:.6f}")
        print(f"RMSE_band  : {metrics['rmse_band']:.6f}")

        print("VI RMSE / R2:")
        for name, v in metrics["vi_metrics"].items():
            print(f"  {name:10s}  RMSE={v['rmse']:.4f}  R2={v['r2']:.4f}")

        if parsed.save_json:
            out_dir = MODEL_DIR / "metrics"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{model_path.stem}_real.json"
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Global metrics saved to {out_path}")


if __name__ == "__main__":
    # simple CNN (纯数据驱动 SimpleCNN)
    print("\n=== simple CNN (pure data-driven) ===")
    main([
        "--model_name", "simple_cnn_data_driven_new_head.pt",
        "--model_type", "simple_cnn",
        "--save_json",
    ])

    # Data-trained M2H
    print("\n=== data-driven m2h_swir_new_head (M2H) ===")
    main([
        "--model_name", "m2h_data_driven_new_head.pt",
        "--model_type", "m2h",
        "--save_json",
    ])

    # LUT-trained M2H
    print("\n=== m2h_swir_lut_new_head (M2H, LUT-trained) ===")
    main([
        "--model_name", "m2h_swir_lut_new_head.pt",
        "--model_type", "m2h",
        "--save_json",
    ])

    # LUT + finetuned on real
    print("\n=== m2h_swir_finetuned_new_head (M2H finetuned) ===")
    main([
        "--model_name", "m2h_swir_finetuned_new_head.pt",
        "--model_type", "m2h",
        "--save_json",
    ])

    print("\n=== All PyTorch evaluations completed! ===")


