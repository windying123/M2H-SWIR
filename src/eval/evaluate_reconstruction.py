"""Evaluate reconstruction performance on LUT test set (with VI metrics)
and optionally on real ASD test set. (PyTorch version)
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
current_dir = os.path.dirname(os.path.abspath(__file__))       # .../src/m2h_swir/eval
parent_dir = os.path.dirname(current_dir)                      # .../src/m2h_swir
sys.path.insert(0, parent_dir)

from config import LUT_DIR, REAL_DIR, MODEL_DIR, WL_MIN, WL_MAX, WL_STEP  # type: ignore
from model.losses import (                                       # type: ignore
    spectral_rmse,
    spectral_angle_mapper,
    band_rmse,
)
from model.models import build_m2h_swir_model, build_simple_cnn  # type: ignore
from data_utils import split_simulated_lut  # type: ignore



# ----------------- VI 相关工具 -----------------

def _build_band_index_map(wavelengths_full: np.ndarray):
    """生成 {波长(整数 nm): 索引} 映射。"""
    return {int(w): idx for idx, w in enumerate(wavelengths_full.astype(int))}


def _compute_vi_from_spectra(Y: np.ndarray,
                             wavelengths_full: np.ndarray) -> dict:
    """
    给定 full 波段光谱 Y (N, D) 和 对应波长 wavelengths_full (D,),
    计算 6 个氮敏感植被指数:
        NDVI(850,670), NDRE(850,730), GNDVI(850,550),
        OSAVI1510(850,1510), N870_1450(870,1450), N850_1510(850,1510)
    返回:
        {"NDVI": (N,), "NDRE": (N,), ...}
    """
    band_idx_map = _build_band_index_map(wavelengths_full)
    eps = 1e-8

    def get_band(R, wl):
        if wl not in band_idx_map:
            raise ValueError(f"目标波长 {wl} nm 不在 wavelengths_full 里，无法计算对应 VI。")
        return R[..., band_idx_map[wl]]

    VIs = {}

    R850  = get_band(Y, 850)
    R870  = get_band(Y, 870)
    R670  = get_band(Y, 670)
    R730  = get_band(Y, 730)
    R550  = get_band(Y, 550)
    R1450 = get_band(Y, 1450)
    R1510 = get_band(Y, 1510)

    # NDVI
    VIs["NDVI"] = (R850 - R670) / (R850 + R670 + eps)
    # NDRE
    VIs["NDRE"] = (R850 - R730) / (R850 + R730 + eps)
    # GNDVI (Green NDVI)
    VIs["GNDVI"] = (R850 - R550) / (R850 + R550 + eps)
    # OSAVI1510
    VIs["OSAVI1510"] = (1.0 + 0.16) * (R850 - R1510) / (R850 + R1510 + 0.16 + eps)
    # N870_1450
    VIs["N870_1450"] = (R870 - R1450) / (R870 + R1450 + eps)
    # N850_1510
    VIs["N850_1510"] = (R850 - R1510) / (R850 + R1510 + eps)

    return VIs


def _vi_error_metrics(VI_true: dict, VI_pred: dict) -> dict:
    """
    对每一个 VI，计算 RMSE 和 R^2。
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


# ----------------- 模型加载（支持 m2h / simple_cnn） -----------------

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
        model, _, _ = build_simple_cnn(
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


# ----------------- LUT 上评估 (带 VI) -----------------

def eval_on_lut(model_path: Path,
                model_type: str = "m2h",
                save_json: bool = True):
    """
    在 LUT test 集上评估：
      - 全谱 RMSE / MAE / SAM / band RMSE
      - 6 个氮敏感 VI 的 RMSE / R^2
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_ms = np.load(LUT_DIR / "X_ms.npy")
    Y_hs = np.load(LUT_DIR / "Y_hs.npy")
    splits = split_simulated_lut(X_ms, Y_hs)
    X_test, Y_test = splits["test"]

    input_dim = X_ms.shape[1]
    output_dim = Y_hs.shape[1]

    model = _load_pytorch_model(model_path, input_dim, output_dim, device, model_type=model_type)

    # 构造 LUT 波长轴（与你训练脚本的设置一致）
    wavelengths_full = WL_MIN + np.arange(output_dim) * WL_STEP

    # 预测
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float().to(device)
        Y_pred_tensor = model(X_test_tensor).cpu()

    Y_pred = Y_pred_tensor.numpy()

    # 光谱误差（numpy 版）
    diff = Y_pred - Y_test
    rmse_all = float(np.sqrt(np.mean(diff ** 2)))
    mae_all = float(np.mean(np.abs(diff)))

    # 使用 PyTorch 版本的 SAM / band_rmse
    y_true_t = torch.from_numpy(Y_test).float()
    y_pred_t = torch.from_numpy(Y_pred).float()
    sam = float(spectral_angle_mapper(y_true_t, y_pred_t).item())
    rmse_band_val = float(band_rmse(y_true_t, y_pred_t).item())

    # VI 误差
    VI_true = _compute_vi_from_spectra(Y_test, wavelengths_full)
    VI_pred = _compute_vi_from_spectra(Y_pred, wavelengths_full)
    vi_metrics = _vi_error_metrics(VI_true, VI_pred)

    metrics = {
        "split": "lut_test",
        "rmse_all": rmse_all,
        "mae_all": mae_all,
        "sam": sam,
        "rmse_band": rmse_band_val,
        "vi_metrics": vi_metrics,
    }

    if save_json:
        model_name = model_path.stem
        out_dir = MODEL_DIR / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}_lut_test.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"LUT metrics (with VI) saved to: {out_path}")

    return metrics


# ----------------- REAL test（可选光谱评估） -----------------

def eval_on_real_test(model_path: Path,
                      model_type: str = "m2h",
                      save_json: bool = True):
    """
    可选：在 REAL test (X_uav_ms_test, Y_asd_hs_test) 上做一个简单评估。
    注意：这里假设 Y_asd_hs_test 已经与模型输出维度对齐；
    若你希望完整的掩蔽逻辑 + VI 指标，请用 eval_on_real_torch.py。
    """
    X_real_path = REAL_DIR / "X_uav_ms_test.npy"
    Y_real_path = REAL_DIR / "Y_asd_hs_test.npy"
    if not X_real_path.exists() or not Y_real_path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_real = np.load(X_real_path)
    Y_real = np.load(Y_real_path)

    input_dim = X_real.shape[1]
    output_dim = Y_real.shape[1]

    model = _load_pytorch_model(model_path, input_dim, output_dim, device, model_type=model_type)

    with torch.no_grad():
        X_real_tensor = torch.from_numpy(X_real).float().to(device)
        Y_pred_tensor = model(X_real_tensor).cpu()

    Y_pred = Y_pred_tensor.numpy()

    y_true_t = torch.from_numpy(Y_real).float()
    y_pred_t = torch.from_numpy(Y_pred).float()

    rmse_all = float(spectral_rmse(y_true_t, y_pred_t).item())
    mae_all = float(torch.mean(torch.abs(y_pred_t - y_true_t)).item())
    sam = float(spectral_angle_mapper(y_true_t, y_pred_t).item())
    rmse_band_val = float(band_rmse(y_true_t, y_pred_t).item())

    metrics = {
        "split": "real_test",
        "rmse_all": rmse_all,
        "mae_all": mae_all,
        "sam": sam,
        "rmse_band": rmse_band_val,
    }

    if save_json:
        model_name = model_path.stem
        out_dir = MODEL_DIR / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}_real_test.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Real-test metrics saved to: {out_path}")

    return metrics


# ----------------- CLI 入口 -----------------

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="m2h_swir_lut_new_head.pt",  # 你可以改成自己的默认文件名
        help="PyTorch state_dict file under MODEL_DIR.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="m2h",
        choices=["m2h", "simple_cnn"],
        help="Architecture type of the model checkpoint.",
    )
    parser.add_argument(
        "--no_real",
        action="store_true",
        help="只评估 LUT test，不在 REAL test 上评估。",
    )
    parsed = parser.parse_args(args=args)

    model_path = MODEL_DIR / parsed.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("=== Evaluating on LUT test set (with VI) ===")
    lut_metrics = eval_on_lut(model_path, model_type=parsed.model_type)
    print(lut_metrics)

    if not parsed.no_real:
        print("\n=== Evaluating on REAL test set (if exists) ===")
        real_metrics = eval_on_real_test(model_path, model_type=parsed.model_type)
        if real_metrics is not None:
            print(real_metrics)
        else:
            print("No REAL test set found; skipping.")


if __name__ == "__main__":
    main()

