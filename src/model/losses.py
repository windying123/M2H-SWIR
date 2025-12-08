"""Loss functions for spectral reconstruction."""

import torch
import sys
import os

# === 路径与导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))       # .../src/m2h_swir/prosail_sim
parent_dir = os.path.dirname(current_dir)                      # .../src/m2h_swir
sys.path.insert(0, parent_dir)
from config import N_SENSITIVE_WAVELENGTHS, WL_MIN, WL_STEP


def spectral_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    全谱 RMSE: sqrt(mean((y_true - y_pred)^2))
    y_* 形状: (batch, num_wavelengths)
    """
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def spectral_angle_mapper(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute SAM (radians) and average over batch.
    y_* 形状: (batch, num_wavelengths)
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    dot_prod = torch.sum(y_true * y_pred, dim=-1)          # (batch,)
    norm_true = torch.norm(y_true, dim=-1)                 # (batch,)
    norm_pred = torch.norm(y_pred, dim=-1)                 # (batch,)

    cos_theta = dot_prod / (norm_true * norm_pred + eps)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angle = torch.acos(cos_theta)                          # (batch,)

    return torch.mean(angle)


def band_rmse(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    band_wavelengths=None,
) -> torch.Tensor:
    """
    RMSE computed on selected nitrogen-sensitive wavelengths.
    默认使用 config.N_SENSITIVE_WAVELENGTHS。
    """
    if band_wavelengths is None:
        band_wavelengths = N_SENSITIVE_WAVELENGTHS

    # 根据 WL_MIN / WL_STEP 计算索引
    indices = [(w - WL_MIN) // WL_STEP for w in band_wavelengths]
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=y_true.device)

    # 选取相应波段
    # y_*: (batch, num_wavelengths)
    y_true_sel = torch.index_select(y_true, dim=-1, index=indices_tensor)
    y_pred_sel = torch.index_select(y_pred, dim=-1, index=indices_tensor)

    return torch.sqrt(torch.mean((y_true_sel - y_pred_sel) ** 2))


def combined_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """
    Global RMSE + alpha * SAM + beta * band RMSE
    """
    rmse_all = spectral_rmse(y_true, y_pred)
    sam = spectral_angle_mapper(y_true, y_pred)
    rmse_band_val = band_rmse(y_true, y_pred)
    return rmse_all + alpha * sam + beta * rmse_band_val
