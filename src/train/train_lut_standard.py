"""Standardized training script for M2H-SWIR on PROSAIL-PRO LUT

Features:
- Trains build_m2h_swir_model on pre-generated LUT (X_ms, Y_hs).
- Supports command-line args for alpha, beta and transformer hyperparameters.
- Logs training with TensorBoard and CSV.
- Saves:
    * Best model (by val_loss)
    * config.json (all hyperparameters & paths)
    * model_summary.txt (PyTorch str(model))
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import os
import sys
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# === 路径与导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/m2h_swir/train 或 prosail_sim
parent_dir = os.path.dirname(current_dir)                 # .../src/m2h_swir
sys.path.insert(0, parent_dir)

from config import (  # type: ignore
    LUT_DIR,
    MODEL_DIR,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    WL_MIN,
    WL_STEP,
)
from data_utils import split_simulated_lut  # type: ignore
from model.models import build_m2h_swir_model  # type: ignore


# ========= PyTorch 版评估指标 =========

def spectral_rmse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    全谱 RMSE: sqrt(mean((y_true - y_pred)^2))
    y_* 形状: (N, B)
    """
    mse = torch.mean((y_true - y_pred) ** 2)
    return torch.sqrt(mse)


def band_rmse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    每个波段先算 RMSE，再对波段取平均。
    """
    mse_per_band = torch.mean((y_true - y_pred) ** 2, dim=0)   # (B,)
    rmse_per_band = torch.sqrt(mse_per_band)                   # (B,)
    return torch.mean(rmse_per_band)


def spectral_angle_mapper_torch(y_true: torch.Tensor, y_pred: torch.Tensor,
                                eps: float = 1e-8) -> torch.Tensor:
    """
    SAM（弧度）: acos( (x·y) / (||x|| ||y||) )，对样本取平均。
    y_* 形状: (N, B)
    """
    dot = torch.sum(y_true * y_pred, dim=1)               # (N,)
    norm_true = torch.norm(y_true, dim=1)                 # (N,)
    norm_pred = torch.norm(y_pred, dim=1)                 # (N,)
    denom = norm_true * norm_pred + eps
    cos_sim = dot / denom
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
    angles = torch.acos(cos_sim)
    return torch.mean(angles)


# ========= 辅助函数 =========

def save_model_summary(model: nn.Module, out_path: Path):
    """Save PyTorch model structure to a text file."""
    with out_path.open("w", encoding="utf-8") as f:
        f.write(str(model) + "\n")


def evaluate_on_split(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    split_name: str,
    device: torch.device,
    batch_size: int = 1024,
):
    """
    在给定数据集上计算全谱 RMSE、SAM、band RMSE，返回 dict。
    """
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            y_pred = model(xb)  # (B, output_dim)
            preds.append(y_pred.cpu())
            trues.append(yb.cpu())

    y_pred_all = torch.cat(preds, dim=0)  # (N, B)
    y_true_all = torch.cat(trues, dim=0)  # (N, B)

    rmse_all = float(spectral_rmse_torch(y_true_all, y_pred_all).item())
    sam = float(spectral_angle_mapper_torch(y_true_all, y_pred_all).item())
    rmse_band = float(band_rmse_torch(y_true_all, y_pred_all).item())

    return {
        "split": split_name,
        "n_samples": int(X.shape[0]),
        "rmse_all": rmse_all,
        "sam": sam,
        "rmse_band": rmse_band,
    }


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Train M2H-SWIR model (PyTorch) on LUT with full logging and config export."
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Loss weight for SAM term.")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA,
                        help="Loss weight for band RMSE term.")
    parser.add_argument("--model_name", type=str, default="m2h_swir_lut_new_head.pt",
                        help="Output model filename under MODEL_DIR. (extension arbitrary, e.g. .pt)")
    # Transformer 超参（和 models.build_m2h_swir_model 对齐）
    parser.add_argument("--use_transformer", type=str, choices=['true', 'false'], default='true',
                        help="Enable transformer encoder blocks.")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads in transformer.")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Transformer model dimension (projected channels).")
    parser.add_argument("--ff_dim", type=int, default=512,
                        help="Transformer feed-forward layer dimension.")
    parser.add_argument("--num_transformer_layers", type=int, default=2,
                        help="Number of stacked transformer encoder layers.")
    # 其他简单开关
    parser.add_argument("--no_multiscale_conv", action="store_true",
                        help="Disable multiscale (3/5/7) conv branches.")
    parser.add_argument("--no_se", action="store_true",
                        help="Disable squeeze-and-excitation blocks.")
    parser.add_argument("--no_residual", action="store_true",
                        help="Disable residual 1D conv blocks.")

    parsed = parser.parse_args(args=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 读取 LUT 数据 ===
    X_ms = np.load(LUT_DIR / "X_ms.npy")   # (N, 10) 或类似
    Y_hs = np.load(LUT_DIR / "Y_hs.npy")   # (N, 2101)

    # === 标准划分：train / val / test ===
    splits = split_simulated_lut(X_ms, Y_hs)
    (X_train, Y_train) = splits["train"]
    (X_val, Y_val) = splits["val"]
    (X_test, Y_test) = splits.get("test", (None, None))  # 若 data_utils 里有 test

    input_shape = (X_ms.shape[1],)    # e.g., (10,)
    output_dim = Y_hs.shape[1]        # e.g., 2101

    model, optimizer, loss_fn = build_m2h_swir_model(
        input_shape=input_shape,
        output_dim=output_dim,
        alpha=parsed.alpha,
        beta=parsed.beta,
        use_multiscale_conv=not parsed.no_multiscale_conv,
        use_se=not parsed.no_se,
        use_residual=not parsed.no_residual,
        use_transformer=parsed.use_transformer,
        num_heads=parsed.num_heads,
        d_model=parsed.d_model,
        ff_dim=parsed.ff_dim,
        num_transformer_layers=parsed.num_transformer_layers,
    )
    model.to(device)

    # === 日志与输出路径 ===
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 例如：models/logs/lut/20251201-153000
    log_dir = MODEL_DIR / "logs" / "lut" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = MODEL_DIR / parsed.model_name
    summary_path = log_dir / f"{parsed.model_name}_summary.txt"
    config_path = log_dir / f"{parsed.model_name}_config.json"
    metrics_path = log_dir / f"{parsed.model_name}_lut_metrics.json"
    csv_log_path = log_dir / "train_log.csv"

    # === 保存 model.summary ===
    save_model_summary(model, summary_path)

    # === TensorBoard ===
    writer = SummaryWriter(log_dir=str(log_dir))

    # === CSV Logger ===
    with csv_log_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_loss", "val_loss", "lr"])

    # === DataLoaders ===
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val).float(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # === LR Scheduler (ReduceLROnPlateau) ===
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    # === Early Stopping 控制 ===
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    # === 训练循环 ===
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)  # 注意：我在 models 里定义的是 loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # === 验证 ===
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_pred = model(xb)
                val_loss = loss_fn(y_pred, yb)
                val_losses.append(val_loss.item())

        val_loss_mean = float(np.mean(val_losses)) if val_losses else 0.0

        # === Scheduler & EarlyStopping ===
        scheduler.step(val_loss_mean)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{EPOCHS} "
            f"- loss: {train_loss:.6f} "
            f"- val_loss: {val_loss_mean:.6f} "
            f"- lr: {current_lr:.2e}"
        )

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss_mean, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # CSV
        with csv_log_path.open("a", newline="", encoding="utf-8") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, train_loss, val_loss_mean, current_lr])

        # 早停 & 保存 best model
        if val_loss_mean < best_val_loss - 1e-6:
            best_val_loss = val_loss_mean
            epochs_no_improve = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state_dict, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered (no improvement in {PATIENCE} epochs).")
                break

    writer.close()

    # === 加载 best 权重（用于后续评估） ===
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
    else:
        print("Warning: no improvement during training, using last epoch weights.")

    # 冗余安全保存一次
    torch.save(model.state_dict(), ckpt_path)
    print(f"Final model state_dict saved to {ckpt_path}")

    # === 在 LUT 上做严格评估（train / val / test）并存 JSON ===
    lut_metrics = []
    lut_metrics.append(evaluate_on_split(model, X_train, Y_train, "train", device))
    lut_metrics.append(evaluate_on_split(model, X_val, Y_val, "val", device))
    if X_test is not None and Y_test is not None:
        lut_metrics.append(evaluate_on_split(model, X_test, Y_test, "test", device))

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(lut_metrics, f, indent=2)
    print(f"LUT metrics saved to {metrics_path}")

    # === 保存完整 config（可在论文 & 复现中引用） ===
    wavelengths_full = WL_MIN + np.arange(output_dim) * WL_STEP

    config_dict = {
        "framework": "pytorch",
        "run_id": run_id,
        "model_file": str(ckpt_path),
        "input_shape": input_shape,
        "output_dim": output_dim,
        "alpha": parsed.alpha,
        "beta": parsed.beta,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "use_multiscale_conv": not parsed.no_multiscale_conv,
        "use_se": not parsed.no_se,
        "use_residual": not parsed.no_residual,
        "use_transformer": parsed.use_transformer,
        "transformer": {
            "num_heads": parsed.num_heads,
            "d_model": parsed.d_model,
            "ff_dim": parsed.ff_dim,
            "num_layers": parsed.num_transformer_layers,
        },
        "data": {
            "LUT_DIR": str(LUT_DIR),
            "MODEL_DIR": str(MODEL_DIR),
            "X_ms_shape": list(X_ms.shape),
            "Y_hs_shape": list(Y_hs.shape),
        },
        "wavelength_axis": {
            "WL_MIN": WL_MIN,
            "WL_STEP": WL_STEP,
            "num_wavelengths": int(output_dim),
            # 不直接保存 2101 个数值，避免 JSON 太大
        },
        "metrics_file": str(metrics_path),
        "summary_file": str(summary_path),
        "train_log_csv": str(csv_log_path),
        "tensorboard_log_dir": str(log_dir),
        "note": "Model saved as torch.state_dict(). Rebuild architecture with same config before loading.",
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Training config saved to {config_path}")


if __name__ == "__main__":
    main()
