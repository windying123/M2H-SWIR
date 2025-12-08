"""Fine-tune LUT-pretrained model on real UAV–ASD pairs."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 可选：让 cuDNN 为固定输入搜索最快实现
torch.backends.cudnn.benchmark = True

# === 路径与导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../src/m2h_swir/train
parent_dir = os.path.dirname(current_dir)                  # .../src/m2h_swir
sys.path.insert(0, parent_dir)

from config import (                                       # type: ignore
    REAL_DIR,
    MODEL_DIR,
    BATCH_SIZE,
    PATIENCE,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    WL_MIN,
    WL_MAX,
    WL_STEP,
)
from model.models import build_m2h_swir_model              # type: ignore
from model.losses import (                                 # type: ignore
    spectral_rmse,
    spectral_angle_mapper,
    band_rmse,
)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="m2h_swir_lut_new_head.pt",
                        help="预训练 LUT 模型 state_dict 文件名（在 MODEL_DIR 下）")
    parser.add_argument("--model_name", type=str, default="m2h_swir_finetuned_new_head.pt",
                        help="微调后模型保存文件名（在 MODEL_DIR 下）")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--epochs", type=int, default=100)
    parsed = parser.parse_args(args=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 读取真实 UAV–ASD 配对
    X_real = np.load(REAL_DIR / "X_uav_ms.npy")             # (N, 10)
    Y_real_compact = np.load(REAL_DIR / "Y_asd_hs.npy")     # (N, 1670)
    wavelength_real = np.load(REAL_DIR / "wavelength.npy")  # (1670,)

    # 2) 用 WL_MIN / WL_MAX / WL_STEP 构造 LUT 的完整波长栅格
    output_dim = int((WL_MAX - WL_MIN) / WL_STEP) + 1       # 例如 2101
    wavelengths_full = WL_MIN + np.arange(output_dim) * WL_STEP

    # 3) 用 ASD 实测波长构造 valid_mask
    valid_mask = np.isin(wavelengths_full, wavelength_real)   # (output_dim,)
    assert valid_mask.sum() == wavelength_real.shape[0], \
        f"波段数不一致：valid_mask 中 True={valid_mask.sum()}，但 wavelength_real 长度={wavelength_real.shape[0]}"

    # 4) 构造完整标签 (N, output_dim)
    Y_real_full = np.zeros((Y_real_compact.shape[0], output_dim), dtype=Y_real_compact.dtype)
    Y_real_full[:, valid_mask] = Y_real_compact

    # 5) train/val 划分
    n = X_real.shape[0]
    n_train = int(0.8 * n)
    X_train, Y_train = X_real[:n_train], Y_real_full[:n_train]
    X_val, Y_val = X_real[n_train:], Y_real_full[n_train:]

    input_shape = (X_real.shape[1],)

    # 6) 构建与 LUT 一致的模型结构，并加载预训练权重
    # 注意：这里假定 LUT 训练时使用 build_m2h_swir_model 默认结构
    model, _, _ = build_m2h_swir_model(
        input_shape=input_shape,
        output_dim=output_dim,
        alpha=parsed.alpha,
        beta=parsed.beta,
    )
    model.to(device)

    pretrained_path = MODEL_DIR / parsed.pretrained
    print(f"Loading pretrained weights from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state_dict)

    # 7) 可选：只微调“头部”，冻结 backbone
    # 这里粗略地只训练 conv_local / conv_out，其他全部冻结
    for name, param in model.named_parameters():
        if ("conv_local" in name) or ("conv_out" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

    # 8) 定义只在有效波段上计算的 loss
    valid_mask_torch = torch.from_numpy(valid_mask.astype("float32")).to(device)   # (output_dim,)
    valid_mask_torch = valid_mask_torch.view(1, -1)                                # (1, output_dim)

    def combined_loss_real(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_*: (batch, output_dim)
        - RMSE & SAM: 只在 ASD 有测量的波段上计算（用掩码）
        - band_rmse: 仍然用你定义好的 N 敏感波段（通常都在有效波段内）
        """
        y_true_sel = y_true * valid_mask_torch
        y_pred_sel = y_pred * valid_mask_torch

        rmse_all = spectral_rmse(y_true_sel, y_pred_sel)
        sam = spectral_angle_mapper(y_true_sel, y_pred_sel)
        rmse_band_val = band_rmse(y_true, y_pred)

        return rmse_all + parsed.alpha * sam + parsed.beta * rmse_band_val

    # 9) 日志与输出路径
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = MODEL_DIR / "logs" / "finetune" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = MODEL_DIR / parsed.model_name
    csv_log_path = log_dir / f"{parsed.model_name}_train_log.csv"

    writer = SummaryWriter(log_dir=str(log_dir))

    with csv_log_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_loss", "val_loss", "lr"])

    # DataLoader
    pin_memory = device.type == "cuda"
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
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    # Scheduler & early stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # 10) 训练循环
    for epoch in range(1, parsed.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = combined_loss_real(y_pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                y_pred = model(xb)
                val_loss = combined_loss_real(y_pred, yb)
                val_losses.append(val_loss.item())

        val_loss_mean = float(np.mean(val_losses)) if val_losses else 0.0

        scheduler.step(val_loss_mean)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{parsed.epochs} "
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

        # EarlyStopping + 只保存最优
        if val_loss_mean < best_val_loss - 1e-6:
            best_val_loss = val_loss_mean
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best finetuned model saved to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered (no improvement in {PATIENCE} epochs).")
                break

    writer.close()

    print(f"Finetuning finished. Best model stored at {ckpt_path}")


if __name__ == "__main__":
    main()

