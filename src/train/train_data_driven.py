"""
Train pure data-driven models (M2H or SimpleCNN) on REAL UAV–ASD pairs,
with full-grid hyperspectral output (e.g. 400–2500nm every 1nm -> 2101 bands).

关键设计：
- 模型输出 full-grid 光谱 (B, D_full)
- 但 loss 只在 ASD 真有观测的波段上计算 (masked RMSE + alpha * masked SAM + beta * band_RMSE)
- band_RMSE 只在 “既是氮敏感波段、又在 ASD 有效波段里”的那些波段上算
- 两个模型共用同一套训练逻辑：
    * --model_type m2h
    * --model_type simple_cnn
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
from torch.utils.data import TensorDataset, DataLoader

# ===== 路径与工程导入 =====
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../src/...
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
DEFAULT_ALPHA=0.2
DEFAULT_BETA=2.0
from config import (  # type: ignore
    REAL_DIR,
    MODEL_DIR,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    WL_MIN,
    WL_MAX,
    WL_STEP,
    N_SENSITIVE_WAVELENGTHS,
)
from model.models import build_m2h_swir_model, build_simple_cnn  # type: ignore
from model.losses import spectral_rmse, spectral_angle_mapper  # type: ignore

SPLIT_PATH = REAL_DIR / "train_val_split_indices.npz"
# ===== 一些辅助函数 =====

def build_full_grid_labels(Y_compact: np.ndarray,
                           wavelength_real: np.ndarray):
    """
    把 ASD 紧凑光谱扩展到 full grid (400–2500, step=WL_STEP)，
    并给出 valid_mask，表示哪些 full-grid 波段在 ASD 中有真实值。

    注意：Y_full 中未观测波段用 0 填，但后续 loss 会用 valid_mask 屏蔽掉这些位置。
    """
    num_wl = int((WL_MAX - WL_MIN) / WL_STEP) + 1
    wl_full = WL_MIN + np.arange(num_wl) * WL_STEP      # (D_full,)

    # 哪些 full-grid 波段在 ASD 中存在
    valid_mask = np.isin(wl_full, wavelength_real)
    if valid_mask.sum() != wavelength_real.shape[0]:
        raise ValueError(
            f"valid_mask True 数={valid_mask.sum()} 与 ASD 波段数={wavelength_real.shape[0]} 不一致"
        )

    # 构造 full-grid 标签
    Y_full = np.zeros((Y_compact.shape[0], num_wl), dtype=Y_compact.dtype)
    Y_full[:, valid_mask] = Y_compact

    return Y_full, wl_full, valid_mask


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Pure data-driven full-grid training (M2H or SimpleCNN) on REAL UAV–ASD."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="m2h",
        choices=["m2h", "simple_cnn"],
        help="Model architecture: 'm2h' or 'simple_cnn'.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Weight for SAM term in loss.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="Weight for band RMSE term (N-sensitive bands).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="data_driven_fullgrid.pt",
        help="Output model filename under MODEL_DIR.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Max training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="Early stopping patience.",
    )

    parsed = parser.parse_args(args=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 1. 读 REAL 数据 ===
    X_real = np.load(REAL_DIR / "X_uav_ms.npy")         # (N, num_ms_bands)
    Y_compact = np.load(REAL_DIR / "Y_asd_hs.npy")      # (N, n_valid)
    wavelength_real = np.load(REAL_DIR / "wavelength.npy")  # (n_valid,)

    if Y_compact.shape[1] != wavelength_real.shape[0]:
        raise ValueError(
            f"Y_asd_hs second dim = {Y_compact.shape[1]} "
            f"!= wavelength_real length = {wavelength_real.shape[0]}"
        )

    # 扩展到 full-grid 标签 + valid_mask
    Y_full, wl_full, valid_mask = build_full_grid_labels(Y_compact, wavelength_real)

    print(f"X_real shape : {X_real.shape}")
    print(f"Y_compact    : {Y_compact.shape}")
    print(f"Y_full       : {Y_full.shape}")
    print(f"#valid bands : {valid_mask.sum()} / {valid_mask.shape[0]}")

    input_shape = (X_real.shape[1],)      # (num_ms_bands,)
    output_dim = Y_full.shape[1]          # full-grid 维度，一般 2101

    # === 1.5 构造 mask 和 氮敏感波段索引 ===
    valid_mask_t = torch.from_numpy(valid_mask.astype("float32")).to(device)   # (D_full,)
    valid_mask_t = valid_mask_t.view(1, -1)                                     # (1, D_full)

    # full-grid 波长索引表
    wl_full_int = wl_full.astype(int)
    wl_to_idx_full = {int(w): i for i, w in enumerate(wl_full_int)}
    wl_real_set = set(int(w) for w in wavelength_real.astype(int))

    band_indices = []
    missing_bands = []
    for w in N_SENSITIVE_WAVELENGTHS:
        w_int = int(w)
        # 要求：既在 full-grid 上存在，也在 ASD 有效波段中存在
        if w_int in wl_to_idx_full and w_int in wl_real_set:
            band_indices.append(wl_to_idx_full[w_int])
        else:
            missing_bands.append(w_int)

    print(f"N-sensitive wavelengths (config): {list(N_SENSITIVE_WAVELENGTHS)}")
    print(f"Used full-grid band indices     : {band_indices}")
    if missing_bands:
        print(f"⚠ 以下敏感波段在 ASD 或 full-grid 中不存在，将在 band_RMSE 中忽略: {missing_bands}")

    band_idx_t = torch.tensor(band_indices, dtype=torch.long, device=device) if band_indices else None

    # === 2. 按生育时期分层划分 train/val（固定一次划分，之后复用） ===
    stages_path = REAL_DIR / "stages.npy"
    if not stages_path.exists():
        raise FileNotFoundError("需要 stages.npy 才能进行按时期分层划分！")

    stages = np.load(stages_path)  # shape = (300,)
    unique_stages = np.unique(stages)

    if SPLIT_PATH.exists():
        # 已经保存过索引 → 直接复用
        data = np.load(SPLIT_PATH)
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        print(f"Loaded existing stratified split from {SPLIT_PATH}")
    else:
        # 第一次划分：按每个时期内部 80/20 分层
        rng = np.random.RandomState(42)
        train_idx_list = []
        val_idx_list = []

        for s in unique_stages:
            idx_s = np.where(stages == s)[0]  # 当前时期的所有样本索引
            rng.shuffle(idx_s)

            n_s = len(idx_s)  # 例如 60
            n_train_s = int(0.8 * n_s)  # 例如 48

            train_idx_list.append(idx_s[:n_train_s])
            val_idx_list.append(idx_s[n_train_s:])

        train_idx = np.concatenate(train_idx_list)
        val_idx = np.concatenate(val_idx_list)

        np.savez(SPLIT_PATH, train_idx=train_idx, val_idx=val_idx)
        print(f"Created new STRATIFIED train/val split and saved to {SPLIT_PATH}")

    # 使用分层后的索引切分数据
    X_train, Y_train = X_real[train_idx], Y_full[train_idx]
    X_val, Y_val = X_real[val_idx], Y_full[val_idx]

    print("=== Stratified Split Summary ===")
    print(f"Total samples: {len(stages)}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # 每个时期的统计信息
    for s in unique_stages:
        n_train_s = np.sum(stages[train_idx] == s)
        n_val_s = np.sum(stages[val_idx] == s)
        print(f"  Stage {s}: train={n_train_s}, val={n_val_s}")
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
        batch_size=parsed.batch_size,
        shuffle=True,  # 每个 epoch 内打乱顺序，但 train/val 样本集合不变
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=parsed.batch_size,
        shuffle=False,
    )
    # === 3. 构建模型 ===
    if parsed.model_type == "m2h":
        model, optimizer, _ = build_m2h_swir_model(
            input_shape=input_shape,
            output_dim=output_dim,
            alpha=parsed.alpha,
            beta=parsed.beta,
            # 这里可以强制开启 transformer 等结构，也可以继续用默认
            use_multiscale_conv=True,
            use_se=True,
            use_residual=True,
            use_transformer=True,
        )
    else:
        model, optimizer, _ = build_simple_cnn(
            input_shape=input_shape,
            output_dim=output_dim,
            alpha=parsed.alpha,
            beta=parsed.beta,
        )

    model.to(device)

    # === 3.5 定义 masked loss ===

    def masked_spectral_rmse(y_true, y_pred):
        # y_*: (B, D_full)
        y_true_sel = y_true * valid_mask_t
        y_pred_sel = y_pred * valid_mask_t
        return spectral_rmse(y_true_sel, y_pred_sel)

    def masked_sam(y_true, y_pred):
        y_true_sel = y_true * valid_mask_t
        y_pred_sel = y_pred * valid_mask_t
        return spectral_angle_mapper(y_true_sel, y_pred_sel)

    def band_rmse_masked(y_true, y_pred):
        if band_idx_t is None or band_idx_t.numel() == 0:
            return torch.tensor(0.0, device=y_true.device)
        y_true_sel = torch.index_select(y_true, dim=-1, index=band_idx_t)
        y_pred_sel = torch.index_select(y_pred, dim=-1, index=band_idx_t)
        diff = y_pred_sel - y_true_sel
        return torch.sqrt(torch.mean(diff ** 2) + 1e-12)

    def loss_fn(y_pred, y_true):
        rmse_all = masked_spectral_rmse(y_true, y_pred)
        sam = masked_sam(y_true, y_pred)
        rmse_band = band_rmse_masked(y_true, y_pred)
        return rmse_all + parsed.alpha * sam + parsed.beta * rmse_band

    # === 4. 日志与输出路径 ===
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = MODEL_DIR / "logs" / "data_driven_fullgrid" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = MODEL_DIR / parsed.model_name
    csv_log_path = log_dir / "train_log.csv"
    config_path = log_dir / f"{parsed.model_name}_config.json"

    with csv_log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_val = float("inf")
    best_state = None
    patience_cnt = 0

    # === 5. 训练循环 ===
    for epoch in range(1, parsed.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_pred = model(xb)
                val_loss = loss_fn(y_pred, yb)
                val_losses.append(val_loss.item())

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{parsed.epochs:03d} "
            f"- train_loss: {train_loss:.6f} "
            f"- val_loss: {val_loss:.6f} "
            f"- lr: {lr:.2e}"
        )

        with csv_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, lr])

        # early stopping
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= parsed.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # === 6. 保存最佳模型 ===
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        model.load_state_dict(best_state)
        print(f"Best model saved to {ckpt_path}")
    else:
        torch.save(model.state_dict(), ckpt_path)
        print(f"No best_state tracked, last model saved to {ckpt_path}")

    # === 7. 在全部 REAL 数据上做一次 quick eval ===
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(X_real).float().to(device)
        Y_pred_all = model(X_all)                      # (N, D_full)
        Y_true_all = torch.from_numpy(Y_full).float().to(device)

    rmse_all = float(masked_spectral_rmse(Y_true_all, Y_pred_all).item())
    sam_all = float(masked_sam(Y_true_all, Y_pred_all).item())
    rmse_band_val = float(band_rmse_masked(Y_true_all, Y_pred_all).item())

    print("\n=== Quick eval on REAL UAV–ASD (full-grid, masked metrics) ===")
    print(f"Model      : {parsed.model_name}")
    print(f"Model type : {parsed.model_type}")
    print(f"#Samples   : {X_real.shape[0]}")
    print(f"Output dim : {output_dim} (full-grid)")
    print(f"RMSE_all   : {rmse_all:.6f}")
    print(f"SAM (rad)  : {sam_all:.6f}")
    print(f"band_RMSE  : {rmse_band_val:.6f}")

    # === 8. 保存配置 ===
    config = {
        "framework": "pytorch",
        "run_id": run_id,
        "model_file": str(ckpt_path),
        "model_type": parsed.model_type,
        "input_shape": input_shape,
        "output_dim": output_dim,
        "alpha": parsed.alpha,
        "beta": parsed.beta,
        "batch_size": parsed.batch_size,
        "epochs": parsed.epochs,
        "patience": parsed.patience,
        "data": {
            "REAL_DIR": str(REAL_DIR),
            "X_real_shape": list(X_real.shape),
            "Y_compact_shape": list(Y_compact.shape),
            "Y_full_shape": list(Y_full.shape),
            "n_valid_bands": int(valid_mask.sum()),
            "used_band_indices": band_indices,
        },
        "note": "Pure data-driven full-grid model (M2H or SimpleCNN) with masked loss on ASD valid bands.",
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {config_path}")


if __name__ == "__main__":
    # 你也可以只用 main()，这里给一个示例：先训 SimpleCNN，再训 M2H
    print("\n=== Train SimpleCNN (full-grid) ===")
    main([
        "--model_type", "simple_cnn",
        "--model_name", "simple_cnn_data_driven_new_head.pt",
    ])

    print("\n=== Train M2H (full-grid) ===")
    main([
        "--model_type", "m2h",
        "--model_name", "m2h_data_driven_new_head.pt",
    ])

    print("\n=== All data-driven full-grid baselines completed! ===")




