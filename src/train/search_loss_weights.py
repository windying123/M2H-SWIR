
"""
Grid search for loss weights (alpha, beta) on LUT validation set.
GPU加速版本 - 使用PyTorch
"""

import json
import itertools
import time
from pathlib import Path
import os
import sys
from typing import Dict, Any, List, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# === 路径与导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import LUT_DIR, MODEL_DIR, BATCH_SIZE
from data_utils import split_simulated_lut

# ====== GPU设置 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    torch.cuda.empty_cache()

# ====== 网格搜索参数 ======
# 粗搜索
ALPHAS_COARSE = [0.0,0.2,0.4,0.6,0.8,1.0]
BETAS_COARSE = [0.0, 0.5, 1.0,1.5, 2.0]
EPOCHS_COARSE = 30

# 细搜索
N_ALPHA_FINE = 5
N_BETA_FINE = 5
EPOCHS_FINE = 50


# ====== PyTorch模型定义 ======
class M2HSWIRModel(nn.Module):
    """多光谱到高光谱转换模型 (PyTorch版本)"""

    def __init__(self, input_dim: int, output_dim: int, alpha: float = 0.0, beta: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, output_dim),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ====== 损失函数定义 ======
def spectral_rmse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """光谱RMSE损失"""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def spectral_angle_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """光谱角损失 (SAM)"""
    # 添加小值避免除零
    eps = 1e-8
    dot_product = torch.sum(y_true * y_pred, dim=1)
    norm_true = torch.norm(y_true, dim=1) + eps
    norm_pred = torch.norm(y_pred, dim=1) + eps
    cos_theta = dot_product / (norm_true * norm_pred)
    # 确保值在[-1, 1]范围内
    cos_theta = torch.clamp(cos_theta, -1 + eps, 1 - eps)
    angle = torch.acos(cos_theta)
    return torch.mean(angle)


def band_rmse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """波段RMSE损失"""
    band_mse = torch.mean((y_true - y_pred) ** 2, dim=0)
    band_rmse = torch.sqrt(band_mse)
    return torch.mean(band_rmse)


def combined_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
                  alpha: float = 0.0, beta: float = 0.0) -> torch.Tensor:
    """组合损失函数"""
    loss_rmse = spectral_rmse_loss(y_true, y_pred)
    loss_sam = spectral_angle_loss(y_true, y_pred)
    loss_band = band_rmse_loss(y_true, y_pred)

    return loss_rmse + alpha * loss_sam + beta * loss_band


from torch.cuda.amp import autocast

def train_epoch(model, loader, optimizer, alpha, beta, device, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                outputs = model(batch_x)
                loss = combined_loss(batch_y, outputs, alpha, beta)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_x)
            loss = combined_loss(batch_y, outputs, alpha, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)



@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             alpha: float, beta: float, device: torch.device) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    total_sam = 0.0
    total_band = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)

        # 计算各个损失
        loss = combined_loss(batch_y, outputs, alpha, beta)
        rmse = spectral_rmse_loss(batch_y, outputs)
        sam = spectral_angle_loss(batch_y, outputs)
        band = band_rmse_loss(batch_y, outputs)

        total_loss += loss.item()
        total_rmse += rmse.item()
        total_sam += sam.item()
        total_band += band.item()
        n_batches += 1

        # 收集预测用于后续分析
        if len(all_preds) < 1000:  # 限制内存使用
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    results = {
        'loss': total_loss / max(n_batches, 1),
        'rmse': total_rmse / max(n_batches, 1),
        'sam': total_sam / max(n_batches, 1),
        'band': total_band / max(n_batches, 1),
    }

    return results


def train_and_eval_single_config(
        X_train: np.ndarray, Y_train: np.ndarray,
        X_val: np.ndarray, Y_val: np.ndarray,
        alpha: float, beta: float, epochs: int,
        batch_size: int = 128, use_amp: bool = True
) -> Dict[str, Any]:
    """训练并评估单个配置"""
    start_time = time.time()

    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(Y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(Y_val)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 设为0以避免多进程问题
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    # 创建模型
    model = M2HSWIRModel(
        input_dim=X_train.shape[1],
        output_dim=Y_train.shape[1],
        alpha=alpha,
        beta=beta
    ).to(device)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    # 学习率调度器 - 修复：移除verbose参数
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # 混合精度训练 (如果可用)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    # 训练循环
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    max_patience = 5

    train_history = []
    val_history = []

    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, alpha, beta, device, scaler)
        val_results = evaluate(model, val_loader, alpha, beta, device)
        val_loss = val_results['loss']

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_results = val_results.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= max_patience:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

        train_history.append(train_loss)
        val_history.append(val_loss)

    # 最终评估
    final_results = evaluate(model, val_loader, alpha, beta, device)

    # 清理GPU内存
    if device.type == "cuda":
        torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time

    return {
        'alpha': alpha,
        'beta': beta,
        'rmse_all': final_results['rmse'],
        'sam': final_results['sam'],
        'rmse_band': final_results['band'],
        'val_loss': final_results['loss'],
        'epochs_used': epoch + 1,
        'best_epoch': best_epoch + 1,
        'best_val_loss': best_val_loss,
        'training_time': elapsed_time,
        'train_history': train_history[:10],  # 只保存前10个epoch的历史
        'val_history': val_history[:10],
    }


# ====== 网格搜索主函数 ======
def run_grid_search(
        alphas: List[float], betas: List[float],
        X_train: np.ndarray, Y_train: np.ndarray,
        X_val: np.ndarray, Y_val: np.ndarray,
        epochs: int, stage_name: str,
        batch_size: int = 128
) -> List[Dict[str, Any]]:
    """运行网格搜索"""
    results = []
    total_tasks = len(alphas) * len(betas)
    completed = 0

    print(f"\n[{stage_name}] Starting {total_tasks} configurations...")
    print(f"[{stage_name}] Alpha values: {alphas}")
    print(f"[{stage_name}] Beta values: {betas}")

    for alpha, beta in itertools.product(alphas, betas):
        completed += 1
        print(f"\n[{stage_name}] ({completed}/{total_tasks}) "
              f"alpha={alpha:.3f}, beta={beta:.3f}")

        try:
            result = train_and_eval_single_config(
                X_train, Y_train, X_val, Y_val,
                alpha, beta, epochs, batch_size
            )

            print(f"    -> RMSE: {result['rmse_all']:.6f}, "
                  f"SAM: {result['sam']:.6f}, "
                  f"Band-RMSE: {result['rmse_band']:.6f}, "
                  f"Epochs: {result['epochs_used']}")

            result['stage'] = stage_name
            results.append(result)

        except Exception as e:
            print(f"    -> ERROR: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'stage': stage_name,
                'alpha': alpha,
                'beta': beta,
                'rmse_all': float('inf'),
                'sam': float('inf'),
                'rmse_band': float('inf'),
                'val_loss': float('inf'),
                'epochs_used': 0,
                'error': str(e)
            })

        # 定期清理内存
        if device.type == "cuda" and completed % 3 == 0:
            torch.cuda.empty_cache()

    return results


def sort_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按照 rmse_all → rmse_band → sam 排序（越小越好）"""
    valid_results = [r for r in results if r['rmse_all'] != float('inf')]

    if not valid_results:
        return results

    return sorted(
        valid_results,
        key=lambda r: (r['rmse_all'], r['rmse_band'], r['sam'])
    )


def pick_top2(results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """选择前两个最佳配置"""
    sorted_res = sort_results(results)

    if not sorted_res:
        raise ValueError("No valid results found!")

    best1 = sorted_res[0]

    # 寻找第二个不同的配置
    best2 = None
    for r in sorted_res[1:]:
        if r['alpha'] != best1['alpha'] or r['beta'] != best1['beta']:
            best2 = r
            break

    if best2 is None:
        best2 = best1

    print("\n" + "=" * 60)
    print(f"[{results[0]['stage'] if results else 'coarse'}] Top Configurations:")
    print(f"Top-1: alpha={best1['alpha']:.3f}, beta={best1['beta']:.3f}, "
          f"RMSE={best1['rmse_all']:.6f}, SAM={best1['sam']:.6f}")
    print(f"Top-2: alpha={best2['alpha']:.3f}, beta={best2['beta']:.3f}, "
          f"RMSE={best2['rmse_all']:.6f}, SAM={best2['sam']:.6f}")
    print("=" * 60)

    return best1, best2


def build_fine_grid_between(best1: Dict[str, Any], best2: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """在两个最佳点之间构建细网格"""
    a1, a2 = best1['alpha'], best2['alpha']
    b1, b2 = best1['beta'], best2['beta']

    # 确定范围
    alpha_min, alpha_max = min(a1, a2), max(a1, a2)
    beta_min, beta_max = min(b1, b2), max(b1, b2)

    # 扩展范围20%以避免局部最优
    alpha_range = alpha_max - alpha_min
    beta_range = beta_max - beta_min

    if alpha_range > 0:
        alpha_min = max(0.0, alpha_min - alpha_range * 0.2)
        alpha_max = min(1.0, alpha_max + alpha_range * 0.2)

    if beta_range > 0:
        beta_min = max(0.0, beta_min - beta_range * 0.2)
        beta_max = min(3.0, beta_max + beta_range * 0.2)

    # 创建细网格
    if alpha_min == alpha_max:
        alphas_fine = [alpha_min]
    else:
        alphas_fine = np.linspace(alpha_min, alpha_max, N_ALPHA_FINE).round(3).tolist()

    if beta_min == beta_max:
        betas_fine = [beta_min]
    else:
        betas_fine = np.linspace(beta_min, beta_max, N_BETA_FINE).round(3).tolist()

    print(f"\n[fine] Constructing fine grid around top configurations:")
    print(f"  Alpha range: [{alpha_min:.3f}, {alpha_max:.3f}] -> {alphas_fine}")
    print(f"  Beta range: [{beta_min:.3f}, {beta_max:.3f}] -> {betas_fine}")

    return alphas_fine, betas_fine


def save_results(results: List[Dict[str, Any]], filename: str):
    """保存结果到JSON文件"""
    out_dir = MODEL_DIR / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # 将numpy数组转换为列表以便JSON序列化
    for r in results:
        if 'train_history' in r:
            r['train_history'] = [float(x) for x in r['train_history']]
        if 'val_history' in r:
            r['val_history'] = [float(x) for x in r['val_history']]

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {out_path}")
    return out_path


def main():
    """主函数"""
    print("=" * 70)
    print("GPU-Accelerated Grid Search for Loss Weights (alpha, beta)")
    print("=" * 70)

    # 1. 加载数据
    print("\n1. Loading data...")
    X_ms = np.load(LUT_DIR / "X_ms.npy")
    Y_hs = np.load(LUT_DIR / "Y_hs.npy")

    # 分割数据
    splits = split_simulated_lut(X_ms, Y_hs)
    X_train, Y_train = splits["train"]
    X_val, Y_val = splits["val"]

    print(f"   Training data: {X_train.shape}")
    print(f"   Validation data: {X_val.shape}")
    print(f"   Device: {device}")

    # 2. 粗搜索
    print("\n2. Starting coarse grid search...")
    coarse_results = run_grid_search(
        ALPHAS_COARSE, BETAS_COARSE,
        X_train, Y_train, X_val, Y_val,
        epochs=EPOCHS_COARSE,
        stage_name="coarse",
        batch_size=min(BATCH_SIZE, 256)  # 增大批处理大小以利用GPU
    )

    # 保存粗搜索结果
    save_results(coarse_results, "coarse_search_results.json")

    # 3. 选择前两个最佳配置
    best1, best2 = pick_top2(coarse_results)

    # 4. 细搜索
    print("\n3. Starting fine grid search...")
    alphas_fine, betas_fine = build_fine_grid_between(best1, best2)

    fine_batch_size = min(BATCH_SIZE * 4, 1024)
    fine_results = run_grid_search(
        alphas_fine, betas_fine,
        X_train, Y_train, X_val, Y_val,
        epochs=EPOCHS_FINE,
        stage_name="fine",
        batch_size=fine_batch_size,
    )
    # 保存细搜索结果
    save_results(fine_results, "fine_search_results.json")

    # 5. 合并结果并分析
    all_results = coarse_results + fine_results
    all_results_path = save_results(all_results, "all_search_results.json")

    # 6. 显示最佳配置
    global_best = sort_results(all_results)[0]

    print("\n" + "=" * 70)
    print("GLOBAL BEST CONFIGURATION:")
    print("=" * 70)
    print(f"  Alpha:        {global_best['alpha']:.3f}")
    print(f"  Beta:         {global_best['beta']:.3f}")
    print(f"  RMSE (All):   {global_best['rmse_all']:.6f}")
    print(f"  SAM:          {global_best['sam']:.6f}")
    print(f"  Band RMSE:    {global_best['rmse_band']:.6f}")
    print(f"  Validation Loss: {global_best.get('val_loss', 'N/A'):.6f}")
    print(f"  Epochs Used:  {global_best['epochs_used']}")

    # 修复：安全的格式化字符串
    training_time = global_best.get('training_time', 'N/A')
    if isinstance(training_time, (int, float)):
        print(f"  Training Time: {training_time:.1f}s")
    else:
        print(f"  Training Time: {training_time}")

    print("=" * 70)

    # 7. 显示前5名配置
    print("\nTop 5 Configurations:")
    print("-" * 70)
    top5 = sort_results(all_results)[:5]
    for i, r in enumerate(top5, 1):
        print(f"{i:2d}. Alpha={r['alpha']:.3f}, Beta={r['beta']:.3f}, "
              f"RMSE={r['rmse_all']:.6f}, SAM={r['sam']:.6f}, "
              f"Band-RMSE={r['rmse_band']:.6f}")

    print("\n" + "=" * 70)
    print("Grid search completed successfully!")
    print(f"Results saved to: {MODEL_DIR / 'search'}")
    print("=" * 70)

    # 8. 可选：生成可视化图表
    try:
        generate_visualization(all_results, MODEL_DIR / "search" / "visualization.png")
    except Exception as e:
        print(f"\nNote: Visualization generation failed: {e}")


def generate_visualization(results: List[Dict[str, Any]], save_path: Path):
    """生成搜索结果的可视化图表"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        # 过滤有效结果
        valid_results = [r for r in results if r['rmse_all'] != float('inf')]
        if not valid_results:
            return

        # 创建DataFrame
        df = pd.DataFrame(valid_results)

        # 设置图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Grid Search Results Analysis', fontsize=16, fontweight='bold')

        # 1. RMSE热图
        pivot_rmse = df.pivot_table(values='rmse_all', index='alpha', columns='beta', aggfunc='mean')
        sns.heatmap(pivot_rmse, annot=True, fmt='.4f', cmap='YlOrRd_r', ax=axes[0, 0])
        axes[0, 0].set_title('RMSE Heatmap')
        axes[0, 0].set_xlabel('Beta')
        axes[0, 0].set_ylabel('Alpha')

        # 2. SAM热图
        pivot_sam = df.pivot_table(values='sam', index='alpha', columns='beta', aggfunc='mean')
        sns.heatmap(pivot_sam, annot=True, fmt='.4f', cmap='YlOrRd_r', ax=axes[0, 1])
        axes[0, 1].set_title('SAM Heatmap')
        axes[0, 1].set_xlabel('Beta')
        axes[0, 1].set_ylabel('Alpha')

        # 3. 损失曲面
        ax = axes[0, 2]
        scatter = ax.scatter(df['alpha'], df['beta'], c=df['rmse_all'],
                             cmap='viridis', s=100, alpha=0.8)
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_title('Loss Surface (RMSE)')
        plt.colorbar(scatter, ax=ax)

        # 4. 参数与性能关系
        axes[1, 0].scatter(df['alpha'], df['rmse_all'], alpha=0.6)
        axes[1, 0].set_xlabel('Alpha')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Alpha vs RMSE')

        axes[1, 1].scatter(df['beta'], df['rmse_all'], alpha=0.6)
        axes[1, 1].set_xlabel('Beta')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Beta vs RMSE')

        # 5. 训练时间分布
        if 'training_time' in df.columns:
            axes[1, 2].hist(df['training_time'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 2].set_xlabel('Training Time (s)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Training Time Distribution')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved to: {save_path}")

    except ImportError as e:
        print(f"\nNote: Visualization requires matplotlib, pandas, and seaborn.")
        print(f"Install with: pip install matplotlib pandas seaborn")
    except Exception as e:
        print(f"\nNote: Visualization error: {e}")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # 运行主函数
    main()