# import os
# import sys
# from pathlib import Path
#
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import tensorflow as tf
# from tensorflow.keras.models import load_model
#
# # ====== 路径与工程导入 ======
# current_dir = os.path.dirname(os.path.abspath(__file__))   # .../src/m2h_swir/tools
# parent_dir = os.path.dirname(current_dir)                  # .../src/m2h_swir
# sys.path.insert(0, parent_dir)
#
# from config import REAL_DIR, MODEL_DIR, WL_MIN, WL_STEP  # type: ignore
# from model.losses import combined_loss  # type: ignore
#
#
# def build_wavelength_axis(output_dim: int):
#     """根据 WL_MIN, WL_STEP 和输出维度构造波长轴。"""
#     return WL_MIN + np.arange(output_dim) * WL_STEP
#
#
# def build_band_index_map(wavelengths_full: np.ndarray):
#     """
#     构造 {整数波长: 索引} 映射，方便按 850, 670 这类中心波长取值。
#     """
#     return {int(w): idx for idx, w in enumerate(wavelengths_full.astype(int))}
#
#
# def compute_vi_from_spectra(Y_full: np.ndarray,
#                             wavelengths_full: np.ndarray) -> dict:
#     """
#     给定 full 维光谱 (N, D) 和对应波长轴，计算 6 个氮敏感指数：
#       NDVI(850,670), NDRE(850,730), GNDVI(850,550),
#       OSAVI1510(850,1510), N870_1450(870,1450), N850_1510(850,1510)
#     返回：
#       dict(name -> (N,))
#     """
#     band_idx_map = build_band_index_map(wavelengths_full)
#     eps = 1e-8
#
#     def band(R, wl):
#         wl_int = int(wl)
#         if wl_int not in band_idx_map:
#             raise ValueError(f"目标波长 {wl_int} nm 不在 wavelengths_full 里")
#         return R[:, band_idx_map[wl_int]]
#
#     R850  = band(Y_full, 850)
#     R870  = band(Y_full, 870)
#     R670  = band(Y_full, 670)
#     R720  = band(Y_full, 720)
#     R550  = band(Y_full, 550)
#     R1450 = band(Y_full, 1450)
#     R1510 = band(Y_full, 1510)
#     R1645=band(Y_full, 1645)
#     R1715=band(Y_full,1715)
#     vi = {}
#     # 1) NDVI
#     vi["NDVI"] = (R850 - R670) / (R850 + R670 + eps)
#     # 2) NDRE
#     vi["NDRE"] = (R850 - R720) / (R850 + R720 + eps)
#     # 3) GNDVI (Green NDVI)
#     vi["GNDVI"] = (R850 - R550) / (R850 + R550 + eps)
#     # 4) OSAVI1510
#     vi["OSAVI1510"] = (1.0 + 0.16) * (R850 - R1510) / (R850 + R1510 + 0.16 + eps)
#     # 5) N1645_1715
#     vi["N1645_1715"] = (R1645 - R1715) / (R1645 + R1715 + eps)
#     # 6) N850_1510
#     vi["N850_1510"] = (R850 - R1510) / (R850 + R1510 + eps)
#
#     return vi
#
#
# def prepare_real_full_labels(model_path: Path):
#     """
#     和 eval_on_real 里类似：
#     - 用模型输出维度构造 400–2500 nm 波长轴
#     - 把 ASD 紧凑光谱 Y_asd_hs (N, n_valid) 填回 full 维 (N, D)
#       注意：这里假设 wavelength.npy 与你训练时使用的 valid 波段一致。
#     """
#     X_real = np.load(REAL_DIR / "X_uav_ms_corrected.npy")          # (N, 10)
#     Y_real_compact = np.load(REAL_DIR / "Y_asd_hs.npy")  # (N, n_valid)
#     wavelength_real = np.load(REAL_DIR / "wavelength.npy")  # (n_valid,)
#
#     # 用模型推断输出维度
#     custom_objects = {
#         "combined_loss": lambda y_true, y_pred: combined_loss(
#             y_true, y_pred, 0.0, 0.0
#         )
#     }
#     dummy = load_model(model_path, custom_objects=custom_objects, compile=False)
#     output_dim = dummy.output_shape[-1]
#     wavelengths_full = build_wavelength_axis(output_dim)
#
#     # 把 ASD 波段嵌回 full 轴
#     valid_mask = np.isin(wavelengths_full, wavelength_real)
#     if valid_mask.sum() != wavelength_real.shape[0]:
#         raise ValueError(
#             f"valid_mask True 数={valid_mask.sum()} != wavelength_real 长度={wavelength_real.shape[0]}"
#         )
#
#     Y_full = np.zeros((Y_real_compact.shape[0], output_dim), dtype=Y_real_compact.dtype)
#     Y_full[:, valid_mask] = Y_real_compact
#     return X_real, Y_full, wavelengths_full
#
#
# def plot_vi_scatter_for_model(model_name: str,
#                               out_path: Path = None):
#     """
#     对单个模型，在 REAL UAV–ASD 数据上画 VI 散点图：
#       x = ASD 真值
#       y = 模型预测
#       颜色按 stage 区分
#     """
#     model_path = MODEL_DIR / model_name
#     if not model_path.exists():
#         raise FileNotFoundError(f"模型不存在: {model_path}")
#
#     print(f"Loading REAL data and model: {model_path}")
#     X_real, Y_true_full, wavelengths_full = prepare_real_full_labels(model_path)
#     stages = np.load(REAL_DIR / "stages.npy")  # shape=(N,)
#     unique_stages = np.unique(stages)
#     # 加载模型
#     custom_objects = {
#         "combined_loss": lambda y_true, y_pred: combined_loss(
#             y_true, y_pred, 0.0, 0.0
#         )
#     }
#     model = load_model(model_path, custom_objects=custom_objects, compile=False)
#     Y_pred_full = model.predict(X_real, verbose=0)
#
#     # 计算 VI
#     VI_true = compute_vi_from_spectra(Y_true_full, wavelengths_full)
#     VI_pred = compute_vi_from_spectra(Y_pred_full, wavelengths_full)
#
#     vi_names = list(VI_true.keys())  # ['NDVI', 'NDRE', ...]
#     num_indices = len(vi_names)
#
#     # 子图布局
#     num_cols = 3
#     num_rows = (num_indices + num_cols - 1) // num_cols
#
#     plt.style.use("default")
#     plt.rcParams["font.family"] = "Times New Roman"
#
#     fig, axs = plt.subplots(num_rows, num_cols,
#                             figsize=(num_cols * 3.5, num_rows * 3.5),
#                             sharex=False, sharey=False)
#     axs = axs.flatten()
#
#     # 颜色列表（够用就行，不够可以再加）
#     color_list = ['#1f77b4', '#ff7f0e', '#2ca02c',
#                   '#d62728', '#9467bd', '#8c564b']
#
#     for i, name in enumerate(vi_names):
#         ax = axs[i]
#         y_true = VI_true[name].reshape(-1)
#         y_pred = VI_pred[name].reshape(-1)
#
#         r2 = r2_score(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         mae = mean_absolute_error(y_true, y_pred)
#
#         # 按 stage 上色
#         for j, stage in enumerate(unique_stages):
#             mask = stages == stage
#             ax.scatter(
#                 y_true[mask],
#                 y_pred[mask],
#                 label=str(stage),
#                 alpha=0.7,
#                 s=15,
#                 color=color_list[j % len(color_list)],
#             )
#
#         # 1:1 参考线
#         min_val = min(y_true.min(), y_pred.min())
#         max_val = max(y_true.max(), y_pred.max())
#         ax.plot([min_val, max_val], [min_val, max_val],
#                 linestyle="--", color="black", linewidth=1.0)
#
#         # 文本标注
#         ax.text(
#             0.95, 0.05,
#             f"R²={r2:.2f}\nMSE={mse:.4f}\nMAE={mae:.4f}",
#             transform=ax.transAxes,
#             fontsize=9,
#             verticalalignment="bottom",
#             horizontalalignment="right",
#             bbox=dict(boxstyle="round,pad=0.3",
#                       edgecolor="gray", facecolor="white"),
#         )
#
#         ax.set_xlabel("ASD Measured")
#         ax.set_ylabel("Model Predicted")
#         ax.set_title(f"({chr(97 + i)}) {name}")
#
#     # 删除多余子图（如果有的话）
#     for k in range(num_indices, len(axs)):
#         fig.delaxes(axs[k])
#
#     # 统一图例
#     fig.legend(
#         [str(s) for s in unique_stages],
#         loc="lower center",
#         ncol=min(len(unique_stages), 5),
#         fontsize=9,
#         bbox_to_anchor=(0.5, -0.02),
#     )
#
#     plt.tight_layout()
#
#     if out_path is None:
#         out_dir = MODEL_DIR / "figures"
#         out_dir.mkdir(parents=True, exist_ok=True)
#         out_path = out_dir / f"{Path(model_name).stem}_VI_scatter_real.jpg"
#
#     fig.savefig(out_path, dpi=600, bbox_inches="tight")
#     print(f"Saved VI scatter figure to: {out_path}")
#     plt.show()
#
#
# if __name__ == "__main__":
#     # 这里可以改成你要画的模型名，比如 LUT+微调版
#     # 也可以改成从命令行读取参数，这里先写死一个方便直接跑
#     plot_vi_scatter_for_model("m2h_swir_finetuned.keras")
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import math
# ====== 路径与工程导入 ======
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/m2h_swir/tools
parent_dir = os.path.dirname(current_dir)  # .../src/m2h_swir
sys.path.insert(0, parent_dir)

from config import REAL_DIR, MODEL_DIR, WL_MIN, WL_STEP, WL_MAX  # type: ignore
from model.models import build_m2h_swir_model, build_simple_cnn  # type: ignore


def build_wavelength_axis(output_dim: int):
    """根据 WL_MIN, WL_STEP 和输出维度构造波长轴。"""
    return WL_MIN + np.arange(output_dim) * WL_STEP


def build_band_index_map(wavelengths_full: np.ndarray):
    """
    构造 {整数波长: 索引} 映射，方便按 850, 670 这类中心波长取值。
    """
    return {int(w): idx for idx, w in enumerate(wavelengths_full.astype(int))}


def compute_vi_from_spectra(Y_full: np.ndarray,
                            wavelengths_full: np.ndarray) -> dict:
    """
    给定 full 维光谱 (N, D) 和对应波长轴，计算 6 个氮敏感指数：
      NDVI(850,670), NDRE(850,730), GNDVI(850,550),
      OSAVI1510(850,1510), N1645_1715(1645,1715), N850_1510(850,1510)
    返回：
      dict(name -> (N,))
    """
    band_idx_map = build_band_index_map(wavelengths_full)
    eps = 1e-8

    def band(R, wl):
        wl_int = int(wl)
        if wl_int not in band_idx_map:
            raise ValueError(f"目标波长 {wl_int} nm 不在 wavelengths_full 里")
        return R[:, band_idx_map[wl_int]]
    R490 = band(Y_full,490)
    R850 = band(Y_full, 850)
    R670 = band(Y_full, 670)
    R720 = band(Y_full, 720)  # NDRE 使用 720nm
    R750 = band(Y_full,749)
    R550 = band(Y_full, 490)
    R1510 = band(Y_full, 1510)
    R1650= band(Y_full,1650)

    vi = {}
    # 1) NDVI
    vi["NDVI"] = (R850 - R670) / (R850 + R670 + eps)
    # 2) NDRE
    vi["NDRE"] = (R850 - R720) / (R850 + R720 + eps)
    # 3) GNDVI (Green NDVI)
    vi["RECI"] = (R850 - R720)
    # 4) OSAVI1510
    vi["OSAVI1510"] = (1.0 + 0.16) * (R850 - R1510) / (R850 + R1510 + 0.16 + eps)
    # 5) NDNI
    #vi["GNDVI"]= (R750-R550)/(R750+R550+eps)
    vi["NDNI"] = (np.log(1 / R1510) - np.log(1 / R1650)) / (np.log(1 / R1510) + np.log(1 / R1650)+eps)
    # 6) N850_1510
    vi["N850_1510"] = (R850 - R1510) / (R850 + R1510 + eps)

    return vi


def prepare_real_full_labels(output_dim: int = None):
    """
    准备真实数据：
    - 用输出维度构造 400–2500 nm 波长轴
    - 把 ASD 紧凑光谱 Y_asd_hs (N, n_valid) 填回 full 维 (N, D)
    """
    X_real = np.load(REAL_DIR / "X_uav_ms.npy")  # (N, 10)
    Y_real_compact = np.load(REAL_DIR / "Y_asd_hs.npy")  # (N, n_valid)
    wavelength_real = np.load(REAL_DIR / "wavelength.npy")  # (n_valid,)

    # 如果未提供 output_dim，根据 WL_MIN/WL_MAX/WL_STEP 计算
    if output_dim is None:
        output_dim = int((WL_MAX - WL_MIN) / WL_STEP) + 1

    wavelengths_full = build_wavelength_axis(output_dim)

    # 把 ASD 波段嵌回 full 轴
    valid_mask = np.isin(wavelengths_full, wavelength_real)
    if valid_mask.sum() != wavelength_real.shape[0]:
        raise ValueError(
            f"valid_mask True 数={valid_mask.sum()} != wavelength_real 长度={wavelength_real.shape[0]}"
        )

    Y_full = np.zeros((Y_real_compact.shape[0], output_dim), dtype=Y_real_compact.dtype)
    Y_full[:, valid_mask] = Y_real_compact
    return X_real, Y_full, wavelengths_full


def load_pytorch_model(model_path: Path, input_dim: int, output_dim: int, device: torch.device):
    """
    加载 PyTorch 模型并推断模型类型
    """
    # 从文件名推断模型类型
    model_name = model_path.stem.lower()
    if "simple_cnn" in model_name:
        model_type = "simple_cnn"
    elif "m2h" in model_name:
        model_type = "m2h"
    else:
        # 默认使用 m2h
        model_type = "m2h"

    # 构建模型
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
        raise ValueError(f"Unknown model type inferred: {model_type}")

    # 加载权重
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def plot_vi_scatter_for_model(model_name: str,
                              out_path: Path = None):
    """
    对单个 PyTorch 模型，在 REAL UAV-ASD 数据上画 VI 散点图：
      x = ASD 真值
      y = 模型预测
      颜色按 stage 区分
    """
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    print(f"Loading REAL data and PyTorch model: {model_path}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 先准备数据以获取维度信息
    X_real, Y_true_full, wavelengths_full = prepare_real_full_labels()
    output_dim = Y_true_full.shape[1]
    input_dim = X_real.shape[1]

    # 加载 PyTorch 模型
    model = load_pytorch_model(model_path, input_dim, output_dim, device)

    # 预测
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_real).float().to(device)
        Y_pred_tensor = model(X_tensor)
        Y_pred_full = Y_pred_tensor.cpu().numpy()

    # 加载阶段标签
    stages_file = REAL_DIR / "stages.npy"
    if not stages_file.exists():
        raise FileNotFoundError(f"Stage file not found: {stages_file}")
    stages = np.load(stages_file)  # shape=(N,)
    unique_stages = np.unique(stages)

    # 计算 VI
    VI_true = compute_vi_from_spectra(Y_true_full, wavelengths_full)
    VI_pred = compute_vi_from_spectra(Y_pred_full, wavelengths_full)

    vi_names = list(VI_true.keys())  # ['NDVI', 'NDRE', ...]
    num_indices = len(vi_names)

    # 子图布局
    num_cols = 3
    num_rows = (num_indices + num_cols - 1) // num_cols

    plt.style.use("default")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"  # 数学字体

    fig, axs = plt.subplots(num_rows, num_cols,
                            figsize=(num_cols * 3.5, num_rows * 3.5),
                            sharex=False, sharey=False)

    # 处理子图轴
    if num_rows == 1:
        axs = axs.reshape(1, -1)
    axs = axs.flatten()

    # 颜色列表（按阶段）
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c',
                  '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#7f7f7f', '#bcbd22']

    for i, name in enumerate(vi_names):
        ax = axs[i]
        y_true = VI_true[name].reshape(-1)
        y_pred = VI_pred[name].reshape(-1)

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # 按 stage 上色
        for j, stage in enumerate(unique_stages):
            mask = stages == stage
            ax.scatter(
                y_true[mask],
                y_pred[mask],
                label=str(stage) if i == 0 else "",  # 只在第一个图添加图例标签
                alpha=0.7,
                s=15,
                color=color_list[j % len(color_list)],
                edgecolors='none'
            )

        # 1:1 参考线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                linestyle="--", color="black", linewidth=1.0)

        # 文本标注
        ax.text(
            0.95, 0.05,
            f"R²={r2:.2f}\nRMSE={rmse:.4f}\nMAE={mae:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3",
                      edgecolor="gray", facecolor="white"),
        )

        ax.set_xlabel("ASD Measured", fontsize=10)
        ax.set_ylabel("Model Predicted", fontsize=10)
        ax.set_title(f"({chr(97 + i)}) {name}", fontsize=11, fontweight='bold')

        # 设置坐标轴范围
        padding = (max_val - min_val) * 0.05
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.set_ylim(min_val - padding, max_val + padding)

        # 网格线
        ax.grid(True, alpha=0.3, linestyle='--')

    # 删除多余子图（如果有的话）
    for k in range(num_indices, len(axs)):
        fig.delaxes(axs[k])

    # 统一图例
    if len(unique_stages) > 0:
        # 获取第一个子图的图例句柄和标签
        handles, labels = axs[0].get_legend_handles_labels()
        if handles:  # 确保有图例
            fig.legend(
                handles, labels,
                loc="lower center",
                ncol=min(len(unique_stages), 5),
                fontsize=9,
                bbox_to_anchor=(0.5, -0.02),
                title="Growth Stage"
            )

    plt.tight_layout()

    if out_path is None:
        out_dir = MODEL_DIR / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(model_name).stem}_VI_scatter_real.jpg"

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Saved VI scatter figure to: {out_path}")
    plt.show()

    # 打印汇总统计信息
    print("\n" + "=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)
    for name in vi_names:
        y_true = VI_true[name].reshape(-1)
        y_pred = VI_pred[name].reshape(-1)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"{name:12s}: R²={r2:.3f}, RMSE={rmse:.4f}")


if __name__ == "__main__":
    # 可以修改这里来绘制不同模型的散点图
    models_to_plot = [
        "m2h_swir_finetuned_new_head.pt", # LUT+微调版
        # "m2h_data_driven_new_head.pt",  # 纯数据驱动版
        # "simple_cnn_data_driven_new_head.pt",  # SimpleCNN基线
         "m2h_swir_lut_new_head.pt",  # LUT预训练版
    ]

    for model_name in models_to_plot:
        try:
            print(f"\n{'=' * 60}")
            print(f"Plotting VI scatter for: {model_name}")
            print(f"{'=' * 60}")
            plot_vi_scatter_for_model(model_name)
        except Exception as e:
            print(f"Error plotting {model_name}: {e}")
            continue

    print("\nAll plots completed!")