import numpy as np
from pyDOE2 import lhs
import prosail
from pathlib import Path
import sys
import os

# === 路径与导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))       # .../src/m2h_swir/prosail_sim
parent_dir = os.path.dirname(current_dir)                      # .../src/m2h_swir
sys.path.insert(0, parent_dir)

from prosail_config import default_stage_configs, get_stage_mixing_params
from prosail_config import default_stage_configs
from srf import load_srf, convolve_to_multispectral
from config import WL_MIN, WL_MAX, WL_STEP, LUT_DIR


# === 1. 根据 LAI 计算 α（土壤）和 β（阴影） ===

def compute_alpha_beta_from_lai(
    lai_array,
    k_ext: float = 0.5,
    shadow_frac_range=(0.2, 0.5),
    max_bg_frac: float = 0.7,
):
    """
    lai_array: (N,) PROSAIL 采样得到的 LAI
    k_ext: Beer-Lambert 消光系数，行作物常见 0.4–0.7，这里默认 0.5
    shadow_frac_range: 背景中阴影比例 s 的范围
    max_bg_frac: 背景总比例的上限，避免极端情况下几乎全是土壤/阴影

    返回:
      alpha: (N,1) 土壤比例
      beta:  (N,1) 阴影比例
    """
    lai_array = np.asarray(lai_array)

    # 背景总比例 f_bg = exp(-k * LAI)
    f_bg = np.exp(-k_ext * lai_array)  # (N,)

    # 限制背景比例上限
    f_bg = np.minimum(f_bg, max_bg_frac)

    # 背景中阴影比例 s
    s = np.random.uniform(
        shadow_frac_range[0],
        shadow_frac_range[1],
        size=lai_array.shape
    )  # (N,)

    alpha = (1.0 - s) * f_bg   # 土壤
    beta = s * f_bg            # 阴影

    alpha = alpha.reshape(-1, 1)
    beta = beta.reshape(-1, 1)
    return alpha, beta


# === 2. 用 α/β 在高光谱空间做混合 ===

def mix_canopy_soil_shadow_hs(
    R_canopy_hs: np.ndarray,
    R_soil_hs: np.ndarray,
    R_shadow_hs: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
):
    """
    R_canopy_hs: (N, n_wl)
    R_soil_hs:   (n_wl,)
    R_shadow_hs: (n_wl,)
    alpha, beta: (N,1)

    返回:
      R_mix_hs: (N, n_wl)
    """
    soil = R_soil_hs.reshape(1, -1)      # (1, n_wl)
    shadow = R_shadow_hs.reshape(1, -1)  # (1, n_wl)

    R_mix = (1.0 - alpha - beta) * R_canopy_hs + alpha * soil + beta * shadow
    return np.clip(R_mix, 0.0, 1.0)


# === 3. UAV 风格扰动（在多光谱空间上做波段相关扰动） ===

def apply_uav_style_perturbations_ms(
    X_ms: np.ndarray,
    illum_std: float = 0.05,
    bias_std: float = 0.01,
    noise_std: float = 0.005,
):
    """
    X_ms: (N, n_bands) 已经是混合后的多光谱（接近 UAV 像元）
    illum_std: 波段相关的乘性扰动标准差
    bias_std:  band 级偏置噪声（标定误差）
    noise_std: 传感器高斯噪声
    """
    N, n_bands = X_ms.shape
    X = X_ms.copy()

    # 波段相关光照变化：每个样本、每个 band 一个乘性扰动（均值1，std=illum_std）
    illum = np.random.normal(1.0, illum_std, size=(N, n_bands))
    X = X * illum

    # band 级偏置噪声（标定 bias）
    bias = np.random.normal(0.0, bias_std, size=(N, n_bands))
    X = X + bias

    # 高斯噪声（传感器 noise）
    noise = np.random.normal(0.0, noise_std, size=(N, n_bands))
    X = X + noise

    return np.clip(X, 0.0, 1.0)


# === 4. PROSAIL 模拟单个 stage 的高光谱 + 返回 LAI 数组 ===

def simulate_stage_spectra(stage_cfg):
    wl = np.arange(WL_MIN, WL_MAX + 1, WL_STEP)
    n_samples = stage_cfg.n_samples

    param_names = list(stage_cfg.ranges.keys())
    n_params = len(param_names)

    lhs_unit = lhs(n_params, samples=n_samples)

    params = {}
    for i, name in enumerate(param_names):
        pr = stage_cfg.ranges[name]
        params[name] = pr.min_val + lhs_unit[:, i] * (pr.max_val - pr.min_val)

    refl_list = []
    for i in range(n_samples):
        refl = prosail.run_prosail(
            n=params["N"][i],
            cab=params["Cab"][i],
            car=params["Car"][i],
            cw=params["Cw"][i],
            cm=params["Cm"][i],
            lai=params["LAI"][i],
            prot=params["Cp"][i],      # 蛋白质
            cbc=params["Cbc"][i],      # 棕色组分
            lidfa=params["ALA"][i],    # 叶倾角
            psoil=params["Psoil"][i],
            rsoil=params["Rsoil"][i],
            hspot=params["hspot"][i],
            cbrown=0.0,
            tts=30,
            tto=10,
            psi=0,
            prospect_version="PRO",
        )
        refl_list.append(refl)

    refl = np.vstack(refl_list)       # (n_samples, n_wl)
    lai_array = params["LAI"]         # (n_samples,)

    return refl, wl, lai_array


# === 5. 生成整体 LUT：多生育期 + HS 空间混合 + UAV 风格扰动 ===

def generate_full_lut(
    add_noise: bool = True,
    noise_std: float = 0.005,
    use_uav_mixing: bool = True,
):
    """
    生成“真实 UAV 风格”的 LUT：
      - PROSAIL-PRO 生成纯冠层高光谱
      - 用实测土壤/阴影高光谱在 HS 空间做混合（LAI 驱动 α/β）
      - SRF 卷积到多光谱
      - 在 MS 空间加入 UAV 风格扰动（illum + bias + noise）
    """
    stage_cfgs = default_stage_configs()
    srf = load_srf()

    # data/ancillary 下读取实测土壤/阴影高光谱（已插值到 400–2500nm）
    root_dir = Path(parent_dir).parent          # .../m2h_swir_github
    data_dir = root_dir / "data"
    ancillary_dir = data_dir / "ancillary"
    ancillary_dir.mkdir(parents=True, exist_ok=True)

    soil_hs_path = ancillary_dir / "soil_hs.npy"
    if not soil_hs_path.exists():
        raise FileNotFoundError(
            f"未找到土壤高光谱文件 {soil_hs_path}，请先生成或拷贝 soil_hs.npy （形状为 (n_wl,)）再运行 generate_full_lut。"
        )
    soil_hs = np.load(soil_hs_path)  # (n_wl,)

    shadow_hs_path = ancillary_dir / "shadow_hs.npy"
    if shadow_hs_path.exists():
        shadow_hs = np.load(shadow_hs_path)
        # 确保阴影足够暗
        shadow_hs = np.minimum(shadow_hs, 0.15 * soil_hs)  # 添加上限限制
    else:
        # 使用更暗的阴影
        shadow_hs = 0.1 * soil_hs  # 从0.3改为0.1

    all_hs = []
    all_ms = []
    stage_labels = []
    wl = None

    for name, cfg in stage_cfgs.items():
        print(f"Simulating stage: {name} with {cfg.n_samples} samples")

        # 1) PROSAIL 纯冠层高光谱 + LAI
        refl_hs_canopy, wl, lai_array = simulate_stage_spectra(cfg)

        # 2) LAI→α/β
        if use_uav_mixing:
            alpha, beta = compute_alpha_beta_from_lai(
                lai_array,
                k_ext=0.5,
                shadow_frac_range=(0.2, 0.5),
                max_bg_frac=0.7,
            )
            # 3) 在高光谱空间混合
            refl_hs_mix = mix_canopy_soil_shadow_hs(
                refl_hs_canopy, soil_hs, shadow_hs, alpha, beta
            )
        else:
            refl_hs_mix = refl_hs_canopy

        # 4) 卷积成多光谱
        ms = convolve_to_multispectral(refl_hs_mix, wl, srf)

        # 5) UAV 风格扰动（illum + bias + noise）
        if add_noise:
            ms = apply_uav_style_perturbations_ms(
                ms,
                illum_std=0.05,
                bias_std=0.01,
                noise_std=noise_std,
            )

        all_hs.append(refl_hs_mix)   # 注意：这里可以保存混合后的 HS
        all_ms.append(ms)
        stage_labels.extend([name] * cfg.n_samples)

    Y_hs = np.vstack(all_hs)        # (N_total, n_wl)
    X_ms = np.vstack(all_ms)        # (N_total, n_bands)
    stage_labels = np.array(stage_labels)

    LUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(LUT_DIR / "X_ms.npy", X_ms)
    np.save(LUT_DIR / "Y_hs.npy", Y_hs)
    np.save(LUT_DIR / "wl_hs.npy", wl)
    np.save(LUT_DIR / "stage_labels.npy", stage_labels)

    print(f"UAV-style LUT generated in {LUT_DIR}")
    print("X_ms shape:", X_ms.shape)
    print("Y_hs shape:", Y_hs.shape)

    return {"X_ms": X_ms, "Y_hs": Y_hs, "wl_hs": wl}


if __name__ == "__main__":
    generate_full_lut()
