"""Parameter ranges and phenological stage configuration for PROSAIL-PRO."""
from dataclasses import dataclass
from typing import Dict


@dataclass
class ParameterRange:
    name: str
    min_val: float
    max_val: float


@dataclass
class StageConfig:
    name: str
    ranges: Dict[str, ParameterRange]
    n_samples: int


def default_stage_configs() -> Dict[str, StageConfig]:
    """
    Return PROSAIL configs for 5 phenological stages.
    LAI & Cab use stage-specific ranges.
    Other parameters use global biological ranges.
    """

    stages: Dict[str, StageConfig] = {}

    # === 统一样本数（可改 20000 / 30000 / 50000）
    N_SAMPLES = 20000

    # === 全局范围 ===
    # 更新全局范围（所有阶段共享的基础调整）
    global_ranges = {
        "N": ParameterRange("N", 1.2, 1.8),
        "Car": ParameterRange("Car", 0.0, 15.0),
        "Cw": ParameterRange("Cw", 0.015, 0.04),  # 统一提高下限
        "Cm": ParameterRange("Cm", 0.004, 0.015),  # 收紧
        "Cp": ParameterRange("Cp", 0.0, 0.0004),
        "Cbc": ParameterRange("Cbc", 0.0, 0.006),
        "ALA": ParameterRange("ALA", 40.0, 80.0),  # 稍偏直立
        "Psoil": ParameterRange("Psoil", 0.2, 0.5),  # 所有阶段统一，不再 0.6–1.0 这种极端
        "Rsoil": ParameterRange("Rsoil", 0.1, 0.4),  # 土壤不再特别亮
        "hspot": ParameterRange("hspot", 0.02, 0.08),
    }

    # === 5阶段范围===
    stage_specific = {
        '2022_Booting': {
            'LAI': (0.1, 4.8),
            'Cab': (20, 55),
            'Cw': (0.02, 0.048),
            'Psoil':(0.15,0.5),
        },
        '2022_Flowering': {
            'LAI': (2.0, 6.0),
            'Cab': (35, 75),
            'Cw': (0.01, 0.04),
            'Psoil': (0.25, 0.62),
        },
        '2023_Booting': {
            'LAI': (0.1, 4.5),
            'Cab': (20, 55),
            'Psoil': (0.35, 0.60),
        },
        '2023_Flowering': {
            'LAI': (2.0, 6.0),
            'Cab': (35, 80),
        },
        '2023_Heading': {
            'LAI': (0.5, 4.5),
            'Cab': (20, 70),
            'Cw': (0.015, 0.028),
            'Psoil':(0.25,0.5)
        },
    }

    for stage_name, sr in stage_specific.items():
        # 1. 先复制一份全局范围
        ranges = dict(global_ranges)

        # 2. 用该阶段的专属范围覆盖全局范围
        #    sr 里写了哪些参数（LAI、Cab、Cw、Cm、Psoil、Rsoil…）
        #    就全部覆盖掉 ranges 里的对应项
        for param_name, (vmin, vmax) in sr.items():
            ranges[param_name] = ParameterRange(param_name, vmin, vmax)

        # 3. 生成 StageConfig
        stages[stage_name] = StageConfig(
            name=stage_name,
            n_samples=N_SAMPLES,
            ranges=ranges
        )

    return stages


# 修改 get_stage_mixing_params 函数：

def get_stage_mixing_params(stage_name):
    """返回阶段特定的混合参数"""
    params = {
        'k_ext': 0.5,
        'shadow_frac_range': (0.2, 0.5),
        'max_bg_frac': 0.7,
        'min_bg_frac': 0.1,  # 新增：最小背景比例
    }

    if 'Booting' in stage_name:
        # 拔节期：允许更低的最小背景比例
        params['max_bg_frac'] = 0.6
        params['min_bg_frac'] = 0.08  # 允许更低的背景
        params['shadow_frac_range'] = (0.15, 0.4)
    elif 'Heading' in stage_name:
        params['k_ext'] = 0.55
        params['max_bg_frac'] = 0.6
        params['min_bg_frac'] = 0.12
    elif 'Flowering' in stage_name:
        params['max_bg_frac'] = 0.5
        params['min_bg_frac'] = 0.15  # 开花期最小背景稍高
        params['shadow_frac_range'] = (0.1, 0.35)

    return params