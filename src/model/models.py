"""
Model architectures for multispectral-to-hyperspectral reconstruction.

包含两个主要模型：
- SimpleCNN: 简单的一维 CNN baseline
- M2H_SWIR_Model: 完整的 M2H-SWIR 模型（多尺度卷积 + SE + 残差 + 可选 Transformer）

注意：
- output_dim 由外部传入，可以是 LUT 全谱 (2101)，也可以是 ASD 紧凑光谱 (例如 1670)。
- 本文件主要负责“结构”，具体 loss 在训练脚本里可以重写。
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model.losses import combined_loss, spectral_angle_mapper,spectral_rmse  # 仍保留，以便 LUT 训练等场景使用
from config import DEFAULT_ALPHA, DEFAULT_BETA, LEARNING_RATE


# === Squeeze-and-Excitation block ===
class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // ratio)
        self.fc2 = nn.Linear(channels // ratio, channels)

    def forward(self, x):
        # x: (B, C, L)
        b, c, l = x.shape
        y = self.global_pool(x).view(b, c)       # (B, C)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))           # (B, C)
        y = y.view(b, c, 1)                      # (B, C, 1)
        return x * y


# === Transformer block (1D) ===
class TransformerBlock1D(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # 输入 (B, L, D)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        x = self.dropout(x)
        return x


# === Residual block ===
class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        shortcut = x
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        y = F.relu(y + shortcut)
        return y


# === Simple baseline CNN ===
class SimpleCNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """
        input_dim: 输入多光谱波段数 (bands)
        output_dim: 输出光谱维度 (可以是 LUT 全谱，也可以是 ASD 紧凑光谱)
        """
        super().__init__()
        # 输入格式 (B, bands) -> (B, 1, bands)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # x: (B, bands)
        x = x.unsqueeze(1)              # (B, 1, bands)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x).squeeze(-1)  # (B, 64)
        x = F.relu(self.fc1(x))        # (B, 256)
        x = self.fc2(x)                # (B, output_dim)
        return x


def build_simple_cnn(
    input_shape: Tuple[int],
    output_dim: int,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
):
    """
    返回:
      model: SimpleCNN 实例
      optimizer: Adam 优化器
      loss_fn: 一个包装了 alpha/beta 的损失函数（基于 combined_loss）
    在纯数据驱动紧凑光谱场景下，你可以在训练脚本里覆盖 loss_fn。
    """
    input_dim = input_shape[0]
    model = SimpleCNN(input_dim=input_dim, output_dim=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def loss_fn(y_pred, y_true):
        # 默认：使用全谱 combined_loss
        return combined_loss(y_true, y_pred, alpha, beta)

    return model, optimizer, loss_fn


# === Full M2H-SWIR model ===
class M2H_SWIR_Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_multiscale_conv: bool = True,
        use_se: bool = True,
        use_residual: bool = True,
        use_transformer: bool = True,
        num_heads: int = 4,
        d_model: int = 256,
        ff_dim: int = 512,
        num_transformer_layers: int = 2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.use_multiscale_conv = use_multiscale_conv
        self.use_se = use_se
        self.use_residual = use_residual
        self.use_transformer = use_transformer

        in_channels = 1  # (B, 1, bands)

        # --- 多尺度卷积 ---
        if use_multiscale_conv:
            self.branch1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
            self.branch2 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
            self.branch3 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
            conv_out_channels = 64 * 3
        else:
            self.single_conv = nn.Conv1d(in_channels, 128, kernel_size=5, padding=2)
            conv_out_channels = 128

        # --- SE 注意力 ---
        if use_se:
            self.se = SqueezeExcite1D(conv_out_channels)

        # --- 残差块 ---
        if use_residual:
            self.res_block1 = ResidualBlock1D(conv_out_channels)
            self.res_block2 = ResidualBlock1D(conv_out_channels)
            residual_out_channels = conv_out_channels
        else:
            self.res_conv = nn.Conv1d(conv_out_channels, conv_out_channels,
                                      kernel_size=3, padding=1)
            residual_out_channels = conv_out_channels

        # --- Transformer encoder 前的通道调整 ---
        if use_transformer:
            self.to_dmodel = nn.Conv1d(residual_out_channels, d_model, kernel_size=1)
            self.transformer_layers = nn.ModuleList([
                TransformerBlock1D(d_model=d_model, num_heads=num_heads,
                                   ff_dim=ff_dim, dropout=0.1)
                for _ in range(num_transformer_layers)
            ])
            trans_out_channels = d_model
        else:
            self.to_256 = nn.Conv1d(residual_out_channels, 256, kernel_size=1)
            trans_out_channels = 256

        # --- 下游卷积与压缩 ---
        self.conv_local = nn.Conv1d(trans_out_channels, 256, kernel_size=3, padding=1)
        self.conv_out = nn.Conv1d(256, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, bands)
        x = x.unsqueeze(1)  # (B, 1, L)

        # 多尺度卷积
        if self.use_multiscale_conv:
            b1 = F.relu(self.branch1(x))
            b2 = F.relu(self.branch2(x))
            b3 = F.relu(self.branch3(x))
            x = torch.cat([b1, b2, b3], dim=1)  # (B, 192, L)
        else:
            x = F.relu(self.single_conv(x))     # (B, 128, L)

        # SE
        if self.use_se:
            x = self.se(x)

        # 残差块
        if self.use_residual:
            x = self.res_block1(x)
            x = self.res_block2(x)
        else:
            x = F.relu(self.res_conv(x))

        # Transformer 或简单 1x1 conv
        if self.use_transformer:
            x = self.to_dmodel(x)        # (B, d_model, L)
            x = x.permute(0, 2, 1)       # (B, L, d_model)
            for layer in self.transformer_layers:
                x = layer(x)
            x = x.permute(0, 2, 1)       # (B, d_model, L)
        else:
            x = F.relu(self.to_256(x))   # (B, 256, L)

        # 插值到目标波段长度 output_dim（可对应 LUT 全谱或 ASD 紧凑光谱）
        x = F.interpolate(x, size=self.output_dim, mode="linear", align_corners=False)

        # 局部光谱特征 + 1x1 压缩通道
        x = F.relu(self.conv_local(x))   # (B, 256, output_dim)
        x = self.conv_out(x)             # (B, 1, output_dim)
        x = x.squeeze(1)                 # (B, output_dim)
        return x


def build_m2h_swir_model(
    input_shape: Tuple[int],
    output_dim: int,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    use_multiscale_conv: bool = True,
    use_se: bool = True,
    use_residual: bool = True,
    use_transformer: bool = True,
    num_heads: int = 4,
    d_model: int = 256,
    ff_dim: int = 512,
    num_transformer_layers: int = 2,
):
    """
    通用构建函数，既可以用于 LUT 训练 (output_dim=2101)，
    也可以用于 ASD 紧凑光谱训练 (例如 output_dim=1670)。

    返回:
      model: M2H_SWIR_Model
      optimizer: Adam
      loss_fn: 默认使用 combined_loss；在纯数据驱动紧凑训练脚本里可以覆盖 loss_fn。
    """
    input_dim = input_shape[0]

    model = M2H_SWIR_Model(
        input_dim=input_dim,
        output_dim=output_dim,
        use_multiscale_conv=use_multiscale_conv,
        use_se=use_se,
        use_residual=use_residual,
        use_transformer=use_transformer,
        num_heads=num_heads,
        d_model=d_model,
        ff_dim=ff_dim,
        num_transformer_layers=num_transformer_layers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def loss_fn(y_pred, y_true):
        return combined_loss(y_true, y_pred, alpha, beta)

    return model, optimizer, loss_fn

class SpectralConvNet(nn.Module):
    """
    纯数据驱动光谱卷积网络：
      - 输入：多光谱向量 (B, num_ms_bands)
      - 中间：MLP 将 MS 映射到 latent
      - 输出：通过 1D 反卷积 + 卷积 decoder 生成 full-grid 高光谱 (B, output_dim)

    所有卷积核和权重均为可学习参数，因此是纯 data-driven。
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2101,
        latent_dim: int = 256,
        spec_channels: int = 64,
        base_len: int = 33,   # 经过 3 次 stride=2 反卷积后：L ≈ base_len * 2^3 ≈ 264，最后再插值到 output_dim
    ):
        super().__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.spec_channels = spec_channels
        self.base_len = base_len

        # 1) MS -> latent 向量
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
            nn.ReLU(inplace=True),
        )

        # 2) latent -> 粗光谱特征 (B, C, L0)
        self.fc_expand = nn.Linear(latent_dim, spec_channels * base_len)

        # 3) decoder：3 层 1D ConvTranspose + 1D Conv 平滑
        self.deconv1 = nn.ConvTranspose1d(spec_channels, spec_channels, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(spec_channels, spec_channels, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose1d(spec_channels, spec_channels, kernel_size=4, stride=2, padding=1)

        self.conv_smooth1 = nn.Conv1d(spec_channels, spec_channels, kernel_size=5, padding=2)
        self.conv_smooth2 = nn.Conv1d(spec_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, num_ms_bands)
        b = x.shape[0]

        # MS -> latent
        z = self.mlp(x)                            # (B, latent_dim)

        # latent -> coarse spectral map
        h = self.fc_expand(z)                      # (B, C*L0)
        h = h.view(b, self.spec_channels, self.base_len)  # (B, C, L0)

        # 1D 反卷积逐步上采样
        h = F.relu(self.deconv1(h))               # (B, C, ~2*L0)
        h = F.relu(self.deconv2(h))               # (B, C, ~4*L0)
        h = F.relu(self.deconv3(h))               # (B, C, ~8*L0)

        # 平滑卷积
        h = F.relu(self.conv_smooth1(h))          # (B, C, L_dec)
        h = self.conv_smooth2(h)                  # (B, 1, L_dec)

        # 若长度与 output_dim 不一致，做线性插值
        if h.shape[-1] != self.output_dim:
            h = F.interpolate(h, size=self.output_dim, mode="linear", align_corners=False)

        out = h.squeeze(1)                       # (B, output_dim)
        return out


def build_spectral_conv_model(
    input_shape,
    output_dim: int,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    latent_dim: int = 256,
    spec_channels: int = 64,
):
    """
    构建 SpectralConvNet 模型和 Adam 优化器。
    loss_fn 只给一个占位（默认 combined_loss 风格），
    实际训练时你可以在脚本里重写 masked loss（推荐）。
    """
    input_dim = input_shape[0]
    model = SpectralConvNet(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        spec_channels=spec_channels,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def dummy_loss_fn(y_pred, y_true):
        # 只是一个占位；真正训练时我们会在 train_data_driven_spectral.py 里使用 masked loss。
        return spectral_rmse(y_true, y_pred) + alpha * spectral_angle_mapper(y_true, y_pred)

    return model, optimizer, dummy_loss_fn


