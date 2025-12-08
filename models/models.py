"""Model architectures for multispectral-to-hyperspectral reconstruction."""

from typing import Tuple
from tensorflow.keras import layers, models, optimizers

import sys
import os

# === 路径与导入 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from .losses import combined_loss
from config import DEFAULT_ALPHA, DEFAULT_BETA, LEARNING_RATE


def squeeze_excite_block(x, ratio: int = 4):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(filters // ratio, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, filters))(se)
    return layers.Multiply()([x, se])


def transformer_block(x, num_heads: int = 4, d_model: int = 256, ff_dim: int = 512, dropout: float = 0.1):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = layers.Dense(ff_dim, activation="relu")(x)
    ffn = layers.Dense(d_model)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_simple_cnn(input_shape: Tuple[int], output_dim: int, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA):
    """Baseline: simple 1D CNN without SE/Transformer."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(inputs)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, alpha, beta),
        metrics=["mae"],
    )
    return model


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
    """Full M2H-SWIR model with parallel conv, SE, residual and transformer encoder."""
    inputs = layers.Input(shape=input_shape)  # (bands,)
    x = layers.Reshape((input_shape[0], 1))(inputs)

    # Multiscale conv branches
    if use_multiscale_conv:
        b1 = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        b2 = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
        b3 = layers.Conv1D(64, 7, padding="same", activation="relu")(x)
        x = layers.Concatenate()([b1, b2, b3])
    else:
        x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)

    # SE attention
    if use_se:
        x = squeeze_excite_block(x)

    # Residual blocks
    def residual_block(z, filters):
        shortcut = z
        y = layers.Conv1D(filters, 3, padding="same", activation="relu")(z)
        y = layers.Conv1D(filters, 3, padding="same", activation=None)(y)
        y = layers.Add()([shortcut, y])
        y = layers.Activation("relu")(y)
        return y

    if use_residual:
        filters = x.shape[-1]
        x = residual_block(x, filters)
        x = residual_block(x, filters)
    else:
        x = layers.Conv1D(x.shape[-1], 3, padding="same", activation="relu")(x)

    # Transformer encoder
    if use_transformer:
        x = layers.Conv1D(d_model, 1, padding="same", activation="linear")(x)
        for _ in range(num_transformer_layers):
            x = transformer_block(x, num_heads=num_heads, d_model=d_model, ff_dim=ff_dim)
    else:
        x = layers.Conv1D(256, 1, padding="same", activation="relu")(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, alpha, beta),
        metrics=["mae"],
    )
    return model
