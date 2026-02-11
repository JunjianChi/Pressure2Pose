"""
Temporal model architectures for Pressure2Pose.

All models share PressureEncoder (spatial) + SMPLRegressor (parameter heads)
from pressure_to_smpl.py, and differ only in how they aggregate temporal context.

Model A: CNN Baseline        — single frame, no temporal
Model B: CNN + BiGRU         — bidirectional GRU
Model C: CNN + BiLSTM        — bidirectional LSTM
Model D: CNN + TCN           — dilated causal convolutions
Model E: CNN + Transformer   — transformer encoder
"""

import math
import torch
import torch.nn as nn

from .pressure_to_smpl import PressureEncoder, SMPLRegressor


# ---------------------------------------------------------------------------
# Model A: CNN Baseline (single frame)
# ---------------------------------------------------------------------------

class CNNBaseline(nn.Module):
    """Single-frame baseline: PressureEncoder -> SMPLRegressor.
    Accepts either (B, 2, H, W) or (B, T, 2, H, W); for sequences uses center frame.
    """

    def __init__(self, feature_dim=512, num_betas=10, num_joints=23):
        super().__init__()
        self.encoder = PressureEncoder(in_channels=2, feature_dim=feature_dim)
        self.regressor = SMPLRegressor(feature_dim, num_betas, num_joints)

    def forward(self, x):
        """
        Args:
            x: (B, 2, H, W) or (B, T, 2, H, W)
        Returns:
            betas (B,10), body_pose (B,69), global_orient (B,3), transl (B,3)
        """
        if x.dim() == 5:
            # Sequence input -> take center frame
            T = x.shape[1]
            x = x[:, T // 2]  # (B, 2, H, W)
        features = self.encoder(x)  # (B, feature_dim)
        return self.regressor(features)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TemporalBase(nn.Module):
    """Base for all temporal models: encode each frame, aggregate, regress."""

    def __init__(self, feature_dim=512, num_betas=10, num_joints=23):
        super().__init__()
        self.encoder = PressureEncoder(in_channels=2, feature_dim=feature_dim)
        self.regressor = SMPLRegressor(feature_dim, num_betas, num_joints)
        self.feature_dim = feature_dim

    def _encode_sequence(self, x):
        """Encode each frame independently.
        Args:
            x: (B, T, 2, H, W)
        Returns:
            (B, T, feature_dim)
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        feats = self.encoder(x_flat)         # (B*T, D)
        return feats.reshape(B, T, -1)       # (B, T, D)


# ---------------------------------------------------------------------------
# Model B: CNN + BiGRU
# ---------------------------------------------------------------------------

class CNNBiGRU(_TemporalBase):
    """CNN spatial encoder + bidirectional GRU temporal aggregator."""

    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=2,
                 num_betas=10, num_joints=23, dropout=0.1):
        super().__init__(feature_dim, num_betas, num_joints)
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # BiGRU outputs hidden_dim*2; project back to feature_dim
        self.proj = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, x):
        """x: (B, T, 2, H, W) -> SMPL params of center frame."""
        feats = self._encode_sequence(x)  # (B, T, D)
        gru_out, _ = self.gru(feats)      # (B, T, hidden*2)
        center = gru_out[:, gru_out.shape[1] // 2]  # center frame
        proj = self.proj(center)           # (B, D)
        return self.regressor(proj)


# ---------------------------------------------------------------------------
# Model C: CNN + BiLSTM
# ---------------------------------------------------------------------------

class CNNBiLSTM(_TemporalBase):
    """CNN spatial encoder + bidirectional LSTM temporal aggregator."""

    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=2,
                 num_betas=10, num_joints=23, dropout=0.1):
        super().__init__(feature_dim, num_betas, num_joints)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.proj = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, x):
        feats = self._encode_sequence(x)
        lstm_out, _ = self.lstm(feats)
        center = lstm_out[:, lstm_out.shape[1] // 2]
        proj = self.proj(center)
        return self.regressor(proj)


# ---------------------------------------------------------------------------
# Model D: CNN + TCN (Temporal Convolutional Network)
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    """Dilated causal Conv1d block with residual connection."""

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # same-length padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, C, T)"""
        residual = x
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        return out + residual


class CNNTCN(_TemporalBase):
    """CNN spatial encoder + TCN (dilated causal convolutions)."""

    def __init__(self, feature_dim=512, num_blocks=4, kernel_size=3,
                 num_betas=10, num_joints=23, dropout=0.1):
        super().__init__(feature_dim, num_betas, num_joints)
        dilations = [2 ** i for i in range(num_blocks)]  # 1, 2, 4, 8
        self.tcn = nn.Sequential(*[
            _TCNBlock(feature_dim, kernel_size, d, dropout) for d in dilations
        ])

    def forward(self, x):
        feats = self._encode_sequence(x)       # (B, T, D)
        feats = feats.transpose(1, 2)          # (B, D, T)
        tcn_out = self.tcn(feats)              # (B, D, T)
        center = tcn_out[:, :, tcn_out.shape[2] // 2]  # (B, D)
        return self.regressor(center)


# ---------------------------------------------------------------------------
# Model E: CNN + Transformer Encoder
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x):
        """x: (B, T, D)"""
        return x + self.pe[:, :x.shape[1]]


class CNNTransformer(_TemporalBase):
    """CNN spatial encoder + Transformer encoder temporal aggregator."""

    def __init__(self, feature_dim=512, nhead=8, num_layers=4,
                 dim_feedforward=1024, num_betas=10, num_joints=23, dropout=0.1):
        super().__init__(feature_dim, num_betas, num_joints)
        self.pos_enc = _PositionalEncoding(feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        feats = self._encode_sequence(x)          # (B, T, D)
        feats = self.pos_enc(feats)               # (B, T, D)
        tf_out = self.transformer(feats)          # (B, T, D)
        center = tf_out[:, tf_out.shape[1] // 2]  # (B, D)
        return self.regressor(center)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    'cnn_baseline': CNNBaseline,
    'cnn_bigru': CNNBiGRU,
    'cnn_bilstm': CNNBiLSTM,
    'cnn_tcn': CNNTCN,
    'cnn_transformer': CNNTransformer,
}


def build_model(config):
    """Instantiate a model from a config dict.

    Required keys:
        config['model']['type']: one of MODEL_REGISTRY keys
    Optional keys forwarded to constructor:
        config['model']['feature_dim'], config['model']['num_betas'], etc.

    Returns:
        nn.Module
    """
    model_type = config['model']['type'].lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")

    cls = MODEL_REGISTRY[model_type]

    # Gather constructor kwargs from config, filtering to only those
    # accepted by the specific model class
    import inspect
    valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {'self'}

    kwargs = {}
    m_cfg = config.get('model', {})
    for key in ['feature_dim', 'num_betas', 'num_joints',
                'hidden_dim', 'num_layers', 'dropout',
                'nhead', 'dim_feedforward', 'num_blocks', 'kernel_size']:
        if key in m_cfg and key in valid_params:
            kwargs[key] = m_cfg[key]

    return cls(**kwargs)
