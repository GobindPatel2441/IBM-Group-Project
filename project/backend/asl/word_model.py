"""
Word-Level ASL Recognition Model Architectures
=============================================
Provides both:
1. SpatioTemporalAttentionModel (Linear + BiLSTM + Multi-Head Self-Attention) - Primary high-accuracy model.
2. PoseTransformer (Standard 6-layer Transformer Encoder).
Input: (batch_size, 32, 225)
  - 32 temporal frames
  - 225 features: right hand (63) + left hand (63) + pose (99)
Output: Vocabulary logits (trained core conversational classes).
"""

import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SpatioTemporalAttentionModel(nn.Module):
    """
    Spatio-Temporal Attention Network for Real-Time ASL Word Recognition.
    - Input: (B, seq_len, input_dim) -> e.g. (B, 32, 225)
    - Output: (B, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 225,
        seq_len: int = 32,
        d_model: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 54,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.n_heads = n_heads
        self.dropout = dropout

        # Spatial Feature Projection
        self.spatial_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional Encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 4, dropout=dropout)

        # Bidirectional Temporal LSTM
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # Multi-Head Self-Attention over Temporal Dimension
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout,
        )

        # Pooling & Classification Head
        lstm_dim = hidden_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_dim),
            nn.Linear(lstm_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.spatial_proj(x)
        feat = self.pos_enc(feat)

        lstm_out, _ = self.bilstm(feat)  # (B, T, 2*hidden_dim)

        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        fused = lstm_out + attn_out  # residual connection

        # Temporal global average + max pooling
        avg_pool = torch.mean(fused, dim=1)
        max_pool, _ = torch.max(fused, dim=1)
        pooled = (avg_pool + max_pool) * 0.5

        logits = self.head(pooled)
        return logits

    def predict_with_confidence(
        self, x: torch.Tensor, top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            topk_conf, topk_pred = torch.topk(probs, k=min(top_k, self.num_classes), dim=-1)
            top1_pred = topk_pred[:, 0]
            top1_conf = topk_conf[:, 0]
        return top1_pred, top1_conf, topk_pred, topk_conf

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "SpatioTemporalAttentionModel",
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class PoseTransformer(nn.Module):
    """
    Pose Transformer for temporal sequence sign recognition.
    """

    def __init__(
        self,
        input_dim: int = 225,
        seq_len: int = 32,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        num_classes: int = 54,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.dropout = dropout

        self.input_projection = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, max_len=seq_len + 1, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.input_projection(x)
        cls_expanded = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_expanded, x], dim=1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        cls_rep = x[:, 0]
        logits = self.classifier(cls_rep)
        return logits

    def predict_with_confidence(
        self, x: torch.Tensor, top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            topk_conf, topk_pred = torch.topk(probs, k=min(top_k, self.num_classes), dim=-1)
            top1_pred = topk_pred[:, 0]
            top1_conf = topk_conf[:, 0]
        return top1_pred, top1_conf, topk_pred, topk_conf

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "PoseTransformer",
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
