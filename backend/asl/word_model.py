"""
Word-Level Pose Transformer Architecture
========================================
Skeletal Transformer for temporal sign language sequence recognition.
Input: (batch_size, 64, 225)
  - 64 temporal frames
  - 225 features: right hand (63) + left hand (63) + pose (99)
Output: Vocabulary logits (200 trained classes).
"""

import math
from typing import Tuple, Dict, Any
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
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PoseTransformer(nn.Module):
    """
    Pose Transformer for temporal sequence sign recognition.
    Matches the exact architecture from Phase 1.
    """

    def __init__(
        self,
        input_dim: int = 225,
        seq_len: int = 64,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        num_classes: int = 200,
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

        # Linear projection of skeletal features to transformer dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding for sequence length + CLS token
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, max_len=seq_len + 1, dropout=dropout
        )

        # Transformer encoder layers (Pre-LN formulation)
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

        # Classification Head on CLS token
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        # Project inputs to d_model: (B, seq_len, d_model)
        x = self.input_projection(x)

        # Prepend CLS token: (B, seq_len + 1, d_model)
        cls_expanded = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_expanded, x], dim=1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder pass
        x = self.transformer(x)

        # Classify from CLS token representation (index 0)
        cls_rep = x[:, 0]
        logits = self.classifier(cls_rep)
        return logits

    def predict_with_confidence(
        self, x: torch.Tensor, top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference pass returning top-1 and top-k predictions with confidence probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            topk_conf, topk_pred = torch.topk(probs, k=min(top_k, self.num_classes), dim=-1)
            top1_pred = topk_pred[:, 0]
            top1_conf = topk_conf[:, 0]
        return top1_pred, top1_conf, topk_pred, topk_conf

    def get_config(self) -> Dict[str, Any]:
        """Return model architecture configuration dictionary."""
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

