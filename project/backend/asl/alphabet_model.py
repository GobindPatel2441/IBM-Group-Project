"""
Fingerspelling MLP Architecture
===============================
Standard Multi-Layer Perceptron for static ASL Alphabet classification.
Input: 126 hand landmark features (21 landmarks * 3 coordinates * 2 hands).
Output: 29 classes (A-Z, del, nothing, space).
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn


class FingerspellingMLP(nn.Module):
    """
    4-Layer Multi-Layer Perceptron with Batch Normalization and Dropout.
    Matches the exact architecture from the Phase 1 standalone ASL system.
    """

    def __init__(self, input_dim: int = 126, num_classes: int = 29):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.network = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output Head
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        return self.network(x)

    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference forward pass returning predicted class indices and probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
        return pred, conf

    def get_config(self) -> Dict[str, Any]:
        """Return model architecture configuration dictionary."""
        return {
            "model_type": "FingerspellingMLP",
            "input_dim": self.input_dim,
            "hidden_dims": [512, 256, 128],
            "dropouts": [0.3, 0.3, 0.2],
            "num_classes": self.num_classes,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

