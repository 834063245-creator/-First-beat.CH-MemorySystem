"""ChuchuCNN — 字符级 3 路 CNN 分类器。

架构：
  Embedding(vocab_size, 64) → 3×Conv1D(3/4/5, 128) → MaxPool → Concat
  → Dropout(0.5) → FC(384, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChuchuCNN(nn.Module):
    """字符级 CNN 分类器，专为中文短文本设计。"""

    def __init__(self, vocab_size: int, num_classes: int,
                 embed_dim: int = 64, num_filters: int = 128,
                 kernel_sizes: tuple[int, ...] = (3, 4, 5),
                 dropout: float = 0.5, max_len: int = 64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) — token IDs
        Returns:
            logits: (batch, num_classes)
        """
        # (batch, seq_len) → (batch, embed_dim, seq_len)
        x = self.embedding(x).transpose(1, 2)

        # 每路卷积: (batch, num_filters, seq_len - k + 1)
        conv_outs = []
        for conv in self.convs:
            h = F.relu(conv(x))
            # MaxPool over time: (batch, num_filters, 1) →  (batch, num_filters)
            h = F.max_pool1d(h, h.size(2)).squeeze(2)
            conv_outs.append(h)

        # 拼接: (batch, num_filters * len(kernel_sizes))
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        return self.fc(x)
