import torch
import torch.nn as nn
from src.models import MultiHeadAttention, FeedForward, ResidualConnection

class EncoderBlock(nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ff = FeedForward(d_model)
    self.dropout = nn.Dropout(dropout)
    self.residual_1 = ResidualConnection(dropout)
    self.residual_2 = ResidualConnection(dropout)

  def forward(self, x, mask):
    x = self.residual_1(x, lambda x: self.mha(x, x, x, mask))
    x = self.residual_2(x, lambda x: self.ff(x))
    return x