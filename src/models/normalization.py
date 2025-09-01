import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
  # x_new = (x - mean)/sqrt(std**2 + eps)
  def __init__(self, eps: float = 10**-6):
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1))
    self.bias = nn.Parameter(torch.zeros(1))

  def forward(self, x):
    mean = x.mean(dim =-1, keepdim = True) # (Batch, seq_len, 1)
    std = x.std(dim = -1, keepdim = True) # (Batch, seq_len, 1)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias