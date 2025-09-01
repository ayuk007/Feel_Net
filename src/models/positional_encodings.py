import math
import torch
import torch.nn as nn


class Positional_Encoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout: float):
    super().__init__()

    self.dropout = nn.Dropout(dropout)
    self.d_model = d_model
    self.seq_len = seq_len

    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype = float).unsqueeze(1)# type: ignore # (seq_len, 1) --> We take tokens as row records to be simple so unsqueeze(1) means we'll have rows of (seq_len) and col_1
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0/d_model))) # (d_model//2) --> [1, 0.1, 0.2, 0.02]

    pe[:, 0::2] = torch.sin(position * div_term) # (seq_len, d_model//2)
    pe[:, 1::2] = torch.cos(position * div_term) # (seq_len, d_model//2)

    self.register_buffer("pe", pe.unsqueeze(0)) # (1, seq_len, d_model)

  def forward(self, x):
    x = x + (self.pe[:, :x.shape[1]:]).requires_grad_(False)
    return self.dropout(x) # (Batch, seq_len, d_model)