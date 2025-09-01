import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads

    assert d_model%num_heads == 0, "d_model must be divisible by num_heads"
    self.q_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)
    self.out = nn.Linear(d_model, d_model)

  def attention_scores(self, query, key, value, mask):
    attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim) # (Batch, num_heads, seq_len, seq_len)
    if mask is not None:
      attention_score = attention_score.masked_fill(mask == 0, -1e9)

    attention_score = torch.softmax(attention_score, dim = -1) # (Batch, num_heads, seq_len, seq_len)
    attention_score = attention_score @ value # (Batch, num_heads, seq_len, head_dim)
    return attention_score

  def forward(self, query, key, value, mask):
    batch, seq_len, d_model = query.shape

    query = self.q_linear(query) # (Batch, seq_len, d_model)
    key = self.k_linear(key) # (Batch, seq_len, d_model)
    value = self.v_linear(value) # (Batch, seq_len, d_model)

    # (Batch, seq_len, d_model) --> (Batch, seq_len, num_heads, head_dim) --> (Batch, num_heads, seq_len, head_dim)
    query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    attention_score = self.attention_scores(query, key, value, mask) # (Batch, num_heads, seq_len, head_dim)
    attention_score = attention_score.transpose(1, 2).contiguous().view(batch, -1, self.d_model)

    return self.out(attention_score) # (Batch, seq_len, d_model)