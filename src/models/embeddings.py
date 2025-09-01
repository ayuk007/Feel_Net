import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
  def __init__(self, vocab_size: int, d_model: int):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.embeddings = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.embeddings(x) * math.sqrt(self.d_model) # (Batch, Seq_Len, d_model)