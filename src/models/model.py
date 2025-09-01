import torch
import torch.nn as nn
from src.models import EncoderBlock, InputEmbeddings, Positional_Encoding, ProjectionLayer


class FeelNet(nn.Module):
  def __init__(self, d_model, num_heads, dropout, n_layers, vocab_size, seq_len, num_classes):
    super().__init__()
    self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, dropout) for _ in range(n_layers)])
    self.embeddings = InputEmbeddings(vocab_size, d_model)
    self.positional_encoding = Positional_Encoding(d_model, seq_len, dropout)
    self.projection_layer = ProjectionLayer(d_model, num_classes)
    self.dropout = nn.Dropout(dropout)
    self.d_model = d_model
    self.num_heads = num_heads
    self.n_layers = n_layers

  def forward(self, x, mask):
    x = self.embeddings(x)
    x = self.positional_encoding(x)
    for layer in self.layers:
      x = layer(x, mask)
    x = x[:, 0, :] # Get the value from `[CLS]` token only, cause it contains all the information regarding other tokens (Batch, 1, d_model)
    x = self.projection_layer(x)
    return x