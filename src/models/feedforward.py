import torch
import torch.nn as nn

class FeedForward(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    self.linear_1 = nn.Linear(d_model, 4*d_model)
    self.linear_2 = nn.Linear(4*d_model, d_model)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.linear_2(self.relu(self.linear_1(x)))