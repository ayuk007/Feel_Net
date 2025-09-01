import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
  def __init__(self, d_model, n_classes: int):
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_model)
    self.relu = nn.ReLU()
    self.linear_2 = nn.Linear(d_model, n_classes)

  def forward(self, x):
    return self.linear_2(self.relu(self.linear_1(x)))