from torch import nn


class LinearProbeModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear_layer = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # assume [batch, feature_dim]
        return self.linear_layer(x)
