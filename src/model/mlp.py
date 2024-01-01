from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, gelu_approximation):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        assert gelu_approximation in ["none", "tanh"]
        self.norm = nn.GELU(approximate=gelu_approximation)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x
