import torch


class LayerNorm(torch.nn.Module):
    """
    A thin wrapper around :class:`torch.nn.LayerNorm` to support fp16.
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        raise NotImplementedError
