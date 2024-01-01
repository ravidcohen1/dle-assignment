import torch
from torch import nn


class AttentionHead(nn.Module):
    def __init__(
        self, input_len: int, input_dim: int, hidden_dim: int, scale_attention: bool
    ):
        """
        A single attention head. Both input and output shapes are (input_len, input_dim)
        :param input_len: the length of the input sequence, also the length of the output sequence
        :param input_dim: the dimension of the input sequence
        :param hidden_dim: the dimension of the hidden state, also the dimension of the output sequence
        :param scale_attention: whether to scale the attention weights by sqrt(input_dim)
        """
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim, bias=True)
        self.key = nn.Linear(input_dim, hidden_dim, bias=True)
        self.value = nn.Linear(input_dim, hidden_dim, bias=True)
        self.hidden_dim = hidden_dim
        self.scale_attention = scale_attention

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        weights_logits = query @ key.T
        if self.scale_attention:
            weights_logits /= self.hidden_dim**0.5
        masked_weights_logits = weights_logits.clone()

        # Apply the mask
        causal_mask = torch.triu(
            torch.ones_like(masked_weights_logits), diagonal=1
        ).bool()
        masked_weights_logits[causal_mask] = float("-inf")
        weights = torch.softmax(masked_weights_logits, dim=-1)
        y = weights @ value
        return y


class Attention(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        scale_attention: bool,
    ):
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [
                AttentionHead(input_len, input_dim, hidden_dim, scale_attention)
                for _ in range(num_heads)
            ]
        )
        self.output_projection = nn.Linear(hidden_dim * num_heads, input_dim, bias=True)

    def forward(self, x):
        attention_outputs = [head(x) for head in self.attention_heads]
        attention_outputs = torch.cat(attention_outputs, dim=-1)
        y = self.output_projection(attention_outputs)
        return y
