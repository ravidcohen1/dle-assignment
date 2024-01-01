from torch import nn

from src.model.attention import Attention
from src.model.embedding import SequenceEmbedding
from src.model.mlp import MLPBlock


def get_model_from_config(config: dict) -> nn.Module:
    return Transformer(
        input_len=config["input_len"],
        embedding_dim=config["embedding_dim"],
        attention_hidden_dim=config["attention_hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_transformer_layers"],
        mlp_hidden_dim=config["mlp_hidden_dim"],
        vocab_size=config["vocab_size"],
        eps=float(config["norm_layer_eps"]),
        scale_attention=config["scale_attention"],
        gelu_approximation=config["gelu_approximation"],
        transformer_forward_alternative=config["transformer_forward_alternative"],
    )


class Transformer(nn.Module):
    def __init__(
        self,
        input_len,
        embedding_dim,
        attention_hidden_dim,
        num_heads,
        num_layers,
        mlp_hidden_dim,
        vocab_size,
        eps,
        scale_attention=True,
        gelu_approximation="none",
        transformer_forward_alternative="none",
    ):
        super().__init__()
        self.embedding = SequenceEmbedding(vocab_size, input_len, embedding_dim)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    input_len,
                    embedding_dim,
                    attention_hidden_dim,
                    num_heads,
                    mlp_hidden_dim,
                    eps,
                    scale_attention,
                    gelu_approximation,
                    transformer_forward_alternative,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim, eps=eps)
        self.lm_head = self.embedding.get_word_embedding()

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.norm(x)
        y = x @ self.lm_head.T
        return y


class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_len,
        input_dim,
        attention_hidden_dim,
        num_heads,
        mlp_hidden_dim,
        eps,
        scale_attention=True,
        gelu_approximation="none",
        transformer_forward_alternative="none",
    ):
        """
        A single transformer layer. Both input and output shapes are (input_len, input_dim)
        :param input_len: the length of the input sequence, also the length of the output sequence
        :param input_dim: the dimension of the input sequence
        :param attention_hidden_dim: the dimension of the hidden layers of the attention heads
        :param num_heads: the number of attention heads
        :param mlp_hidden_dim: the dimension of the hidden layers of the MLP block
        :param eps: the epsilon value for the layer normalization
        :param scale_attention: whether to scale the attention weights by sqrt(input_dim)
        :param gelu_approximation: approximation of GELU
        :param transformer_forward_alternative: whether to use the alternative implementation of the transformer, one
        of ['none', '1', '2', '3']
        """
        super().__init__()
        self.attention = Attention(
            input_len, input_dim, attention_hidden_dim, num_heads, scale_attention
        )
        self.mlp = MLPBlock(input_dim, mlp_hidden_dim, input_dim, gelu_approximation)
        self.norm1 = nn.LayerNorm(input_dim, eps=eps)
        self.norm2 = nn.LayerNorm(input_dim, eps=eps)
        assert transformer_forward_alternative in ["none", "1", "2", "3"]
        if transformer_forward_alternative == "1":
            self.forward = self.forward_alternative_1
        elif transformer_forward_alternative == "2":
            self.forward = self.forward_alternative_2
        elif transformer_forward_alternative == "3":
            self.forward = self.forward_alternative_3

    def forward(self, x):
        t = x + self.attention(self.norm1(x))
        y = t + self.mlp(self.norm2(t))
        return y

    def forward_alternative_1(self, x):
        t = x + self.attention(self.norm1(x))
        y = x + self.mlp(self.norm2(t))
        return y

    def forward_alternative_2(self, x):
        t = x + self.attention(self.norm1(x))
        y = self.mlp(self.norm2(t))
        return y

    def forward_alternative_3(self, x):
        y = x + self.attention(self.norm1(x)) + self.mlp(self.norm2(x))
        return y
