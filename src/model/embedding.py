import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_items, embedding_dim))

    def forward(self, x):
        return self.embedding[x]


class SequenceEmbedding(nn.Module):
    """
    A simple embedding layer that also adds positional embeddings.
    """

    def __init__(self, vocab_size, max_len, embedding_dim):
        super().__init__()
        self.word_embedding = Embedding(vocab_size, embedding_dim)
        self.position_embedding = Embedding(max_len, embedding_dim)

    def forward(self, x):
        assert x.shape[0] == 1, "SequenceEmbedding only supports batch size 1"

        word_embeds = self.word_embedding(x[0])
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        sequence_embedding = word_embeds + pos_embeds
        return sequence_embedding.squeeze(0)

    def get_word_embedding(self):
        return self.word_embedding.embedding
