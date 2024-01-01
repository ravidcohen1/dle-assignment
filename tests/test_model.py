import torch

from src.model.attention import Attention, AttentionHead
from src.model.embedding import SequenceEmbedding
from src.model.mlp import MLPBlock
from src.model.transformer import Transformer, TransformerLayer


def overfit_module(
    module, input_shape, output_shape, n_steps=1000, lr=1e-3, int_input=False
):
    optim = torch.optim.Adam(module.parameters(), lr=lr)
    if int_input:
        x = torch.randint(0, 100, input_shape)
    else:
        x = torch.randn(*input_shape)
    y_pred = module(x)
    assert (
        y_pred.shape == output_shape
    ), f"Expected output shape {output_shape}, got {y_pred.shape}"
    y_true = torch.randn(*output_shape)
    for i in range(n_steps):
        optim.zero_grad()
        y_pred = module(x)
        loss = torch.nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        optim.step()
    assert loss < 1e-5, f"Loss did not converge, got {loss}"


def test_mlp_block():
    input_dim = 10
    hidden_dim = 20
    output_dim = 3
    batch_size = 10

    mlp = MLPBlock(input_dim, hidden_dim, output_dim)
    overfit_module(mlp, (batch_size, input_dim), (batch_size, output_dim))


def test_attention_head():
    input_len = 5
    input_dim = 10
    hidden_dim = 20

    head = AttentionHead(
        input_len=input_len, input_dim=input_dim, hidden_dim=hidden_dim
    )
    overfit_module(head, (input_len, input_dim), (input_len, hidden_dim), n_steps=10000)


def test_attention_block():
    input_len = 5
    input_dim = 10
    hidden_dim = 20
    num_heads = 4

    block = Attention(input_len, input_dim, hidden_dim, num_heads)
    overfit_module(block, (input_len, input_dim), (input_len, input_dim), n_steps=10000)


def test_embedding():
    vocab_size = 100
    max_len = 20
    embedding_dim = 30

    embedding = SequenceEmbedding(vocab_size, max_len, embedding_dim)
    overfit_module(
        embedding, (1, max_len), (max_len, embedding_dim), int_input=True, n_steps=10000
    )


def test_transformer_layer():
    input_len = 5
    input_dim = 10
    attention_hidden_dim = 20
    num_heads = 4
    mlp_hidden_dim = 30

    layer = TransformerLayer(
        input_len, input_dim, attention_hidden_dim, num_heads, mlp_hidden_dim, eps=1e-6
    )
    overfit_module(layer, (input_len, input_dim), (input_len, input_dim), n_steps=1000)


def test_transformer():
    input_len = 2
    embedding_dim = 4
    attention_hidden_dim = 4
    num_heads = 2
    mlp_hidden_dim = 4
    vocab_size = 100
    num_layers = 2

    transformer = Transformer(
        input_len,
        embedding_dim,
        attention_hidden_dim,
        num_heads,
        num_layers,
        mlp_hidden_dim,
        vocab_size,
        eps=1e-6,
    )
    overfit_module(
        transformer,
        (1, input_len),
        (input_len, vocab_size),
        int_input=True,
        n_steps=10000,
    )


def test_transformer_input_len():
    max_input_len = 100
    embedding_dim = 4
    attention_hidden_dim = 4
    num_heads = 2
    mlp_hidden_dim = 4
    vocab_size = 100
    num_layers = 2

    transformer = Transformer(
        max_input_len,
        embedding_dim,
        attention_hidden_dim,
        num_heads,
        num_layers,
        mlp_hidden_dim,
        vocab_size,
        eps=1e-6,
    )
    for input_len in [1, 10, 100]:
        x = torch.randint(0, vocab_size, (1, input_len))
        y = transformer(x)
        assert y.shape == (
            input_len,
            vocab_size,
        ), f"Expected output shape {(input_len, vocab_size)}, got {y.shape}"
