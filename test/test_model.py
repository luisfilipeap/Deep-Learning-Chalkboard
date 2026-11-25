"""Software tests for the assembled miniBERT (``dlcb/transformer.py``).

The same five kinds of evidence as ``test/test_attention.py``, one level up:

1. **An independent reference** — the whole model written again in plain NumPy,
   token by token, from the BERT paper's description; it must agree to 1e-10.
2. **Properties** — padding cannot leak into real tokens; positions break the
   permutation symmetry of attention; every output token is standardised by the
   final LayerNorm; the MLM decoder really is the embedding table.
3. **Finite differences** — every parameter of a tiny MLM model, end to end.
4. **The masking recipe** — 15 % chosen, then 80 / 10 / 10, specials untouched.
5. **Behaviour** — the model must learn to fill a masked token that can *only*
   be recovered from its neighbours.

Run them with::

    pytest test/test_model.py
"""

import numpy as np
import pytest

from dlcb import engine as cpu
from dlcb.attention import MultiHeadAttention
from dlcb.engine import Tensor
from dlcb.loss import cross_entropy, masked_lm_loss
from dlcb.optim import Adam
from dlcb.transformer import BERTForMaskedLM, BERTModel, EncoderLayer, mask_tokens
from test.test_attention import reference_multihead
from test.test_engine import numeric_gradient


def tiny_mlm(vocab=12, d_model=8, num_layers=2, num_heads=2, d_ff=16, max_len=6, dropout=0.0):
    bert = BERTModel(vocab, d_model, num_layers, num_heads, d_ff, max_len, dropout)
    return BERTForMaskedLM(bert, vocab)


# ====================================================================== #
# 1. Independent reference (plain NumPy, no engine)
# ====================================================================== #
def _affine(layer, t):
    W = layer.weight.data
    return W[0] + t @ W[1:]


def _layernorm(ln, t):
    mu = t.mean(axis=-1, keepdims=True)
    var = ((t - mu) ** 2).mean(axis=-1, keepdims=True)
    return (t - mu) / np.sqrt(var + ln.eps) * ln.gamma.data + ln.beta.data


def _gelu(z):
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))


def reference_mlm(mlm: BERTForMaskedLM, ids: np.ndarray, attention_mask=None) -> np.ndarray:
    """BERT for masked LM, token by token, reusing only the model's weights."""
    bert = mlm.bert
    emb = bert.embeddings
    batch, seq = ids.shape
    x = np.zeros((batch, seq, bert.d_model))
    for b in range(batch):
        for i in range(seq):
            x[b, i] = (emb.token.weight.data[ids[b, i]] + emb.position.weight.data[i]
                       + emb.segment.weight.data[0])
    x = _layernorm(emb.norm, x)

    mask = None
    if attention_mask is not None:
        mask = np.where(attention_mask == 0, -np.inf, 0.0)[:, None, None, :]
    for layer in bert.encoder.layers:
        x = _layernorm(layer.norm1, x + reference_multihead(layer.attention, x, mask))
        ffn = np.zeros_like(x)
        for b in range(batch):
            for i in range(seq):
                ffn[b, i] = _affine(layer.ffn.fc2, _gelu(_affine(layer.ffn.fc1, x[b, i])))
        x = _layernorm(layer.norm2, x + ffn)

    logits = np.zeros((batch, seq, bert.vocab_size))
    E = emb.token.weight.data
    for b in range(batch):
        for i in range(seq):
            t = _layernorm(mlm.norm, _gelu(_affine(mlm.transform, x[b, i])))
            logits[b, i] = E @ t + mlm.bias.data
    return logits


def test_mlm_matches_numpy_reference_with_padding():
    """Every parameter is re-drawn at random before comparing.

    At initialisation some parameters are special values — biases and LayerNorm
    ``beta`` are 0, ``gamma`` is 1 — and a bug that drops or misplaces them is
    invisible there. Random values make every parameter count.
    """
    cpu.set_seed(0)
    mlm = tiny_mlm()
    for p in mlm.parameters():
        p.data[...] = np.random.randn(*p.shape)
    ids = np.array([[2, 7, 9, 5, 3, 0], [2, 11, 6, 3, 0, 0]])
    am = (ids != 0).astype(int)
    out = mlm(ids, attention_mask=am).data
    assert out.shape == (2, 6, 12)
    assert np.allclose(out, reference_mlm(mlm, ids, am), atol=1e-10)


# ====================================================================== #
# Shapes (the original stubs)
# ====================================================================== #
def test_encoder_layer_shape():
    """An encoder layer preserves ``(batch, seq, d_model)``."""
    cpu.set_seed(1)
    layer = EncoderLayer(d_model=8, num_heads=2, d_ff=16, dropout=0.0)
    assert layer(Tensor(np.random.randn(3, 5, 8))).shape == (3, 5, 8)


def test_multihead_attention_shape():
    """Self-attention preserves ``(batch, seq, d_model)`` (full checks: test_attention.py)."""
    cpu.set_seed(2)
    mha = MultiHeadAttention(d_model=8, num_heads=2, dropout=0.0)
    assert mha(Tensor(np.random.randn(3, 5, 8))).shape == (3, 5, 8)


def test_bert_forward_shape():
    """BERTModel returns ``(batch, seq, d_model)`` hidden states."""
    cpu.set_seed(3)
    bert = BERTModel(vocab_size=20, d_model=16, num_layers=2, num_heads=4, d_ff=32, max_len=10)
    assert bert(np.random.randint(5, 20, size=(4, 7))).shape == (4, 7, 16)


# ====================================================================== #
# 2. Properties
# ====================================================================== #
def test_padding_cannot_change_real_tokens():
    """Whatever sits in the padded positions, the real tokens' outputs are identical."""
    cpu.set_seed(4)
    mlm = tiny_mlm().eval()
    ids = np.array([[2, 7, 9, 3, 0, 0]])
    am = (ids != 0).astype(int)
    other = ids.copy()
    other[0, 4:] = [10, 11]                       # garbage where the padding was
    a = mlm(ids, attention_mask=am).data[0, :4]
    b = mlm(other, attention_mask=am).data[0, :4]
    assert np.allclose(a, b, atol=1e-12)


def test_positions_break_the_permutation_symmetry():
    """Attention alone is permutation-equivariant; with position embeddings BERT is not."""
    cpu.set_seed(5)
    bert = tiny_mlm().bert
    ids = np.array([[5, 6, 7, 8, 9]])
    perm = np.array([4, 2, 0, 3, 1])
    out = bert(ids).data
    out_perm = bert(ids[:, perm]).data
    assert not np.allclose(out_perm, out[:, perm], atol=1e-6)


def test_every_output_token_is_standardised():
    """The last op is a LayerNorm with gamma=1, beta=0 at init: mean 0, variance 1 per token."""
    cpu.set_seed(6)
    h = tiny_mlm().bert(np.random.randint(5, 12, size=(3, 6))).data
    assert np.allclose(h.mean(axis=-1), 0.0, atol=1e-10)
    assert np.allclose(h.var(axis=-1), 1.0, atol=1e-3)


def test_dropout_only_in_training_mode():
    cpu.set_seed(7)
    mlm = tiny_mlm(dropout=0.3)
    ids = np.random.randint(5, 12, size=(2, 6))
    mlm.eval()
    assert np.array_equal(mlm(ids).data, mlm(ids).data)
    mlm.train()
    assert not np.allclose(mlm(ids).data, mlm(ids).data)


def test_decoder_is_tied_to_the_token_embeddings():
    """One table, listed once, receiving gradient even for ids absent from the input."""
    cpu.set_seed(8)
    mlm = tiny_mlm()
    params = mlm.parameters()
    assert len(params) == len({id(p) for p in params})
    ids = np.array([[5, 6, 7, 5]])               # ids 8..11 never appear in the input
    mlm.zero_grad()
    masked_lm_loss(mlm(ids), np.array([[6, -100, -100, -100]])).backward()
    assert np.all(np.abs(mlm.bert.embeddings.token.weight.grad[8:]).sum(axis=1) > 0)


def test_sequence_longer_than_max_len_is_rejected():
    with pytest.raises(ValueError):
        tiny_mlm(max_len=4)(np.ones((1, 5), dtype=int))


def test_mlm_loss_counts_only_masked_positions():
    cpu.set_seed(9)
    logits = Tensor(np.random.randn(2, 4, 6))
    labels = np.array([[-100, 3, -100, 1], [5, -100, -100, -100]])
    loss = float(masked_lm_loss(logits, labels).data)
    z = logits.data
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    expected = -np.mean([logp[0, 1, 3], logp[0, 3, 1], logp[1, 0, 5]])
    assert np.isclose(loss, expected)
    assert np.isclose(loss, float(cross_entropy(logits, labels).data))


# ====================================================================== #
# 3. Finite differences, every parameter
# ====================================================================== #
def test_mlm_gradcheck_every_parameter():
    cpu.set_seed(10)
    mlm = tiny_mlm(vocab=7, d_model=4, num_layers=1, num_heads=2, d_ff=6, max_len=4)
    ids = np.array([[2, 5, 6, 0], [2, 4, 3, 0]])
    am = (ids != 0).astype(int)
    labels = np.array([[-100, 5, 6, -100], [-100, 1, -100, -100]])

    def loss():
        return masked_lm_loss(mlm(ids, attention_mask=am), labels)

    mlm.zero_grad()
    loss().backward()
    for name, p in mlm.named_parameters():
        numeric = numeric_gradient(lambda: float(loss().data), p.data)
        assert np.allclose(p.grad, numeric, atol=1e-6), f"gradient mismatch for {name}"


# ====================================================================== #
# 4. The masking recipe
# ====================================================================== #
def test_mask_tokens_follows_the_bert_recipe():
    cpu.set_seed(11)
    ids = np.random.randint(5, 50, size=(400, 100))
    ids[:, 0], ids[:, -1] = 2, 3                 # [CLS] ... [SEP]
    inputs, labels = mask_tokens(ids, mask_id=4, vocab_size=50)
    chosen = labels != -100

    assert not chosen[:, [0, -1]].any()           # specials are never targets
    assert np.array_equal(labels[chosen], ids[chosen])
    assert np.array_equal(inputs[~chosen], ids[~chosen])
    assert abs(chosen[:, 1:-1].mean() - 0.15) < 0.01

    masked = (inputs[chosen] == 4).mean()
    kept = (inputs[chosen] == ids[chosen]).mean()
    assert abs(masked - 0.8) < 0.02 and abs(kept - 0.1) < 0.02


# ====================================================================== #
# 5. Behaviour
# ====================================================================== #
def test_learns_to_fill_a_mask_from_context():
    """Sequences count upwards from a random start: a, a+1, a+2, … (mod 10).

    One token is replaced by ``[MASK]``. Its value is uniformly random given its
    position alone (10 % accuracy at best), but fully determined by any
    neighbour plus the positions. Near-perfect accuracy on fresh sequences is
    therefore evidence that attention, positions and the MLM head work together.
    """
    vocab, seq, batch = 15, 6, 32

    def sample(rng):
        start = rng.randint(0, 10, size=(batch, 1))
        ids = 5 + (start + np.arange(seq)) % 10
        where = rng.randint(0, seq, size=batch)
        inputs, labels = ids.copy(), np.full_like(ids, -100)
        inputs[np.arange(batch), where] = 4
        labels[np.arange(batch), where] = ids[np.arange(batch), where]
        return inputs, labels

    cpu.set_seed(12)
    rng = np.random.RandomState(12)
    mlm = tiny_mlm(vocab, d_model=32, num_layers=2, num_heads=4, d_ff=64, max_len=seq)
    opt = Adam(mlm.parameters(), lr=3e-3)
    for _ in range(300):
        inputs, labels = sample(rng)
        opt.zero_grad()
        masked_lm_loss(mlm(inputs), labels).backward()
        opt.step()

    inputs, labels = sample(np.random.RandomState(999))
    target = labels != -100
    accuracy = (mlm(inputs).data.argmax(axis=-1)[target] == labels[target]).mean()
    assert accuracy > 0.95, accuracy


def test_overfit_tiny_batch():
    """Smoke test: the model can drive the loss down on a tiny fixed batch."""
    cpu.set_seed(13)
    mlm = tiny_mlm()
    ids = np.array([[2, 7, 9, 5, 3, 0], [2, 11, 6, 8, 10, 3]])
    am = (ids != 0).astype(int)
    labels = np.where(am == 1, ids, -100)
    opt = Adam(mlm.parameters(), lr=1e-2)
    first = float(masked_lm_loss(mlm(ids, attention_mask=am), labels).data)
    for _ in range(100):
        opt.zero_grad()
        loss = masked_lm_loss(mlm(ids, attention_mask=am), labels)
        loss.backward()
        opt.step()
    assert float(loss.data) < 0.05 * first, (first, float(loss.data))
