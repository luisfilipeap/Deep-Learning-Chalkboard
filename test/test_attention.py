"""Software tests for attention (``dlcb/attention.py``) and ``nn.Dropout``.

How do we know attention is *right*, and not merely *running*?
==============================================================
A test that only checks shapes passes for a wrong implementation as easily as
for a right one: ``softmax(k @ qᵀ)`` or a missing ``√d_k`` returns exactly the
same shape. So the tests below are organised as **five independent kinds of
evidence**. Each one could catch a bug that the others miss, and none of them
trusts the code it is checking:

1. **An independent reference.** The same formula written a second time, the
   slowest and most literal way possible — explicit Python loops over queries
   and keys, plain NumPy, no engine — straight from Vaswani et al. (2017), eq. 1.
   Two implementations that share no code and agree to 1e-12 are very unlikely
   to share a bug.
2. **Properties the maths guarantees.** Facts a human can derive on paper
   without running anything: every row of weights is a probability
   distribution; a masked key gets *exactly* zero weight; self-attention without
   positions is permutation-equivariant; a zero query attends uniformly, so the
   output is the plain mean of the values.
3. **Finite differences.** ``backward()`` against central differences (the same
   check as ``exercises/grading.py``). Finite differences only run the forward
   pass, so they know nothing about the engine's backward rules.
4. **Hand-computable cases.** A 2-token example small enough to verify with
   pencil and paper; the expected numbers are written in the test.
5. **Behaviour.** A tiny model must *learn* a task that is impossible without
   attending to the right token, and its attention weights must point at that
   token. This is the closest a test gets to "a human looks at it and sees it
   working".

Run them with::

    pytest test/test_attention.py
"""

import numpy as np
import pytest

from dlcb import engine as cpu
from dlcb import nn
from dlcb.attention import MultiHeadAttention, padding_mask, scaled_dot_product_attention
from dlcb.optim import Adam
from test.test_engine import numeric_gradient


# ====================================================================== #
# Independent references (plain NumPy, explicit loops, no engine)
# ====================================================================== #
def reference_attention(q, k, v, mask=None):
    """Attention for ONE head, written as literally as possible.

    ``q``: (seq_q, d_k), ``k``: (seq_k, d_k), ``v``: (seq_k, d_v), ``mask``:
    (seq_q, seq_k) additive or None. Returns (output, weights).
    """
    seq_q, d_k = q.shape
    seq_k = k.shape[0]
    weights = np.zeros((seq_q, seq_k))
    out = np.zeros((seq_q, v.shape[1]))
    for i in range(seq_q):
        scores = [sum(q[i, m] * k[j, m] for m in range(d_k)) / np.sqrt(d_k)
                  for j in range(seq_k)]
        if mask is not None:
            scores = [s + mask[i, j] for j, s in enumerate(scores)]
        top = max(scores)
        exps = [np.exp(s - top) for s in scores]
        total = sum(exps)
        for j in range(seq_k):
            weights[i, j] = exps[j] / total
            out[i] += weights[i, j] * v[j]
    return out, weights


def reference_multihead(mha: MultiHeadAttention, x: np.ndarray, mask=None):
    """Multi-head attention from the paper's definition, token by token.

    Uses the layer's weights (bias in row 0, per the ``Linear`` convention) but
    none of its code: per-token affine maps, an explicit loop over heads, and a
    plain concatenation of the head outputs.
    """
    def affine(layer, t):                     # t: (d,) -> (d_out,)
        W = layer.weight.data
        return W[0] + t @ W[1:]

    batch, seq, d = x.shape
    out = np.zeros_like(x)
    for b in range(batch):
        Q = np.stack([affine(mha.w_q, x[b, i]) for i in range(seq)])
        K = np.stack([affine(mha.w_k, x[b, i]) for i in range(seq)])
        V = np.stack([affine(mha.w_v, x[b, i]) for i in range(seq)])
        heads = []
        for h in range(mha.num_heads):
            cols = slice(h * mha.d_k, (h + 1) * mha.d_k)
            m = None if mask is None else np.broadcast_to(mask[b, 0], (seq, seq))
            heads.append(reference_attention(Q[:, cols], K[:, cols], V[:, cols], m)[0])
        concat = np.concatenate(heads, axis=1)
        out[b] = np.stack([affine(mha.w_o, concat[i]) for i in range(seq)])
    return out


def _qkv(batch=2, heads=3, seq_q=4, seq_k=5, d_k=6, d_v=7):
    q = cpu.Tensor(np.random.randn(batch, heads, seq_q, d_k))
    k = cpu.Tensor(np.random.randn(batch, heads, seq_k, d_k))
    v = cpu.Tensor(np.random.randn(batch, heads, seq_k, d_v))
    return q, k, v


# ====================================================================== #
# 1. Independent reference
# ====================================================================== #
def test_attention_matches_loop_reference():
    cpu.set_seed(0)
    q, k, v = _qkv()
    out, w = scaled_dot_product_attention(q, k, v, return_weights=True)
    assert out.shape == (2, 3, 4, 7) and w.shape == (2, 3, 4, 5)
    for b in range(2):
        for h in range(3):
            ref_out, ref_w = reference_attention(q.data[b, h], k.data[b, h], v.data[b, h])
            assert np.allclose(out.data[b, h], ref_out, atol=1e-12)
            assert np.allclose(w.data[b, h], ref_w, atol=1e-12)


def test_masked_attention_matches_loop_reference():
    cpu.set_seed(1)
    q, k, v = _qkv(batch=2, heads=1, seq_q=5, seq_k=5)
    is_pad = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 0, 1]], dtype=bool)
    mask = padding_mask(is_pad)
    out = scaled_dot_product_attention(q, k, v, mask=mask)
    for b in range(2):
        m = np.broadcast_to(mask.data[b, 0], (5, 5))
        ref_out, _ = reference_attention(q.data[b, 0], k.data[b, 0], v.data[b, 0], m)
        assert np.allclose(out.data[b, 0], ref_out, atol=1e-12)


def test_multihead_matches_reference():
    cpu.set_seed(2)
    mha = MultiHeadAttention(d_model=8, num_heads=2, dropout=0.0)
    x = np.random.randn(2, 5, 8)
    is_pad = np.array([[0, 0, 0, 0, 1], [0, 0, 1, 1, 1]], dtype=bool)
    mask = padding_mask(is_pad)
    out = mha(cpu.Tensor(x), mask=mask)
    assert out.shape == (2, 5, 8)
    assert np.allclose(out.data, reference_multihead(mha, x, mask.data), atol=1e-12)


# ====================================================================== #
# 2. Properties the maths guarantees
# ====================================================================== #
def test_weights_are_probability_distributions():
    cpu.set_seed(3)
    _, w = scaled_dot_product_attention(*_qkv(), return_weights=True)
    assert np.all(w.data >= 0)
    assert np.allclose(w.data.sum(axis=-1), 1.0)


def test_masked_keys_get_exactly_zero_weight_and_cannot_leak():
    """Padded keys get weight 0, so changing what is stored there changes nothing."""
    cpu.set_seed(4)
    q, k, v = _qkv(batch=1, heads=2, seq_q=4, seq_k=4)
    mask = padding_mask([[0, 0, 1, 1]])
    out, w = scaled_dot_product_attention(q, k, v, mask=mask, return_weights=True)
    assert np.all(w.data[..., 2:] == 0.0)

    k2, v2 = k.data.copy(), v.data.copy()
    k2[..., 2:, :] = 1e3 * np.random.randn(*k2[..., 2:, :].shape)
    v2[..., 2:, :] = 1e3 * np.random.randn(*v2[..., 2:, :].shape)
    out2 = scaled_dot_product_attention(q, cpu.Tensor(k2), cpu.Tensor(v2), mask=mask)
    assert np.allclose(out.data, out2.data, atol=1e-12)


def test_zero_query_attends_uniformly_to_the_mean_value():
    """q = 0 gives equal scores, so the output is the plain average of the values."""
    cpu.set_seed(5)
    _, k, v = _qkv(batch=1, heads=1, seq_q=3, seq_k=6)
    q = cpu.zeros(1, 1, 3, 6)
    out = scaled_dot_product_attention(q, k, v)
    assert np.allclose(out.data, v.data.mean(axis=-2, keepdims=True))


def test_self_attention_is_permutation_equivariant():
    """Without positional information, shuffling the tokens just shuffles the output."""
    cpu.set_seed(6)
    mha = MultiHeadAttention(d_model=8, num_heads=4, dropout=0.0)
    x = np.random.randn(1, 6, 8)
    perm = np.array([3, 0, 5, 1, 4, 2])
    out = mha(cpu.Tensor(x)).data
    out_perm = mha(cpu.Tensor(x[:, perm])).data
    assert np.allclose(out_perm, out[:, perm], atol=1e-12)


def test_split_and_merge_heads_are_inverse():
    cpu.set_seed(7)
    mha = MultiHeadAttention(d_model=12, num_heads=3, dropout=0.0)
    x = cpu.randn(2, 5, 12)
    split = mha._split_heads(x)
    assert split.shape == (2, 3, 5, 4)
    # Head h of token i holds features [h*d_k, (h+1)*d_k) of that token.
    assert np.array_equal(split.data[1, 2, 3], x.data[1, 3, 8:12])
    assert np.array_equal(mha._merge_heads(split).data, x.data)


def test_num_heads_must_divide_d_model():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=10, num_heads=3)


def test_last_weights_are_recorded_for_inspection():
    cpu.set_seed(8)
    mha = MultiHeadAttention(d_model=8, num_heads=2, dropout=0.0)
    mha(cpu.randn(3, 4, 8))
    assert mha.last_weights.shape == (3, 2, 4, 4)
    assert np.allclose(mha.last_weights.sum(axis=-1), 1.0)


# ====================================================================== #
# 3. Finite differences
# ====================================================================== #
def test_attention_gradcheck_with_mask():
    cpu.set_seed(9)
    q, k, v = _qkv(batch=1, heads=2, seq_q=3, seq_k=4, d_k=3, d_v=2)
    mask = padding_mask([[0, 0, 0, 1]])
    target = np.random.randn(1, 2, 3, 2)

    def loss():
        return ((scaled_dot_product_attention(q, k, v, mask=mask) - target) ** 2).mean()

    for t in (q, k, v):
        t.zero_grad()
    loss().backward()
    for name, t in (("q", q), ("k", k), ("v", v)):
        numeric = numeric_gradient(lambda: float(loss().data), t.data)
        assert np.allclose(t.grad, numeric, atol=1e-6), f"gradient mismatch for {name}"
    # The padded key receives no gradient at all, not just a small one.
    assert np.all(k.grad[..., 3, :] == 0.0) and np.all(v.grad[..., 3, :] == 0.0)


def test_multihead_gradcheck_input_and_every_parameter():
    cpu.set_seed(10)
    mha = MultiHeadAttention(d_model=4, num_heads=2, dropout=0.0)
    x = cpu.Tensor(np.random.randn(2, 3, 4))
    mask = padding_mask([[0, 0, 0], [0, 0, 1]])
    target = np.random.randn(2, 3, 4)

    def loss():
        return ((mha(x, mask=mask) - target) ** 2).mean()

    for t in (x, *mha.parameters()):
        t.zero_grad()
    loss().backward()
    params = [("x", x), *mha.named_parameters()]
    assert len(params) == 5                          # x + w_q, w_k, w_v, w_o
    for name, t in params:
        numeric = numeric_gradient(lambda: float(loss().data), t.data)
        assert np.allclose(t.grad, numeric, atol=1e-6), f"gradient mismatch for {name}"


# ====================================================================== #
# 4. A case small enough to check by hand
# ====================================================================== #
def test_two_token_example_by_hand():
    """One query, two keys, d_k = 1 — numbers chosen so the softmax is 1/5, 4/5.

    q = 2, keys = [0, ln 2], values = [10, 20]:

        scores  = q·k / √1 = [2·0, 2·ln 2] = [0, ln 4]
        weights = [e^0, e^ln4] / (1 + 4) = [1/5, 4/5]
        output  = 1/5 · 10 + 4/5 · 20 = 18
    """
    q = cpu.Tensor([[[2.0]]])
    k = cpu.Tensor([[[0.0], [np.log(2.0)]]])
    v = cpu.Tensor([[[10.0], [20.0]]])
    out, w = scaled_dot_product_attention(q, k, v, return_weights=True)
    assert np.allclose(w.data, [[[0.2, 0.8]]])
    assert np.allclose(out.data, [[[18.0]]])


# ====================================================================== #
# 5. Behaviour: attention must learn to look at the right token
# ====================================================================== #
def test_learns_to_retrieve_the_token_with_the_largest_first_feature():
    """Each output must equal the token whose feature 0 is largest.

    No per-token map can solve this: the answer for position ``i`` lives in a
    *different* token, chosen by comparing all of them. A single attention
    layer can — keys must grow with feature 0 and every query must ask for
    "large" — so after training the attention weights should point at the
    argmax token. We check both the loss and *where it looks*.
    """
    cpu.set_seed(11)
    batch, seq, d = 64, 5, 4
    x_np = np.random.randn(batch, seq, d)
    target_idx = x_np[:, :, 0].argmax(axis=1)
    y_np = np.repeat(x_np[np.arange(batch), target_idx][:, None, :], seq, axis=1)
    x, y = cpu.Tensor(x_np, requires_grad=False), cpu.Tensor(y_np, requires_grad=False)

    mha = MultiHeadAttention(d_model=d, num_heads=1, dropout=0.0)
    opt = Adam(mha.parameters(), lr=0.05)

    def loss():
        return ((mha(x) - y) ** 2).mean()

    first = float(loss().data)
    for _ in range(300):
        mha.zero_grad()
        current = loss()
        current.backward()
        opt.step()
    last = float(loss().data)

    on_target = mha.last_weights[np.arange(batch), 0, :, target_idx]   # (batch, seq)
    assert last < 0.1 * first, (first, last)
    assert on_target.mean() > 0.8, on_target.mean()


# ====================================================================== #
# Dropout
# ====================================================================== #
def test_dropout_is_identity_in_eval_mode_and_for_p_zero():
    x = cpu.randn(4, 5)
    assert nn.Dropout(0.5).eval()(x) is x
    assert nn.Dropout(0.0)(x) is x


def test_dropout_zeroes_about_p_and_preserves_the_mean():
    cpu.set_seed(12)
    x = cpu.ones(200, 500)
    y = nn.Dropout(0.3)(x).data
    assert abs((y == 0).mean() - 0.3) < 0.01
    assert set(np.unique(y)) <= {0.0, 1.0 / 0.7}
    assert abs(y.mean() - 1.0) < 0.01


def test_dropout_gradient_follows_the_mask():
    cpu.set_seed(13)
    x = cpu.randn(6, 7)
    y = nn.Dropout(0.5)(x)
    y.sum().backward()
    kept = y.data != 0
    assert np.allclose(x.grad[kept], 2.0) and np.all(x.grad[~kept] == 0.0)


def test_dropout_rejects_invalid_p():
    with pytest.raises(ValueError):
        nn.Dropout(1.0)
