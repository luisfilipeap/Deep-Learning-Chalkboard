"""Attention mechanisms for the Transformer encoder.

Attention is the one genuinely new idea the Transformer adds on top of the layers
already in this library. Everything else in an encoder block — ``Linear``,
residual additions, ``LayerNorm``, GELU — has appeared before. And attention
itself turns out to need **no new backward rule**: it is a composition of ops the
engine already differentiates (``@``, ``*``, ``+``, ``softmax``, ``reshape`` and
``transpose``). That is the didactic point of this module: once you trust those
few ops, you can trust attention, because the gradient is assembled by the same
chain rule you watched in ``learn/viz01_engine``.

The idea in one paragraph
-------------------------
Each token in a sequence asks a question (its *query* ``q``), advertises what it
contains (its *key* ``k``) and offers a payload (its *value* ``v``). Token ``i``
scores every token ``j`` by how well its question matches their key,
``s_ij = q_i · k_j``, turns the scores into a probability distribution with a
softmax, and returns the **weighted average of the values**::

    out_i = Σ_j  softmax_j( q_i · k_j / sqrt(d_k) )  v_j

So attention is a *soft, differentiable dictionary lookup*: with a hard argmax it
would return the one value whose key best matches the query; the softmax makes
the lookup smooth enough to train with gradients (Bahdanau et al. [1];
Vaswani et al. [2]).

Layout
------
Unlike ``Linear`` (column-oriented ``(features, batch)``), attention uses the
token-major layout of ``Embedding`` and of the BERT literature:
``(batch, seq, d_model)``, and ``(batch, heads, seq, d_k)`` once split into
heads. The engine's ``@`` is a *batched* matmul over the leading axes, so one
``@`` computes the scores for every batch element and every head at once.

References
----------
[1] D. Bahdanau, K. Cho and Y. Bengio, "Neural Machine Translation by Jointly
    Learning to Align and Translate," ICLR 2015. arXiv:1409.0473.
[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
    Ł. Kaiser and I. Polosukhin, "Attention Is All You Need," NeurIPS 2017.
    arXiv:1706.03762.
[3] J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, "BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding," NAACL 2019.
    arXiv:1810.04805.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from dlcb.engine import Tensor
from dlcb.nn import Dropout, Linear, Module, tokenwise


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[Dropout] = None,
    return_weights: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute ``softmax(q @ kᵀ / sqrt(d_k) + mask) @ v`` (Vaswani et al. [2], eq. 1).

    The computation is four engine ops, and each one has a plain meaning::

        scores  = q @ kᵀ            # (…, seq_q, seq_k): s_ij = q_i · k_j
        scores  = scores / √d_k     # keep the scores' variance near 1
        weights = softmax(scores)   # each row i: a distribution over keys j
        out     = weights @ v       # each row i: a weighted average of values

    Why divide by ``√d_k``?
    -----------------------
    If the ``d_k`` components of ``q`` and ``k`` are independent with mean 0 and
    variance 1, their dot product ``Σ_m q_m k_m`` has variance ``d_k``. For large
    ``d_k`` the scores spread far apart, the softmax saturates into a near
    one-hot vector, and its gradient ``sm * (g - Σ g·sm)`` all but vanishes.
    Dividing by ``√d_k`` brings the variance back to 1, whatever the head size
    ([2], footnote 4).

    Masking
    -------
    ``mask`` is **added** to the scores before the softmax. A ``0`` leaves a
    score untouched; ``-inf`` sends it to ``exp(-inf) = 0``, so that key gets
    *exactly* zero weight — and, since the softmax backward multiplies by the
    weight, exactly zero gradient as well. BERT uses this to hide ``[PAD]``
    tokens ([3]); see :func:`padding_mask`. A row whose keys are *all* masked has
    no distribution to form and produces NaN — never mask every key of a query.

    Parameters
    ----------
    q : Tensor
        Queries, shape ``(…, seq_q, d_k)``.
    k : Tensor
        Keys, shape ``(…, seq_k, d_k)``.
    v : Tensor
        Values, shape ``(…, seq_k, d_v)``.
    mask : Tensor, optional
        Additive mask broadcastable to ``(…, seq_q, seq_k)`` (``0`` = keep,
        ``-inf`` = hide). It is a constant: no gradient flows into it.
    dropout : Dropout, optional
        Dropout applied to the attention *weights* (as in [2]); it is inactive
        in eval mode.
    return_weights : bool
        If ``True`` also return the attention weights, shape
        ``(…, seq_q, seq_k)`` — the matrix you inspect to see *what attends to
        what*.

    Returns
    -------
    Tensor, or (Tensor, Tensor)
        The output ``(…, seq_q, d_v)``, and the weights if requested.
    """
    d_k = q.shape[-1]
    scores = (q @ k.T) * (1.0 / np.sqrt(d_k))   # k.T swaps only the last two axes
    if mask is not None:
        scores = scores + mask
    weights = scores.softmax(axis=-1)
    attended = dropout(weights) if dropout is not None else weights
    out = attended @ v
    if return_weights:
        return out, weights
    return out


def padding_mask(is_pad) -> Tensor:
    """Build the additive attention mask that hides padding tokens.

    Sentences in a batch have different lengths, so the shorter ones are
    right-padded with ``[PAD]`` (id 0) to a common length. Those positions are
    filler, not text: no token should read from them. This function turns the
    "which positions are padding" flags into the ``mask`` that
    ``BERTModel.forward`` builds once and that ``TransformerEncoder`` passes,
    unchanged, to every ``EncoderLayer`` and on into the attention.

    From flags to mask
    ------------------
    ::

        ids            = [[ [CLS]  the  cat  [SEP]  [PAD]  [PAD] ]]
        attention_mask = [[   1     1    1     1      0      0   ]]   # 1 = real, 0 = padding
        is_pad         = attention_mask == 0
        padding_mask   ->  [   0     0    0     0    -inf   -inf  ]   # shape (1, 1, 1, 6)

    The two size-1 axes broadcast the same row over every **head** and every
    **query**, so the mask applies to the whole ``(batch, heads, seq, seq)``
    score tensor.

    How it acts
    -----------
    :func:`scaled_dot_product_attention` **adds** it to the scores before the
    softmax::

        scores  = q @ kᵀ / √d_k          # (batch, heads, seq, seq)
        scores  = scores + mask          # -inf in the padded key columns
        weights = softmax(scores)        # exp(-inf) = 0  ->  weight exactly 0

    so the weight matrix (rows = queries, columns = keys) looks like::

                  [CLS]  the   cat  [SEP] [PAD] [PAD]
         [CLS]  [  .2    .3    .3    .2    0     0  ]
         the    [  .1    .5    .3    .1    0     0  ]
         ...
         [PAD]  [  .2    .2    .4    .2    0     0  ]   <- a pad is still a query,
                                                            but only reads real tokens

    Why *add* ``-inf`` before the softmax rather than zero the weights after it?

    * **The distribution stays normalised.** Zeroing weights afterwards would
      leave rows that no longer sum to 1; with ``-inf`` beforehand, the softmax
      spreads all the probability over the real tokens.
    * **The gradient is zero too.** The softmax backward multiplies by the
      weight, so a masked position receives *exactly* zero gradient
      (``test/test_attention.py::test_attention_gradcheck_with_mask``).

    Only *keys* are masked, never queries: a query whose keys were all masked
    would give ``softmax`` a row of ``-inf`` — ``0 / 0`` — and produce NaN. The
    hidden states computed at padded positions are simply ignored downstream
    (their MLM labels are ``ignore_index``).

    The papers
    ----------
    * **Vaswani et al., "Attention Is All You Need," 2017, §3.2.3** is the
      origin of the mechanism. It masks the *decoder* so that a position cannot
      see the ones after it: "We implement this inside of scaled dot-product
      attention by masking out (setting to −∞) all values in the input of the
      softmax which correspond to illegal connections." That is the triangular
      **causal** mask (token ``i`` sees only ``j ≤ i``). This function uses the
      same trick for a different rule: nobody sees ``[PAD]``.
    * **Devlin et al., "BERT," 2019.** *Bidirectional* means BERT has **no**
      causal mask — every token sees both its left and its right — so only the
      padding mask remains. The paper's text does not describe it; it is in the
      official code (``modeling.py`` of google-research/bert), as
      ``adder = (1.0 - attention_mask) * -10000.0``. ``-10000`` instead of
      ``-inf`` avoids NaN in float16 and is equivalent in practice, since
      ``exp(-10000) ≈ 0``.

    Not to be confused with ``[MASK]``
    ----------------------------------
    BERT has two unrelated "masks":

    ==================  ==================================  ===================================
    \                   attention mask (this function)      MLM masking (``[MASK]``)
    ==================  ==================================  ===================================
    decides             **who may look at whom**            **which input tokens are hidden**
    acts on             the attention scores                the input ids, before the model
    in this library     ``padding_mask``, the ``mask``      ``transformer.mask_tokens``
                        argument                            (token id 4)
    paper               Vaswani §3.2.3 (the mechanism)      Devlin §3.1, after the *Cloze*
                                                            task (Taylor, 1953)
    ==================  ==================================  ===================================

    A ``[MASK]`` token is **not** hidden by the attention mask: it is a real
    position that attends to the whole context — which is exactly how the model
    works out what word was there.

    Parameters
    ----------
    is_pad : array-like of bool, shape ``(batch, seq)``
        ``True`` where the token is ``[PAD]``.

    Returns
    -------
    Tensor
        Constant (``requires_grad=False``) of shape ``(batch, 1, 1, seq)``:
        ``-inf`` at padded *keys*, ``0`` elsewhere.
    """
    is_pad = np.asarray(is_pad, dtype=bool)
    mask = np.where(is_pad, -np.inf, 0.0)[:, None, None, :]
    return Tensor(mask, requires_grad=False)


class MultiHeadAttention(Module):
    """Multi-head self-attention (Vaswani et al. [2], §3.2.2).

    One attention head can only form one distribution per query — one "kind of
    question". Multi-head attention runs ``num_heads`` of them in parallel, each
    in its own ``d_k = d_model / num_heads``-dimensional subspace, so that one
    head can, say, follow syntax while another follows coreference::

        Q, K, V   = x W_q, x W_k, x W_v             # (batch, seq, d_model) each
        Q, K, V   = split_heads(·)                  # (batch, heads, seq, d_k)
        heads     = attention(Q, K, V, mask)        # (batch, heads, seq, d_k)
        out       = merge_heads(heads) W_o          # (batch, seq, d_model)

    Splitting into heads costs nothing: it is a ``reshape`` plus a
    ``transpose`` (zero FLOPs, see the engine's counter), and the total
    parameter count is the same as for a single head of width ``d_model``.

    How the projections reuse ``Linear``
    ------------------------------------
    ``Linear`` is column-oriented: it maps ``(in_features, n)`` to
    ``(out_features, n)`` with the bias folded into row 0 of its weight. A
    sequence is just ``batch * seq`` token vectors, so each projection lays every
    token out as one *column*, applies the layer, and folds the result back::

        (batch, seq, d) -> (batch*seq, d) -> ᵀ -> (d, batch*seq) -> Linear -> ᵀ -> back

    No new backward rule is needed anywhere in this class.

    Parameters
    ----------
    d_model : int
        Width of the token vectors.
    num_heads : int
        Number of heads; must divide ``d_model``.
    dropout : float
        Dropout probability on the attention weights (``0`` disables it).

    Attributes
    ----------
    last_weights : np.ndarray or None
        The attention weights ``(batch, heads, seq, seq)`` of the most recent
        forward pass, kept as a plain array for inspection and visualisation.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
        self.dropout = Dropout(dropout)
        self.last_weights: Optional[np.ndarray] = None

    def _project(self, layer: Linear, x: Tensor) -> Tensor:
        """Apply a column-oriented ``Linear`` to every token (see ``nn.tokenwise``)."""
        return tokenwise(layer, x)

    def _split_heads(self, x: Tensor) -> Tensor:
        """Reshape ``(batch, seq, d_model)`` -> ``(batch, heads, seq, d_k)``.

        The ``d_model`` features of each token are cut into ``num_heads``
        contiguous blocks of ``d_k``; the transpose then brings the head axis in
        front of ``seq`` so the batched ``@`` treats every head independently.
        """
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Inverse of :meth:`_split_heads`: ``(batch, heads, seq, d_k)`` -> ``(batch, seq, d_model)``."""
        batch, _, seq, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Self-attention over ``x`` of shape ``(batch, seq, d_model)``.

        ``mask`` is additive and must broadcast to ``(batch, heads, seq, seq)`` —
        e.g. the ``(batch, 1, 1, seq)`` tensor from :func:`padding_mask`.
        """
        q = self._split_heads(self._project(self.w_q, x))
        k = self._split_heads(self._project(self.w_k, x))
        v = self._split_heads(self._project(self.w_v, x))
        heads, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, return_weights=True
        )
        self.last_weights = weights.data.copy()
        return self._project(self.w_o, self._merge_heads(heads))
