"""Transformer encoder and BERT-style model assembly.

A miniBERT is three ideas stacked:

    ids ──► BERTEmbeddings ──► EncoderLayer × N ──► MLM head ──► logits over the vocabulary
            token + position     attention + FFN      Linear, GELU,
            + segment, LayerNorm  each wrapped in      LayerNorm, tied
                                  residual + LayerNorm decoder

* **Embeddings** are lookups (``nn.Embedding``) — one table for the token, one
  for its position, one for its segment — summed, so every token vector carries
  *what* it is and *where* it is. 

* **The encoder layer** tokens exchange information through
  multi-head self-attention, then each token is transformed on its own by a
  two-layer feed-forward network. Both sub-blocks are *residual*,
  ``x + f(x)``, so the gradient always has an identity path back to the input,
  and both are followed by ``LayerNorm`` (the "post-norm" layout of the original
  Transformer and BERT).

* **The MLM head** turns each final hidden state into a score for every word in
  the vocabulary. Its output matrix is the token embedding table itself
  (*weight tying*): the same vector represents a word on the way in and on the
  way out.

Everything is composed from layers already in the library, so there is still
no new backward rule: the whole miniBERT is differentiated by the chain rule
over ``@``, ``+``, ``*``, ``softmax``, ``mean``, ``var``, ``sqrt``, ``gelu``,
``reshape``, ``transpose`` and indexing.

Layout: activations are token-major, ``(batch, seq, d_model)``. ``Linear`` is
column-oriented, so it is applied token by token through ``nn.tokenwise``.

References
----------
[1] A. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
    arXiv:1706.03762.
[2] J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, "BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding," NAACL 2019.
    arXiv:1810.04805.
[3] O. Press and L. Wolf, "Using the Output Embedding to Improve Language
    Models," EACL 2017. arXiv:1608.05859.
[4] R. Xiong et al., "On Layer Normalization in the Transformer Architecture,"
    ICML 2020. arXiv:2002.04745.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

from dlcb.attention import MultiHeadAttention, padding_mask
from dlcb.engine import Tensor
from dlcb.nn import Dropout, Embedding, LayerNorm, Linear, Module, Parameter, tokenwise


class PositionwiseFeedForward(Module):
    """Two-layer feed-forward network applied to each token independently.

    ``FFN(x) = W_2ᵀ GELU(W_1ᵀ x)``, with ``W_1: d_model -> d_ff`` and
    ``W_2: d_ff -> d_model`` (biases folded in, as everywhere in this library).
    Attention *mixes* tokens; this block *transforms* each one, with the same
    weights at every position. BERT uses ``d_ff = 4 * d_model`` and GELU ([2]).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq, d_model), token-major. fc1 and fc2 are column-oriented
        # Linears, so ``tokenwise`` lays the batch*seq tokens out as columns,
        # applies the layer to all of them in one matmul, and puts each result
        # back at its (sentence, position). No token sees another one here —
        # that is what "position-wise" means; mixing tokens is attention's job.

        # Expand: each token d_model -> d_ff (4x wider), then GELU elementwise.
        # (batch, seq, d_model) -> (batch, seq, d_ff)
        hidden = self.dropout(tokenwise(self.fc1, x).gelu())

        # Compress: each token d_ff -> d_model, back to the width the residual
        # connection ``x + FFN(x)`` in EncoderLayer needs.
        # (batch, seq, d_ff) -> (batch, seq, d_model)
        return tokenwise(self.fc2, hidden)


class EncoderLayer(Module):
    """A single Transformer encoder block (post-norm, as in BERT [1, 2])::

        x = LayerNorm(x + Dropout(MultiHeadAttention(x)))
        x = LayerNorm(x + Dropout(FeedForward(x)))

    The residual ``x + …`` gives the gradient a path that skips the sub-block
    entirely (its local derivative is the identity), which is what lets deep
    stacks train. Most modern models move the LayerNorm *before* each sub-block
    ("pre-norm"), which trains more stably without learning-rate warm-up [4];
    swapping the two lines in ``forward`` is a good exercise.
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm1(x + self.drop1(self.attention(x, mask)))
        return self.norm2(x + self.drop2(self.ffn(x)))


class TransformerEncoder(Module):
    """A stack of ``num_layers`` :class:`EncoderLayer` blocks, applied in order.

        x_0 = embeddings                         (batch, seq, d_model)
        x_1 = EncoderLayer_1(x_0, mask)          (batch, seq, d_model)
        ...
        x_N = EncoderLayer_N(x_{N-1}, mask)      (batch, seq, d_model)  -> the output

    Why the shape never changes
    ---------------------------
    Every block maps ``(batch, seq, d_model)`` to the same shape — it has to,
    because of its residual ``x + f(x)``. That is what makes the blocks
    *stackable*: any number of them can be chained, and the depth is a single
    hyper-parameter. Each block keeps one vector per token and *refines* it:
    the output of layer ``k`` is the input of layer ``k + 1``, so the token
    vectors become more and more contextual as they go up.

    What depth buys (and what it does not)
    --------------------------------------
    Already after **one** layer every token has seen every other token —
    attention connects all pairs directly. Depth is therefore not about *reach*
    (in a convolutional network the receptive field grows layer by layer; here
    it is complete from the start) but about **composition**: layer 2 attends
    over vectors that are already summaries of the whole sentence, so it can
    combine relations that layer 1 found separately.

    The same mask at every layer
    ----------------------------
    ``mask`` (the ``-inf`` padding mask) is passed to **every** block, not only
    the first. Padded positions still get a hidden state at each layer — they
    are queries too, and attend to the real tokens — so if any single layer
    forgot the mask, the real tokens of that layer would read from them and the
    padding would leak into the result
    (``test/test_model.py::test_padding_cannot_change_real_tokens``).

    Independent weights per layer
    -----------------------------
    The list comprehension calls ``EncoderLayer(...)`` ``num_layers`` times, so
    each block has its **own** parameters. Writing ``[EncoderLayer(...)] *
    num_layers`` instead would put the *same* object in the list N times: one
    set of weights applied N times. That is a real design — ALBERT shares its
    layers this way to cut parameters (Lan et al., ICLR 2020, arXiv:1909.11942)
    — but it is a different model, and ``parameters()`` would then list the
    shared weights N times. Because ``layers`` is a list attribute, ``Module``
    finds every block's parameters on its own, named ``layers.0.…``,
    ``layers.1.…``, and so on.

    Cost
    ----
    Per layer, the four attention projections hold ``4 d²`` weights and the
    feed-forward network ``2 · d · 4d = 8 d²``: about ``12 d²`` parameters
    (plus biases and LayerNorm). For each token that is ``24 d²`` FLOPs for the
    matmuls, plus ``4 · seq · d`` for the attention scores and the weighted sum —
    the only term that grows with the sequence length. Parameters and compute
    therefore grow **linearly** with ``num_layers``.

    ============  ======  =======  =====  ======  ==========
    model         layers  d_model  heads  d_ff    parameters
    ============  ======  =======  =====  ======  ==========
    BERT-base       12      768     12    3072    110 M
    BERT-large      24     1024     16    4096    340 M
    this default     2      128      2     512    ~0.4 M (encoder only)
    ============  ======  =======  =====  ======  ==========

    Parameters
    ----------
    num_layers : int
        Number of stacked blocks ``N``.
    d_model, num_heads, d_ff, dropout
        Passed unchanged to every :class:`EncoderLayer`.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        # N separate calls -> N blocks with their own weights (not one shared block).
        self.layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Each block refines the token vectors; the padding mask goes to every one.
        for layer in self.layers:
            x = layer(x, mask)        # (batch, seq, d_model) -> (batch, seq, d_model)
        return x


class BERTEmbeddings(Module):
    """Token + position + segment embeddings, summed, then LayerNorm and Dropout.

    The model's input layer: it turns integer ids into the vectors the encoder
    works on (Devlin et al. [2], §3, Figure 2). Three learned tables, each an
    ``nn.Embedding``, answer three different questions about a token:

    ==========  ===============================  ====================  =================
    table       answers                          shape                 looked up with
    ==========  ===============================  ====================  =================
    token       *what* is this token?            (vocab_size, d)       ``input_ids``
    position    *where* in the sequence is it?   (max_len, d)          ``0 … seq-1``
    segment     *which sentence* is it part of?  (type_vocab_size, d)  ``token_type_ids``
    ==========  ===============================  ====================  =================

    and the output is::

        e = token[ids] + position[0 … seq-1] + segment[token_type_ids]   # (batch, seq, d)
        e = Dropout(LayerNorm(e))

    An example, for a sentence pair::

        tokens          [CLS]  my   dog  [SEP]   it  barks [SEP]
        input_ids         2    17   54     3     29   88     3
        positions         0     1    2     3      4    5     6
        token_type_ids    0     0    0     0      1    1     1

        e[i] = token[input_ids[i]] + position[i] + segment[token_type_ids[i]]

    Why positions are needed
    ------------------------
    Attention by itself is **permutation-equivariant**: shuffle the tokens and
    the outputs are shuffled the same way, so "dog bites man" and "man bites
    dog" would be the same bag of vectors
    (``test/test_attention.py::test_self_attention_is_permutation_equivariant``).
    Adding ``position[i]`` makes the same word at a different place a different
    vector, and that is the *only* source of word order in the whole model
    (``test/test_model.py::test_positions_break_the_permutation_symmetry``).

    BERT **learns** the position table, one row per position up to ``max_len``,
    instead of the fixed sinusoids of the original Transformer (Vaswani et al.
    [1], §3.5, who report nearly identical results for both). The price is a
    hard limit: a sequence longer than ``max_len`` has no position vectors, and
    ``forward`` raises ``ValueError``. Later models encode *relative*
    positions inside attention instead (Shaw, Uszkoreit & Vaswani, NAACL 2018,
    arXiv:1803.02155; RoPE, Su et al., arXiv:2104.09864).

    Why a *sum* and not a concatenation
    -----------------------------------
    Summing looks like it would blur the three pieces of information
    together, but it is exactly equivalent to concatenating three one-hot
    vectors and applying one linear layer. With ``W`` stacked from the three
    tables::

        W · [onehot(token) ; onehot(position) ; onehot(segment)]
          = token[id] + position[i] + segment[s]

    so the sum *is* a learned linear map of all three, at the cost of ``d``
    numbers per token instead of ``3d``. The model is free to keep the three
    contributions in different directions of the ``d``-dimensional space, and
    the layers above can tell them apart.

    Segments and sentence pairs
    ---------------------------
    Some tasks take two sentences (question + passage, premise + hypothesis).
    They are joined as ``[CLS] A [SEP] B [SEP]``, and ``token_type_ids`` marks
    each token 0 (sentence A) or 1 (sentence B), so ``type_vocab_size = 2``.
    For a single sentence it defaults to all zeros, and the segment table then
    just adds the same vector to every token.

    LayerNorm and Dropout
    ---------------------
    The tables start tiny (``nn.Embedding`` draws from ``N(0, 0.02²)``), so the
    raw sum has a variance far below 1. ``LayerNorm`` standardises every token
    vector — mean 0, variance 1 across its ``d`` features — so the first
    encoder layer receives inputs on the scale it expects; Dropout then
    regularises them during training.

    Gradients
    ---------
    There is no new backward rule. Each lookup is the engine's indexing op,
    whose backward ``np.add.at`` *accumulates*: a token id used ten times in a
    batch gets the sum of ten gradients in its row. The position term has shape
    ``(seq, d)`` and is broadcast over the batch, so the backward of ``+``
    sums the gradient of position ``i`` over all sentences into
    ``position[i]``.

    ``[PAD]`` is id 0, and its token row starts at zero (``padding_idx=0``). It
    is still an ordinary learnable row; what keeps padding out of the real
    tokens is the attention mask, not this table.

    Size
    ----
    ``(vocab_size + max_len + type_vocab_size) · d`` weights plus ``2d`` for the
    LayerNorm. In BERT-base, ``(30 522 + 512 + 2) · 768 ≈ 23.8 M`` — about a
    fifth of its 110 M parameters sit in this layer, almost all in the token
    table.

    Parameters
    ----------
    vocab_size : int
        Number of token ids (rows of the token table), special tokens included.
    d_model : int
        Width ``d`` of every embedding vector.
    max_len : int
        Longest sequence the model accepts (rows of the position table).
    type_vocab_size : int
        Number of segments (2: sentence A and sentence B).
    dropout : float
        Dropout probability applied after the LayerNorm.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        self.max_len = max_len
        self.token = Embedding(vocab_size, d_model, padding_idx=0)
        self.position = Embedding(max_len, d_model)
        self.segment = Embedding(type_vocab_size, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None) -> Tensor:
        ids = np.asarray(input_ids.data if isinstance(input_ids, Tensor) else input_ids, dtype=int)
        seq = ids.shape[1]
        if seq > self.max_len:
            raise ValueError(f"sequence length {seq} exceeds max_len {self.max_len}")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(ids)
        e = self.token(ids) + self.position(np.arange(seq)) + self.segment(token_type_ids)
        return self.dropout(self.norm(e))


class BERTModel(Module):
    """BERT-style bidirectional Transformer encoder [2].

    Embeds the input ids, runs the encoder stack, and returns the per-token
    contextual representations, shape ``(batch, seq, d_model)``. "Bidirectional"
    means every token attends to every other — left and right — except
    padding.

    ``attention_mask`` follows the usual convention: shape ``(batch, seq)``, 1
    for a real token and 0 for ``[PAD]``. It becomes the additive ``-inf`` mask
    of :func:`dlcb.attention.padding_mask`, so no token can attend to padding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        d_ff: int = 512,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embeddings = BERTEmbeddings(vocab_size, d_model, max_len, dropout=dropout)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None) -> Tensor:
        mask = None
        if attention_mask is not None:
            mask = padding_mask(np.asarray(attention_mask) == 0)
        return self.encoder(self.embeddings(input_ids, token_type_ids), mask)


class BERTForMaskedLM(Module):
    """BERT encoder with a masked-language-modelling head for pretraining [2].

    For each token's final hidden state ``h``::

        t      = LayerNorm(GELU(W_tᵀ h))          # one more per-token transform
        logits = t @ Eᵀ + b                       # E: the token embedding table

    The decoder reuses ``E`` (*weight tying* [3]): a word's input vector and its
    output vector are the same parameter, which halves the largest matrix in a
    small model and gives the embedding a gradient from both ends. The table is
    read from ``self.bert`` at forward time rather than stored a second time, so
    ``parameters()`` lists it once and the optimiser updates it once.
    """

    def __init__(self, bert: BERTModel, vocab_size: int) -> None:
        if vocab_size != bert.vocab_size:
            raise ValueError("the MLM head must predict the encoder's own vocabulary")
        self.bert = bert
        self.transform = Linear(bert.d_model, bert.d_model)
        self.norm = LayerNorm(bert.d_model)
        self.bias = Parameter(np.zeros(vocab_size, dtype=bert.embeddings.token.weight.data.dtype))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None) -> Tensor:
        """Return vocabulary logits of shape ``(batch, seq, vocab_size)``."""
        h = self.bert(input_ids, attention_mask, token_type_ids)
        t = self.norm(tokenwise(self.transform, h).gelu())
        return t @ self.bert.embeddings.token.weight.T + self.bias


def mask_tokens(
    input_ids,
    mask_id: int,
    vocab_size: int,
    special_ids: Iterable[int] = (0, 1, 2, 3, 4),
    p: float = 0.15,
    ignore_index: int = -100,
) -> Tuple[np.ndarray, np.ndarray]:
    """BERT's masking recipe for MLM pretraining [2, §3.1].

    Chooses ``p`` (15 %) of the non-special tokens as prediction targets. Of
    those, 80 % are replaced by ``[MASK]``, 10 % by a random token and 10 % are
    left unchanged — so the model cannot learn that "only ``[MASK]`` positions
    matter" and must keep a useful representation of every token.

    Returns
    -------
    (inputs, labels)
        ``inputs``: the corrupted ids. ``labels``: the original id at every
        chosen position and ``ignore_index`` everywhere else, ready for
        :func:`dlcb.loss.masked_lm_loss`. Randomness uses NumPy's global RNG,
        so ``set_seed`` makes it reproducible.
    """
    ids = np.asarray(input_ids, dtype=int)
    candidates = ~np.isin(ids, list(special_ids))
    chosen = candidates & (np.random.rand(*ids.shape) < p)

    labels = np.where(chosen, ids, ignore_index)
    inputs = ids.copy()
    action = np.random.rand(*ids.shape)
    inputs[chosen & (action < 0.8)] = mask_id
    randomise = chosen & (action >= 0.8) & (action < 0.9)
    inputs[randomise] = np.random.randint(0, vocab_size, size=int(randomise.sum()))
    return inputs, labels
