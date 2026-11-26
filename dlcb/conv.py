"""Convolutional layers: ``Conv2d``, ``ConvTranspose2d``, ``MaxPool2d``, ``AvgPool2d``, ``Flatten``.

Everything here follows *Lecture V — Convolutional Networks*, and in particular
its central section, *Implementação por im2col e uma multiplicação matricial*:
a convolutional layer is **one** matrix product,

    Z = Wᵀ X̃,      X̃ = [ 1 ; X_col ] ∈ R^{(K+1) × N},   W ∈ R^{(K+1) × F},

where column ``p`` of ``X_col`` is the receptive field of output position
``p``, column ``f`` of ``W`` is filter ``f`` flattened, and row 0 of ``W``
holds the biases. That is *literally* the equation of ``nn.Linear`` (Lecture I,
the bias trick), with one change of meaning: a column of the input is no
longer a sample, it is the **neighbourhood of one position**. So ``Conv2d``
is ``im2col`` followed by a ``Linear(K, F)``, and needs no backward rule of
its own:

- ``Z = Wᵀ X̃`` is the engine's ``@``, whose backward gives ``∂L/∂W = X̃ Dᵀ``
  and ``∂L/∂X_col = W' D`` — the lecture's two boxed equations;
- ``im2col`` is fancy indexing (``Tensor.__getitem__``), whose backward is
  ``np.add.at``: every copy of a pixel sends its gradient back to the same
  pixel, *accumulating*. That is ``col2im``, the adjoint of ``im2col``
  ("where im2col copies, col2im sums").

``ConvTranspose2d`` runs the same two steps the other way round: a ``Linear``
turns each input pixel into a whole ``k × k`` window of output values, and
``col2im`` adds the overlapping windows into the image. It is the operation
``Conv2d``'s backward applies to the gradient, promoted to a layer of its own —
the way a network learns to *enlarge* a feature map (Dumoulin & Visin, 2016).

Layout
------
Images are **channels-last**, ``(batch, H, W, C)``: the lecture's ``H × W × C``
with a batch axis in front, and the same "features on the last axis" rule as
the Transformer's ``(batch, seq, d_model)``. Inside a layer the lecture's
column layout takes over: ``X_col`` has shape ``(K, batch · N)`` — all the
positions of all the images side by side, one column each — so the column-
oriented ``Linear`` applies to it unchanged.

Flattening orders (declared, as the lecture insists, because the weights only
multiply the right pixels if the window and the filter are read the same way):

- positions ``p = 0 … N-1`` run **row by row** over the output map;
- inside a window, ``K = C · k_h · k_w`` values are stacked **channel by
  channel**, each channel's window read in **row-major** order (Lecture V,
  *Exemplo multicanal*);
- in ``X_col``'s column index, the batch is the outer axis: column
  ``b · N + p`` is position ``p`` of image ``b``.

Cost
----
``X̃`` has ``(K+1) · N`` entries per image, about ``k_h k_w`` times the input
when ``S = 1``: im2col trades memory for one large, fast matrix product
(Chellapilla et al., 2006). The FLOP counter sees only the ``@``:
``2 · F · (K+1) · N`` per image; the gather and the reshapes cost 0.

References
----------
[1] Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. "Gradient-based learning applied
    to document recognition." Proc. IEEE 86(11), 1998.
[2] K. Chellapilla, S. Puri, P. Simard. "High performance convolutional neural
    networks for document processing." IWFHR, 2006. (im2col + GEMM)
[3] V. Dumoulin, F. Visin. "A guide to convolution arithmetic for deep
    learning." arXiv:1603.07285, 2016. (output sizes, stride, padding,
    transposed convolution)
[4] J. Long, E. Shelhamer, T. Darrell. "Fully convolutional networks for
    semantic segmentation." CVPR 2015, arXiv:1411.4038. (learned upsampling)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple, Union

import numpy as np

from dlcb.engine import Tensor, cat
from dlcb.nn import Linear, Module, Parameter

IntPair = Union[int, Tuple[int, int]]


def _pair(value: IntPair) -> Tuple[int, int]:
    """``3 -> (3, 3)``; ``(3, 5) -> (3, 5)``."""
    return (value, value) if isinstance(value, int) else (int(value[0]), int(value[1]))


def output_size(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """Lecture V, *A fórmula das dimensões*: ``⌊(size + 2P − k) / S⌋ + 1``.

    A window starts flush with the top-left corner of the padded input and
    counts only if it fits entirely; when ``S`` does not divide
    ``size + 2P − k`` the floor drops the last rows/columns that cannot fill a
    window. With ``P = 0`` and ``S = 1`` this is ``size − k + 1``.
    """
    out = (size + 2 * padding - kernel) // stride + 1
    if out < 1:
        raise ValueError(
            f"kernel {kernel} does not fit an input of {size} with padding {padding}"
        )
    return out


def transposed_output_size(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """Output size of a transposed convolution: ``(size − 1) · S − 2P + k``.

    The inverse of ``output_size`` read backwards: ``size`` input pixels, ``S``
    apart, each paint a ``k``-wide window, so the windows span
    ``(size − 1) · S + k`` pixels; ``P`` is then cut from each border. When
    ``S`` divides ``size' + 2P − k``, ``output_size`` of the result gives back
    ``size`` — the two layers map the same pair of shapes in opposite
    directions (Dumoulin & Visin, 2016, Section 4).
    """
    out = (size - 1) * stride - 2 * padding + kernel
    if out < 1:
        raise ValueError(
            f"padding {padding} removes the whole output of a transposed convolution "
            f"with kernel {kernel}, stride {stride} on an input of {size}"
        )
    return out


def pad2d(x: Tensor, padding: IntPair) -> Tensor:
    """Surround each image of ``x`` ``(batch, H, W, C)`` with rows/columns of zeros.

    ``padding = (P_h, P_w)`` adds ``P_h`` rows above *and* below and ``P_w``
    columns left *and* right (Lecture V, *Padding P=1: mais colunas, com zeros*).
    It is built with the engine's ``cat`` and constant zero tensors, so the
    gradient of the padded border is simply dropped and the interior's flows
    back unchanged — no new backward rule.
    """
    ph, pw = _pair(padding)
    if ph == 0 and pw == 0:
        return x
    b, h, w, c = x.shape
    dtype = x.data.dtype
    if ph:
        rows = Tensor(np.zeros((b, ph, w, c), dtype=dtype), requires_grad=False)
        x = cat([rows, x, rows], axis=1)
    if pw:
        cols = Tensor(np.zeros((b, h + 2 * ph, pw, c), dtype=dtype), requires_grad=False)
        x = cat([cols, x, cols], axis=2)
    return x


def window_indices(
    height: int, width: int, kernel: IntPair, stride: IntPair = 1
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Row and column of every pixel of every window: the index maps of im2col.

    Returns ``rows, cols`` of shape ``(N, k_h · k_w)`` and ``(H_out, W_out)``.
    Entry ``[p, t]`` is the pixel that sits at offset ``t`` (row-major inside
    the window) of the window for output position ``p`` (row by row over the
    output). On Lecture V's ``4 × 4`` example with ``2 × 2`` windows::

        rows[0] = [0, 0, 1, 1]   cols[0] = [0, 1, 0, 1]   # pixels 1, 2, 5, 6
        rows[4] = [1, 1, 2, 2]   cols[4] = [1, 2, 1, 2]   # pixels 6, 7, 10, 11

    Pixel 6 (row 1, column 1) appears in both: the windows overlap, and this
    repetition is exactly what im2col materialises — and what its backward
    must sum.
    """
    kh, kw = _pair(kernel)
    sh, sw = _pair(stride)
    hout = output_size(height, kh, sh)
    wout = output_size(width, kw, sw)
    top = np.repeat(np.arange(hout) * sh, wout)       # (N,) top row of each window
    left = np.tile(np.arange(wout) * sw, hout)        # (N,) left column of each window
    di = np.repeat(np.arange(kh), kw)                 # (k_h k_w,) row offsets, row-major
    dj = np.tile(np.arange(kw), kh)                   # (k_h k_w,) column offsets
    return top[:, None] + di[None, :], left[:, None] + dj[None, :], (hout, wout)


def extract_windows(
    x: Tensor, kernel: IntPair, stride: IntPair = 1, padding: IntPair = 0
) -> Tuple[Tensor, Tuple[int, int]]:
    """Every window of every image, as ``(batch, N, k_h · k_w, C)``.

    One fancy-indexing gather, ``x[:, rows, cols, :]``. Its backward is the
    engine's ``np.add.at``, which **accumulates** over repeated indices — so a
    pixel that sits in four windows receives the sum of four gradients. That
    accumulation *is* col2im (Lecture V, *Passo 5*). Pooling uses this directly;
    ``im2col`` reorders it into columns for the matrix product.
    """
    x = pad2d(x, padding)
    _, h, w, _ = x.shape
    rows, cols, out_hw = window_indices(h, w, kernel, stride)
    return x[:, rows, cols, :], out_hw


def im2col(
    x: Tensor, kernel: IntPair, stride: IntPair = 1, padding: IntPair = 0
) -> Tuple[Tensor, Tuple[int, int]]:
    """Image to columns: ``(batch, H, W, C) -> X_col`` of shape ``(K, batch · N)``.

    Column ``b · N + p`` is the receptive field of position ``p`` in image ``b``,
    flattened channel by channel, each channel in row-major order, so that
    ``K = C · k_h · k_w``. On Lecture V's *Exemplo numérico 4.1* (the integers
    1…16 in a ``4 × 4`` image, ``2 × 2`` windows) this returns exactly::

        1  2  3  5  6  7  9 10 11
        2  3  4  6  7  8 10 11 12
        5  6  7  9 10 11 13 14 15
        6  7  8 10 11 12 14 15 16

    Data are only moved and copied — no arithmetic, no parameters; the FLOP
    counter records 0 for it.
    """
    windows, out_hw = extract_windows(x, kernel, stride, padding)   # (B, N, kk, C)
    b, n, kk, c = windows.shape
    # (B, N, kk, C) -> (C, kk, B, N): channel-major K on the rows, then batch-major
    # positions on the columns; the reshape glues each pair of axes together.
    return windows.transpose(3, 2, 0, 1).reshape(c * kk, b * n), out_hw


@lru_cache(maxsize=64)
def _col2im_table(height: int, width: int, kernel: Tuple[int, int],
                  stride: Tuple[int, int]) -> np.ndarray:
    """For every pixel of a ``height × width`` image, which window entries land on it.

    Window entry ``s = p · k_h k_w + t`` is offset ``t`` of the window of
    position ``p`` (the order of ``window_indices``). Row ``q`` of the returned
    ``(height · width, M)`` table lists the entries whose pixel is ``q``
    (row-major), padded with the index ``N · k_h k_w`` — one past the last
    entry, which ``col2im`` points at a zero. ``M = ⌈k_h/S_h⌉ · ⌈k_w/S_w⌉`` is
    the most windows any pixel belongs to.
    """
    rows, cols, _ = window_indices(height, width, kernel, stride)
    pixel = (rows * width + cols).ravel()                   # pixel of entry s
    order = np.argsort(pixel, kind="stable")                # entries grouped by pixel
    counts = np.bincount(pixel, minlength=height * width)
    first = np.cumsum(counts) - counts                      # where each pixel's group starts
    rank = np.arange(pixel.size) - first[pixel[order]]      # 0, 1, … inside the group
    table = np.full((height * width, counts.max()), pixel.size)
    table[pixel[order], rank] = order
    return table


def col2im(
    x_col: Tensor,
    image_size: IntPair,
    kernel: IntPair,
    stride: IntPair = 1,
    padding: IntPair = 0,
) -> Tensor:
    """Columns to image, **summing** overlaps: the adjoint of ``im2col``.

    ``x_col`` has ``im2col``'s layout, ``(C · k_h k_w, batch · N)``: column
    ``b · N + p`` holds a ``k_h × k_w × C`` window for position ``p`` of image
    ``b``. Each window is put back where ``im2col`` would have read it, and
    where windows overlap their values are **added** (Lecture V, *Passo 5*:
    "where im2col copies, col2im sums"). Finally the ``P`` padding rows and
    columns are cut off. Returns ``(batch, H, W, C)`` with ``(H, W) =
    image_size``, the size *without* padding.

    Adjoint means: for every image ``x`` and columns ``y`` of matching shapes,
    ``⟨im2col(x), y⟩ = ⟨x, col2im(y)⟩``. That is why ``im2col``'s backward is a
    col2im (``np.add.at`` inside ``Tensor.__getitem__``) and, here, col2im's
    backward is an im2col — no new rule is needed.

    It is built from engine operations only. A pixel lies in at most
    ``M = ⌈k_h/S_h⌉ ⌈k_w/S_w⌉`` windows, so a precomputed table lists, for each
    pixel, the (up to ``M``) window entries that land on it; one gather reads
    them into ``(batch, H_p W_p, M, C)`` and a sum over ``M`` adds them. Missing
    entries point at an appended zero.
    """
    kh, kw = _pair(kernel)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    height, width = _pair(image_size)
    hp, wp = height + 2 * ph, width + 2 * pw               # the padded canvas
    _, _, (hout, wout) = window_indices(hp, wp, (kh, kw), (sh, sw))
    n, kk = hout * wout, kh * kw
    if x_col.ndim != 2 or x_col.shape[0] % kk or x_col.shape[1] % n:
        raise ValueError(
            f"col2im: columns {x_col.shape} do not fit {kh}×{kw} windows at "
            f"{n} positions of a {hp}×{wp} padded image"
        )
    c, b = x_col.shape[0] // kk, x_col.shape[1] // n
    # (C·kk, B·N) -> (B, N·kk, C): entry s = p·kk + t of each image, channels last.
    entries = x_col.reshape(c, kk, b, n).transpose(2, 3, 1, 0).reshape(b, n * kk, c)
    zero = Tensor(np.zeros((b, 1, c), dtype=x_col.data.dtype), requires_grad=False)
    entries = cat([entries, zero], axis=1)                  # index n·kk reads 0
    table = _col2im_table(hp, wp, (kh, kw), (sh, sw))       # (H_p W_p, M)
    image = entries[:, table, :].sum(axis=2).reshape(b, hp, wp, c)
    if ph or pw:
        image = image[:, ph:ph + height, pw:pw + width, :]
    return image


class Conv2d(Module):
    """2-D multi-channel convolution (strictly, cross-correlation) as im2col + ``Linear``.

    Each of the ``F = out_channels`` filters has shape ``k_h × k_w × C_in``:
    it covers **all** input channels and slides only along the two spatial axes
    (Lecture V, *Convolução 2D, 2D multicanal e 3D*). Like every deep-learning
    library, this is cross-correlation — the kernel is not flipped; since the
    kernel is learned, the flip would only relabel its entries (Lecture V,
    *Convolução matemática discreta e correlação cruzada*).

    The layer is three lines::

        X_col = im2col(x)              # (K, B·N)     data moved, nothing learned
        Z     = linear(X_col)          # (F, B·N)     Wᵀ [1; X_col], W: (K+1) × F
        y     = Z → (B, H_out, W_out, F)                reshape + transpose

    and ``self.weight`` is the lecture's ``W`` exactly: shape
    ``(K + 1, F)``, column ``f`` = ``[b_f ; filter f flattened]``, row 0 =
    biases. So ``(K + 1) · F`` parameters — independent of the image size,
    which is the whole economy of convolution over a dense layer (Lecture V,
    *Contando parâmetros nas duas*). The filter ``f`` as a
    ``(C_in, k_h, k_w)`` array is ``weight.data[1:, f].reshape(C_in, k_h, k_w)``.

    Initialisation is ``Linear``'s: Glorot/Xavier uniform over fan-in ``K`` and
    fan-out ``F``, biases at zero.

    Parameters
    ----------
    in_channels : int
        ``C_in``, channels of the input (1 for greyscale, 3 for RGB, ``F`` of the
        previous layer otherwise).
    out_channels : int
        ``F``, number of filters = channels of the output.
    kernel_size : int or (int, int)
        ``k`` or ``(k_h, k_w)``.
    stride : int or (int, int)
        ``S``; ``S = 2`` keeps every other column of ``X̃`` (Lecture V,
        *Stride S=2: menos colunas*).
    padding : int or (int, int)
        ``P`` rows/columns of zeros on each border; ``P = (k − 1) / 2`` with
        ``S = 1`` keeps the spatial size ("same" padding, odd ``k``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntPair,
        stride: IntPair = 1,
        padding: IntPair = 0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        kh, kw = self.kernel_size
        self.linear = Linear(in_channels * kh * kw, out_channels)

    @property
    def weight(self) -> Tensor:
        """The lecture's ``W``, ``(K + 1, F)``: row 0 biases, column ``f`` filter ``f``."""
        return self.linear.weight

    def forward(self, x: Tensor) -> Tensor:
        """``(batch, H, W, C_in) -> (batch, H_out, W_out, F)``."""
        if x.ndim != 4 or x.shape[3] != self.in_channels:
            raise ValueError(
                f"Conv2d expects (batch, H, W, {self.in_channels}), got {x.shape}"
            )
        b = x.shape[0]
        x_col, (hout, wout) = im2col(x, self.kernel_size, self.stride, self.padding)
        z = self.linear(x_col)                             # (F, B·N): row f = map of filter f
        # Undo the flattening: column b·N + p is image b, position p = i·W_out + j,
        # so Y[b, i, j, f] = Z[f, b·N + i·W_out + j] (Lecture V, *o reshape*).
        return z.reshape(self.out_channels, b, hout, wout).transpose(1, 2, 3, 0)


class ConvTranspose2d(Module):
    """Transposed convolution: each input pixel paints a ``k × k`` window of the output.

    Where ``Conv2d`` *reads* a window and writes one number, this layer *reads*
    one pixel and writes a window. Two lines, ``Conv2d``'s in reverse::

        Z = linear(X)               # (C_out·k_h·k_w, B·N): one window per input pixel
        y = col2im(Z) + b           # windows S apart, overlaps summed, P cut away

    With stride ``S > 1`` the windows land ``S`` pixels apart, so the output is
    about ``S`` times larger along each axis,
    ``H_out = (H − 1) · S − 2P + k`` (``transposed_output_size``): the learned
    way to **upsample** a feature map, used in decoders and segmentation
    networks (Long, Shelhamer & Darrell, 2015).

    Why "transposed": a convolution is a linear map, a (sparse) matrix
    ``C`` acting on the flattened image. With the same filters, this layer
    applies ``Cᵀ`` — exactly what ``Conv2d``'s backward does to the gradient.
    In weights: if ``conv = Conv2d(A, B, k, S, P)``, then
    ``ConvTranspose2d(B, A, k, S, P)`` with ``weight = conv.weight[1:].T``
    satisfies ``⟨conv(x), y⟩ = ⟨x, convT(y)⟩`` when both biases are zero. It is
    *not* the inverse of the convolution: it restores the **shape**, not the
    values. Equivalently (Dumoulin & Visin, 2016), it is an ordinary
    convolution over the input with ``S − 1`` zeros inserted between pixels.

    Parameters live in two places, and this is the one departure from the
    lecture's bias trick: ``self.weight`` is ``(C_in, C_out · k_h k_w)``, row
    ``i`` = what input channel ``i`` paints, column ``c · k_h k_w + t`` = offset
    ``t`` of output channel ``c``; ``self.bias`` is ``(C_out,)``. A bias row in
    ``weight`` would be added once per *window entry*, so an output pixel
    covered by four windows would get it four times; the bias must be added
    once per output pixel, after the sum. ``C_in · C_out · k_h k_w + C_out``
    parameters.

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size : int or (int, int)
    stride : int or (int, int)
        Distance between the windows painted by neighbouring input pixels.
    padding : int or (int, int)
        Rows/columns removed from each border of the result — the padding the
        matching ``Conv2d`` would add.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntPair,
        stride: IntPair = 1,
        padding: IntPair = 0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        kh, kw = self.kernel_size
        self.linear = Linear(in_channels, out_channels * kh * kw, bias=False)
        self.bias = Parameter(np.zeros(out_channels, dtype=self.linear.weight.data.dtype))

    @property
    def weight(self) -> Tensor:
        """``(C_in, C_out · k_h k_w)``: row ``i`` is the window input channel ``i`` paints."""
        return self.linear.weight

    def forward(self, x: Tensor) -> Tensor:
        """``(batch, H, W, C_in) -> (batch, H_out, W_out, C_out)``."""
        if x.ndim != 4 or x.shape[3] != self.in_channels:
            raise ValueError(
                f"ConvTranspose2d expects (batch, H, W, {self.in_channels}), got {x.shape}"
            )
        b, h, w, _ = x.shape
        (kh, kw), (sh, sw), (ph, pw) = self.kernel_size, self.stride, self.padding
        hout = transposed_output_size(h, kh, sh, ph)
        wout = transposed_output_size(w, kw, sw, pw)
        # One column per input pixel, batch-major then row-major: column b·N + p.
        x_col = x.reshape(b * h * w, self.in_channels).T          # (C_in, B·N)
        z = self.linear(x_col)                                    # (C_out·kk, B·N)
        return col2im(z, (hout, wout), self.kernel_size, self.stride, self.padding) + self.bias


class MaxPool2d(Module):
    """Max pooling: the largest value of each ``k × k`` window, channel by channel.

    No parameters, and so no row of ones (Lecture V, *Pooling e os limites da
    analogia matricial*). It reuses the window gather of im2col and takes the
    maximum over each window's ``k_h k_w`` values.

    Max pooling is *piecewise* linear: the selector of the winning pixel depends
    on the input. Backward sends the incoming gradient to the winner only — the
    engine's ``max`` rule — and ``np.add.at`` sums it when overlapping windows
    share a winner. **Ties:** the lecture gives the whole gradient to the first
    winner in row-major order; the engine's ``max`` splits it evenly among tied
    winners. Both are valid subgradients, and with real-valued activations ties
    have probability zero.

    Parameters
    ----------
    kernel_size : int or (int, int)
    stride : int or (int, int), optional
        Defaults to ``kernel_size``: non-overlapping windows, the usual choice.
    """

    def __init__(self, kernel_size: IntPair, stride: IntPair | None = None) -> None:
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size

    def forward(self, x: Tensor) -> Tensor:
        """``(batch, H, W, C) -> (batch, H_out, W_out, C)``."""
        windows, (hout, wout) = extract_windows(x, self.kernel_size, self.stride)
        return windows.max(axis=2).reshape(x.shape[0], hout, wout, x.shape[3])


class AvgPool2d(Module):
    """Average pooling: the mean of each ``k × k`` window, channel by channel.

    Unlike max pooling it is **linear**: a fixed row ``a = [1/k², …, 1/k²]``
    times the window columns — a convolution with a constant filter, no bias,
    stride ``k`` (Lecture V, *Average pooling é linear; max pooling não*).
    Backward gives every pixel of a window ``1/k²`` of the window's gradient.
    """

    def __init__(self, kernel_size: IntPair, stride: IntPair | None = None) -> None:
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size

    def forward(self, x: Tensor) -> Tensor:
        """``(batch, H, W, C) -> (batch, H_out, W_out, C)``."""
        windows, (hout, wout) = extract_windows(x, self.kernel_size, self.stride)
        return windows.mean(axis=2).reshape(x.shape[0], hout, wout, x.shape[3])


class Flatten(Module):
    """``(batch, H, W, C) -> (H · W · C, batch)``: one **column** per image.

    The bridge from feature maps to the column-oriented ``Linear`` head (Lecture
    V, *Uma coluna por imagem, uma coluna por posição*). Each image is read
    row-major over ``(H, W, C)`` — so entry ``(i · W + j) · C + c`` is channel
    ``c`` at pixel ``(i, j)`` — and becomes one column, exactly as a sample is a
    column in Lecture I. No arithmetic; the gradient is reshaped back.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1).T
