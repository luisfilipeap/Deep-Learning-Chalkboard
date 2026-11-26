"""Tests for ``dlcb.conv``: how do we know the convolution is right?

Passing one test proves little, so the layer is checked five independent ways.
Each section answers a different question a human would ask:

1. **An independent reference.** Does it compute the textbook formula
   ``Y[b,i,j,f] = b_f + Σ_c Σ_u Σ_v X[b, iS+u−P, jS+v−P, c] · w_f[c,u,v]``,
   written with explicit loops and no engine, for every stride and padding?
2. **The lecture's worked examples.** Does it reproduce, number for number, the
   hand computations of Lecture V (im2col, Z = WᵀX̃, multichannel, backprop
   and col2im, pooling)?
3. **Properties the maths guarantees.** Translation equivariance, linearity,
   the output-size formula, "same" padding, 1×1 conv = per-pixel dense layer,
   parameter count independent of image size, the FLOP count.
4. **Finite differences.** Does ``backward()`` agree with a numerical gradient
   for the input and the weights, with stride and padding on?
5. **Behaviour.** Can a small CNN learn a task that needs translation-invariant
   detection, from data, with ``backward`` + Adam?
"""

import numpy as np
import pytest

from dlcb import engine as cpu
from dlcb.conv import (
    AvgPool2d,
    Conv2d,
    Flatten,
    MaxPool2d,
    im2col,
    output_size,
    window_indices,
)
from dlcb.loss import cross_entropy
from dlcb.nn import Linear
from dlcb.optim import Adam
from test.test_engine import numeric_gradient


# ====================================================================== #
# Independent references (plain NumPy, explicit loops, no engine)
# ====================================================================== #
def reference_conv(x, filters, bias, stride=1, padding=0):
    """Cross-correlation written as literally as possible.

    ``x``: (B, H, W, C); ``filters``: (F, C, kh, kw); ``bias``: (F,).
    Pixels outside the image read as 0 (that is what padding means).
    """
    b_, h, w, c_ = x.shape
    f_, _, kh, kw = filters.shape
    hout = (h + 2 * padding - kh) // stride + 1
    wout = (w + 2 * padding - kw) // stride + 1
    y = np.zeros((b_, hout, wout, f_))
    for b in range(b_):
        for i in range(hout):
            for j in range(wout):
                for f in range(f_):
                    total = bias[f]
                    for c in range(c_):
                        for u in range(kh):
                            for v in range(kw):
                                r, s = i * stride + u - padding, j * stride + v - padding
                                if 0 <= r < h and 0 <= s < w:
                                    total += x[b, r, s, c] * filters[f, c, u, v]
                    y[b, i, j, f] = total
    return y


def reference_pool(x, k, stride, reduce):
    """Max or mean over each k×k window, per channel, with loops."""
    b_, h, w, c_ = x.shape
    hout, wout = (h - k) // stride + 1, (w - k) // stride + 1
    y = np.zeros((b_, hout, wout, c_))
    for b in range(b_):
        for i in range(hout):
            for j in range(wout):
                for c in range(c_):
                    y[b, i, j, c] = reduce(
                        x[b, i * stride:i * stride + k, j * stride:j * stride + k, c]
                    )
    return y


def set_filters(conv, filters, bias):
    """Write (F, C, kh, kw) filters and (F,) biases into the layer's W."""
    f_ = filters.shape[0]
    conv.weight.data[0] = bias
    conv.weight.data[1:] = filters.reshape(f_, -1).T


def random_conv(c_in, f, k, stride=1, padding=0):
    """A Conv2d with random filters AND random biases.

    Biases start at zero, so a test with the default init cannot see a bug in
    the bias row. Randomising every parameter makes every one of them visible.
    """
    conv = Conv2d(c_in, f, k, stride=stride, padding=padding)
    filters = np.random.randn(f, c_in, k, k)
    bias = np.random.randn(f)
    set_filters(conv, filters, bias)
    return conv, filters, bias


# ====================================================================== #
# 1. Against the independent reference
# ====================================================================== #
@pytest.mark.parametrize(
    "h, w, c_in, f, k, stride, padding",
    [
        (5, 5, 1, 1, 3, 1, 0),    # the plain case
        (6, 7, 3, 4, 3, 1, 1),    # RGB-like, non-square, "same" padding
        (7, 7, 2, 3, 3, 2, 0),    # stride 2
        (8, 6, 2, 2, 2, 2, 1),    # stride and padding together, even kernel
        (5, 5, 3, 2, 5, 1, 2),    # kernel as large as the image
        (9, 9, 2, 3, 3, 3, 1),    # stride that does not divide H + 2P − k
    ],
)
def test_conv_matches_loop_reference(h, w, c_in, f, k, stride, padding):
    cpu.set_seed(0)
    conv, filters, bias = random_conv(c_in, f, k, stride, padding)
    x = np.random.randn(2, h, w, c_in)
    y = conv(cpu.Tensor(x)).data
    expected = reference_conv(x, filters, bias, stride, padding)
    assert y.shape == expected.shape
    assert np.allclose(y, expected, atol=1e-12)


@pytest.mark.parametrize("k, stride", [(2, 2), (3, 3), (3, 2), (2, 1)])
def test_pools_match_loop_reference(k, stride):
    cpu.set_seed(1)
    x = np.random.randn(2, 7, 6, 3)
    assert np.allclose(MaxPool2d(k, stride)(cpu.Tensor(x)).data,
                       reference_pool(x, k, stride, np.max))
    assert np.allclose(AvgPool2d(k, stride)(cpu.Tensor(x)).data,
                       reference_pool(x, k, stride, np.mean))


def test_images_in_a_batch_do_not_interact():
    """Each image's output depends only on that image (batch is not a spatial axis)."""
    cpu.set_seed(2)
    conv, _, _ = random_conv(2, 3, 3, padding=1)
    x = np.random.randn(3, 5, 5, 2)
    batched = conv(cpu.Tensor(x)).data
    for b in range(3):
        alone = conv(cpu.Tensor(x[b:b + 1])).data[0]
        assert np.allclose(batched[b], alone)


# ====================================================================== #
# 2. Lecture V's worked examples, number for number
# ====================================================================== #
X16 = np.arange(1, 17, dtype=float).reshape(1, 4, 4, 1)      # the integers 1…16


def test_im2col_example_4_1():
    x_col, out_hw = im2col(cpu.Tensor(X16), 2)
    assert out_hw == (3, 3)
    assert np.array_equal(x_col.data, [
        [1, 2, 3, 5, 6, 7, 9, 10, 11],
        [2, 3, 4, 6, 7, 8, 10, 11, 12],
        [5, 6, 7, 9, 10, 11, 13, 14, 15],
        [6, 7, 8, 10, 11, 12, 14, 15, 16],
    ])


def test_window_indices_overlap_on_pixel_6():
    rows, cols, _ = window_indices(4, 4, 2)
    assert rows[0].tolist() == [0, 0, 1, 1] and cols[0].tolist() == [0, 1, 0, 1]
    assert rows[4].tolist() == [1, 1, 2, 2] and cols[4].tolist() == [1, 2, 1, 2]
    # pixel 6 = (1, 1) is central: it sits in 4 windows; pixel 1 = (0, 0) in one.
    hits = ((rows == 1) & (cols == 1)).sum(), ((rows == 0) & (cols == 0)).sum()
    assert hits == (4, 1)


def test_two_filters_example_4_2():
    """W = [b; filters] as a (K+1)×F matrix, exactly as printed in the lecture."""
    conv = Conv2d(1, 2, 2)
    conv.weight.data[:] = [[2, -1], [1, 0], [0, 1], [0, 1], [-1, 0]]
    y = conv(cpu.Tensor(X16)).data[0]
    assert np.array_equal(y[:, :, 0], np.full((3, 3), -3.0))
    assert np.array_equal(y[:, :, 1], [[6, 8, 10], [14, 16, 18], [22, 24, 26]])


def test_multichannel_example_13_1():
    x0 = [[1, 0, 2], [3, 1, 0], [0, 2, 1]]
    x1 = [[0, 1, 1], [2, 0, 1], [1, 1, 0]]
    x = np.stack([x0, x1], axis=-1)[None].astype(float)        # (1, 3, 3, 2)
    conv = Conv2d(2, 1, 2)
    conv.weight.data[:, 0] = [1, 1, 0, 0, -1, 0, 1, 1, 0]       # bias, channel 0, channel 1
    assert np.array_equal(conv(cpu.Tensor(x)).data[0, :, :, 0], [[4, 2], [3, 3]])


def test_backprop_example_5_1():
    """Forward, ReLU, ∂L/∂W = X̃Dᵀ (bias first) and col2im, all checkable by hand."""
    x = cpu.Tensor(np.array([[1, 2, 0], [0, 1, 3], [2, 1, 1]], dtype=float)[None, :, :, None])
    conv = Conv2d(1, 1, 2)
    conv.weight.data[:, 0] = [-1, 1, 1, -1, 1]
    a = conv(x).relu()
    assert a.data.ravel().tolist() == [3, 3, 0, 3]
    upstream = cpu.Tensor(np.array([1.0, -1.0, 2.0, 1.0]).reshape(1, 2, 2, 1), requires_grad=False)
    (a * upstream).sum().backward()                 # makes ∂L/∂A = upstream
    assert conv.weight.grad[:, 0].tolist() == [1, 0, 5, 0, -1]
    assert np.array_equal(x.grad[0, :, :, 0], [[1, 0, -1], [-1, 3, 0], [0, -1, 1]])


def test_sum_of_preactivations_example_5_2():
    """L = Σ Z: the bias gradient counts positions; each weight sums its pixels."""
    x = cpu.Tensor(np.array([[1, 2, 0], [0, 1, 3], [2, 1, 1]], dtype=float)[None, :, :, None])
    conv = Conv2d(1, 1, 2)
    conv(x).sum().backward()
    assert conv.weight.grad[:, 0].tolist() == [4, 4, 6, 4, 6]


def test_pooling_example_9_1_forward_and_backward():
    u = np.array([[2, 0, 3, 1, 0, 1], [0, 4, 6, 2, 2, 1], [1, 2, 1, 7, 1, 3],
                  [1, 2, 3, 3, 3, 4], [0, 5, 7, 6, 7, 5], [2, 0, 2, 3, 3, 4]], dtype=float)
    g = np.array([[1, -2], [0.5, 3]])
    x = cpu.Tensor(u[None, :, :, None])
    y = MaxPool2d(3)(x)
    assert np.array_equal(y.data[0, :, :, 0], [[6, 7], [7, 7]])
    (y * cpu.Tensor(g[None, :, :, None], requires_grad=False)).sum().backward()
    expected = np.zeros((6, 6))
    expected[1, 2], expected[2, 3], expected[4, 2], expected[4, 4] = 1, -2, 0.5, 3
    assert np.array_equal(x.grad[0, :, :, 0], expected)       # only the winners

    x = cpu.Tensor(u[None, :, :, None])
    y = AvgPool2d(3)(x)
    assert np.allclose(y.data[0, :, :, 0], np.array([[19, 18], [22, 38]]) / 9)
    (y * cpu.Tensor(g[None, :, :, None], requires_grad=False)).sum().backward()
    assert np.allclose(x.grad[0, :, :, 0], np.kron(g, np.ones((3, 3))) / 9)


def test_dimension_exercise_rgb_32():
    """Lecture V exercise: RGB 32×32, 16 filters 5×5 → W 76×16, output 28×28×16."""
    conv = Conv2d(3, 16, 5)
    assert conv.weight.shape == (76, 16)
    x_col, _ = im2col(cpu.Tensor(np.zeros((1, 32, 32, 3))), 5)
    assert x_col.shape == (75, 784)                  # X_col; X̃ adds the row of ones
    assert conv(cpu.Tensor(np.zeros((1, 32, 32, 3)))).shape == (1, 28, 28, 16)


# ====================================================================== #
# 3. Properties the maths guarantees
# ====================================================================== #
def test_translation_equivariance():
    """Shift the image, and the feature map shifts by the same amount.

    This is the property that makes convolution the right tool for images
    (Lecture V, *Localidade e compartilhamento*). The object is placed away
    from the borders, so no window ever sees the edge of the image.
    """
    cpu.set_seed(3)
    conv, _, _ = random_conv(2, 3, 3)
    blob = np.random.randn(4, 4, 2)
    a = np.zeros((1, 12, 12, 2)); a[0, 2:6, 3:7] = blob
    b = np.zeros((1, 12, 12, 2)); b[0, 5:9, 4:8] = blob          # moved 3 down, 1 right
    ya, yb = conv(cpu.Tensor(a)).data, conv(cpu.Tensor(b)).data
    assert np.allclose(yb[0, 3:, 1:], ya[0, :-3, :-1])


def test_conv_is_linear_in_the_input_when_bias_is_zero():
    cpu.set_seed(4)
    conv, filters, _ = random_conv(2, 2, 3, padding=1)
    set_filters(conv, filters, np.zeros(2))
    x1, x2 = np.random.randn(1, 5, 5, 2), np.random.randn(1, 5, 5, 2)
    f = lambda x: conv(cpu.Tensor(x)).data
    assert np.allclose(f(2.0 * x1 - 3.0 * x2), 2.0 * f(x1) - 3.0 * f(x2))


def test_avg_pool_is_linear_max_pool_is_not():
    x1 = np.array([[1.0, 0.0], [0.0, 0.0]])[None, :, :, None]
    x2 = np.array([[0.0, 1.0], [0.0, 0.0]])[None, :, :, None]
    for pool, linear in ((AvgPool2d(2), True), (MaxPool2d(2), False)):
        f = lambda x: pool(cpu.Tensor(x)).data
        assert np.allclose(f(x1 + x2), f(x1) + f(x2)) == linear


@pytest.mark.parametrize("size, k, s, p", [(28, 5, 1, 0), (28, 3, 1, 1), (7, 3, 2, 0),
                                          (8, 3, 2, 1), (224, 7, 2, 3), (9, 2, 3, 0)])
def test_output_size_formula(size, k, s, p):
    expected = (size + 2 * p - k) // s + 1
    assert output_size(size, k, s, p) == expected
    y = Conv2d(1, 1, k, stride=s, padding=p)(cpu.Tensor(np.zeros((1, size, size, 1))))
    assert y.shape == (1, expected, expected, 1)


def test_same_padding_keeps_the_spatial_size():
    for k in (1, 3, 5, 7):
        y = Conv2d(2, 4, k, padding=(k - 1) // 2)(cpu.Tensor(np.zeros((1, 9, 6, 2))))
        assert y.shape == (1, 9, 6, 4)


def test_one_by_one_conv_is_a_dense_layer_per_pixel():
    """A 1×1 filter mixes channels and looks at no neighbour."""
    cpu.set_seed(5)
    conv, _, _ = random_conv(4, 3, 1)
    x = np.random.randn(2, 3, 5, 4)
    dense = Linear(4, 3)
    dense.weight.data[:] = conv.weight.data
    per_pixel = dense(cpu.Tensor(x.reshape(-1, 4).T)).data.T.reshape(2, 3, 5, 3)
    assert np.allclose(conv(cpu.Tensor(x)).data, per_pixel)


def test_parameter_count_does_not_depend_on_image_size():
    conv = Conv2d(3, 8, 3, padding=1)
    n_params = sum(p.data.size for p in conv.parameters())
    assert n_params == (3 * 3 * 3 + 1) * 8
    for side in (8, 32):
        conv(cpu.Tensor(np.zeros((1, side, side, 3))))
        assert sum(p.data.size for p in conv.parameters()) == n_params


def test_flop_count_is_one_matrix_product():
    """im2col, padding, reshapes cost 0; the layer costs 2·F·(K+1)·N per image."""
    conv = Conv2d(3, 4, 3, stride=2, padding=1)
    x = cpu.Tensor(np.zeros((2, 9, 9, 3)))
    cpu.reset_flops()
    conv(x)
    k_aug, n = 3 * 3 * 3 + 1, 5 * 5
    assert cpu.flop_count() == 2 * 4 * k_aug * n * 2


def test_flatten_gives_one_column_per_image():
    x = np.random.randn(3, 2, 4, 5)
    cols = Flatten()(cpu.Tensor(x)).data
    assert cols.shape == (40, 3)
    assert np.array_equal(cols[:, 1], x[1].ravel())             # row-major over (H, W, C)
    assert cols[(1 * 4 + 2) * 5 + 3, 0] == x[0, 1, 2, 3]


def test_wrong_input_shape_raises():
    conv = Conv2d(3, 2, 3)
    with pytest.raises(ValueError):
        conv(cpu.Tensor(np.zeros((1, 5, 5, 1))))                # 1 channel, expected 3
    with pytest.raises(ValueError):
        conv(cpu.Tensor(np.zeros((5, 5, 3))))                   # no batch axis
    with pytest.raises(ValueError):
        Conv2d(1, 1, 7)(cpu.Tensor(np.zeros((1, 5, 5, 1))))      # kernel larger than image


# ====================================================================== #
# 4. Finite differences
# ====================================================================== #
@pytest.mark.parametrize("stride, padding", [(1, 0), (2, 1), (1, 2)])
def test_conv_gradcheck_input_weight(stride, padding):
    cpu.set_seed(6)
    conv, _, _ = random_conv(2, 3, 3, stride, padding)
    x = cpu.Tensor(np.random.randn(2, 5, 6, 2))
    target = np.random.randn(*conv(x).shape)

    def loss():
        return ((conv(x).tanh() - cpu.Tensor(target, requires_grad=False)) ** 2).mean()

    x.zero_grad(); conv.zero_grad()
    loss().backward()
    for name, t in (("x", x), ("weight", conv.weight)):
        numeric = numeric_gradient(lambda: float(loss().data), t.data)
        assert np.allclose(t.grad, numeric, atol=1e-6), f"gradient mismatch for {name}"


@pytest.mark.parametrize("pool", [MaxPool2d(2), MaxPool2d(3, 2), AvgPool2d(2), AvgPool2d(3, 1)])
def test_pool_gradcheck(pool):
    """Overlapping windows (stride < k) exercise the accumulation in col2im."""
    cpu.set_seed(7)
    x = cpu.Tensor(np.random.randn(2, 7, 7, 2))                 # distinct values: no ties
    w = np.random.randn(*pool(x).shape)
    loss = lambda: (pool(x) * cpu.Tensor(w, requires_grad=False)).sum()
    x.zero_grad()
    loss().backward()
    assert np.allclose(x.grad, numeric_gradient(lambda: float(loss().data), x.data), atol=1e-6)


def test_cnn_gradcheck_every_parameter():
    """Conv → ReLU → MaxPool → Conv → Flatten → Linear → cross-entropy."""
    cpu.set_seed(8)
    c1, _, _ = random_conv(1, 2, 3, padding=1)
    c2, _, _ = random_conv(2, 2, 2, stride=2)
    head = Linear(2 * 2 * 2, 3)
    head.weight.data[:] = np.random.randn(*head.weight.shape)
    x = cpu.Tensor(np.random.randn(2, 8, 8, 1))
    y = np.array([0, 2])

    def loss():
        h = MaxPool2d(2)(c1(x).relu())                         # (2, 4, 4, 2)
        logits = head(Flatten()(c2(h)))                         # (3, 2)
        return cross_entropy(logits.T, y)

    params = [x, c1.weight, c2.weight, head.weight]
    for t in params:
        t.zero_grad()
    loss().backward()
    for t in params:
        assert np.allclose(t.grad, numeric_gradient(lambda: float(loss().data), t.data),
                           atol=1e-6)


# ====================================================================== #
# 5. Behaviour: learn a translation-invariant detector
# ====================================================================== #
def make_cross_or_square(n, size=10, rng=None):
    """Images with a 3×3 cross (label 1) or a 3×3 hollow square (label 0), anywhere.

    The two shapes have the same 4-fold symmetry and almost the same number of
    lit pixels, and they appear at a random position with noise, so no single
    pixel and no pixel count decides the label: the network has to detect a
    *local pattern* wherever it is — what a shared filter plus pooling does.
    """
    rng = rng or np.random
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float)
    square = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
    x = 0.1 * rng.randn(n, size, size, 1)
    y = rng.randint(0, 2, n)
    for b in range(n):
        i, j = rng.randint(0, size - 2, 2)
        x[b, i:i + 3, j:j + 3, 0] += cross if y[b] else square
    return x, y


def test_cnn_learns_a_shape_anywhere_in_the_image():
    cpu.set_seed(9)
    rng = np.random.RandomState(9)
    x_train, y_train = make_cross_or_square(256, rng=rng)
    x_test, y_test = make_cross_or_square(200, rng=rng)
    conv = Conv2d(1, 4, 3)
    pool = MaxPool2d(8)                                         # global max: "is it anywhere?"
    head = Linear(4, 2)
    model_params = list(conv.parameters()) + list(head.parameters())
    opt = Adam(model_params, lr=0.05)

    def logits(x):
        return head(Flatten()(pool(conv(cpu.Tensor(x)).relu()))).T   # (batch, 2)

    for _ in range(60):
        opt.zero_grad()
        loss = cross_entropy(logits(x_train), y_train)
        loss.backward()
        opt.step()
    accuracy = (logits(x_test).data.argmax(axis=1) == y_test).mean()
    assert accuracy > 0.95, f"held-out accuracy {accuracy:.3f}"


# ====================================================================== #
# 6. Transposed convolution and col2im
# ====================================================================== #
from dlcb.conv import ConvTranspose2d, col2im, transposed_output_size  # noqa: E402


def reference_conv_transpose(x, filters, bias, stride=1, padding=0):
    """Each input pixel paints its window, written with loops.

    ``x``: (B, H, W, C_in); ``filters``: (C_in, C_out, kh, kw); ``bias``: (C_out,).
    Pixel (i, j) adds ``x[i, j, f] · filters[f, c, u, v]`` to output pixel
    (iS + u − P, jS + v − P); whatever lands outside the output is dropped.
    """
    b_, h, w, cin = x.shape
    _, cout, kh, kw = filters.shape
    hout, wout = (h - 1) * stride - 2 * padding + kh, (w - 1) * stride - 2 * padding + kw
    y = np.zeros((b_, hout, wout, cout)) + bias
    for b in range(b_):
        for i in range(h):
            for j in range(w):
                for f in range(cin):
                    for c in range(cout):
                        for u in range(kh):
                            for v in range(kw):
                                r, s = i * stride + u - padding, j * stride + v - padding
                                if 0 <= r < hout and 0 <= s < wout:
                                    y[b, r, s, c] += x[b, i, j, f] * filters[f, c, u, v]
    return y


def random_conv_transpose(cin, cout, k, stride=1, padding=0):
    """A ConvTranspose2d with random weights and random biases."""
    layer = ConvTranspose2d(cin, cout, k, stride=stride, padding=padding)
    filters = np.random.randn(cin, cout, k, k)
    bias = np.random.randn(cout)
    layer.weight.data[:] = filters.reshape(cin, -1)        # column c·kk + t
    layer.bias.data[:] = bias
    return layer, filters, bias


@pytest.mark.parametrize(
    "h, w, cin, cout, k, stride, padding",
    [
        (3, 3, 1, 1, 3, 1, 0),    # plain, windows overlap heavily
        (4, 5, 2, 3, 3, 2, 1),    # stride 2, padding, non-square
        (3, 3, 2, 2, 2, 2, 0),    # k = S: windows tile without overlap
        (2, 4, 3, 2, 4, 2, 1),    # k = 2S: every pixel in four windows (the usual ×2 upsampler)
        (3, 3, 1, 2, 2, 3, 0),    # S > k: gaps between windows stay at the bias
    ],
)
def test_conv_transpose_matches_loop_reference(h, w, cin, cout, k, stride, padding):
    cpu.set_seed(10)
    layer, filters, bias = random_conv_transpose(cin, cout, k, stride, padding)
    x = np.random.randn(2, h, w, cin)
    assert np.allclose(layer(cpu.Tensor(x)).data,
                       reference_conv_transpose(x, filters, bias, stride, padding))


def test_conv_transpose_by_hand_stride_1():
    """[[1,2],[3,4]] with a 2×2 window of ones: each pixel adds itself to a 2×2 block."""
    layer = ConvTranspose2d(1, 1, 2)
    layer.weight.data[:] = 1.0
    x = cpu.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]])[None, :, :, None])
    assert np.array_equal(layer(x).data[0, :, :, 0], [[1, 3, 2], [4, 10, 6], [3, 7, 4]])


def test_conv_transpose_by_hand_stride_2_is_block_upsampling():
    """k = S = 2, window of ones: no overlap, each pixel becomes a 2×2 block of itself."""
    layer = ConvTranspose2d(1, 1, 2, stride=2)
    layer.weight.data[:] = 1.0
    layer.bias.data[:] = 0.5
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = layer(cpu.Tensor(x[None, :, :, None])).data[0, :, :, 0]
    assert np.array_equal(y, np.kron(x, np.ones((2, 2))) + 0.5)


def test_bias_is_added_once_per_output_pixel():
    """Zero input: every pixel equals the bias, however many windows cover it."""
    layer = ConvTranspose2d(2, 3, 4, stride=2, padding=1)
    layer.bias.data[:] = [1.0, -2.0, 3.0]
    y = layer(cpu.Tensor(np.zeros((1, 3, 3, 2)))).data
    assert np.array_equal(y, np.broadcast_to([1.0, -2.0, 3.0], y.shape))


@pytest.mark.parametrize("h, k, s, p", [(5, 3, 1, 0), (4, 4, 2, 1), (3, 3, 2, 1), (6, 5, 3, 2)])
def test_transposed_output_size_inverts_output_size(h, k, s, p):
    """When S divides H' + 2P − k, Conv2d maps H' -> H and ConvTranspose2d maps H -> H'."""
    h_up = transposed_output_size(h, k, s, p)
    assert ConvTranspose2d(1, 1, k, s, p)(cpu.Tensor(np.zeros((1, h, h, 1)))).shape[1] == h_up
    if (h_up + 2 * p - k) % s == 0:
        assert output_size(h_up, k, s, p) == h


@pytest.mark.parametrize("k, stride, padding", [(3, 1, 0), (3, 2, 1), (4, 2, 1), (2, 2, 0)])
def test_conv_transpose_is_the_adjoint_of_conv(k, stride, padding):
    """⟨conv(x), y⟩ = ⟨x, convT(y)⟩ with the same filters and zero biases.

    This is the definition of "transposed": if a convolution is the matrix C
    acting on flattened images, the layer applies Cᵀ.
    """
    cpu.set_seed(11)
    a, b = 2, 3
    h = transposed_output_size(4, k, stride, padding)          # an H the stride divides
    conv, _, _ = random_conv(a, b, k, stride, padding)
    conv.weight.data[0] = 0.0
    convt = ConvTranspose2d(b, a, k, stride, padding)
    convt.weight.data[:] = conv.weight.data[1:].T
    x = np.random.randn(2, h, h, a)
    y_shape = conv(cpu.Tensor(x)).shape
    assert transposed_output_size(y_shape[1], k, stride, padding) == h   # shapes match exactly
    y = np.random.randn(*y_shape)
    lhs = (conv(cpu.Tensor(x)).data * y).sum()
    rhs = (x * convt(cpu.Tensor(y)).data).sum()
    assert np.isclose(lhs, rhs)


def test_conv_transpose_equals_conv_backward():
    """convT(y) is the gradient Conv2d's backward sends to its input when ∂L/∂out = y."""
    cpu.set_seed(12)
    conv, _, _ = random_conv(2, 3, 4, stride=2, padding=1)
    convt = ConvTranspose2d(3, 2, 4, stride=2, padding=1)
    convt.weight.data[:] = conv.weight.data[1:].T
    x = cpu.Tensor(np.random.randn(1, 8, 8, 2))
    out = conv(x)
    y = np.random.randn(*out.shape)
    (out * cpu.Tensor(y, requires_grad=False)).sum().backward()
    assert np.allclose(x.grad, convt(cpu.Tensor(y)).data)


def test_conv_transpose_is_a_conv_over_a_zero_stuffed_input():
    """Dumoulin & Visin: insert S − 1 zeros between pixels, pad by k − 1 − P,
    and convolve with the filters flipped and their channel roles swapped."""
    cpu.set_seed(13)
    cin, cout, k, s, p = 2, 3, 4, 2, 1
    layer, filters, bias = random_conv_transpose(cin, cout, k, s, p)
    x = np.random.randn(1, 3, 4, cin)
    stuffed = np.zeros((1, (3 - 1) * s + 1, (4 - 1) * s + 1, cin))
    stuffed[:, ::s, ::s, :] = x
    flipped = filters[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)      # (C_out, C_in, k, k)
    expected = reference_conv(stuffed, flipped, bias, stride=1, padding=k - 1 - p)
    assert np.allclose(layer(cpu.Tensor(x)).data, expected)


def test_col2im_is_the_adjoint_of_im2col():
    """⟨im2col(x), y⟩ = ⟨x, col2im(y)⟩, with overlapping windows and padding."""
    cpu.set_seed(14)
    x = np.random.randn(2, 6, 5, 3)
    cols, _ = im2col(cpu.Tensor(x), 3, 2, 1)
    y = np.random.randn(*cols.shape)
    back = col2im(cpu.Tensor(y), (6, 5), 3, 2, 1).data
    assert back.shape == x.shape
    assert np.isclose((cols.data * y).sum(), (x * back).sum())


def test_col2im_counts_window_membership():
    """col2im of all-ones columns = how many windows cover each pixel (4×4, k = 2, S = 1)."""
    cols, _ = im2col(cpu.Tensor(np.zeros((1, 4, 4, 1))), 2)
    counts = col2im(cpu.Tensor(np.ones(cols.shape)), (4, 4), 2).data[0, :, :, 0]
    assert np.array_equal(counts, [[1, 2, 2, 1], [2, 4, 4, 2], [2, 4, 4, 2], [1, 2, 2, 1]])


def test_conv_transpose_wrong_input_raises():
    with pytest.raises(ValueError):
        ConvTranspose2d(2, 1, 3)(cpu.Tensor(np.zeros((1, 4, 4, 3))))
    with pytest.raises(ValueError):
        ConvTranspose2d(1, 1, 2, padding=2)(cpu.Tensor(np.zeros((1, 1, 1, 1))))


@pytest.mark.parametrize("k, stride, padding", [(3, 1, 0), (4, 2, 1), (3, 2, 1)])
def test_conv_transpose_gradcheck(k, stride, padding):
    cpu.set_seed(15)
    layer, _, _ = random_conv_transpose(2, 3, k, stride, padding)
    x = cpu.Tensor(np.random.randn(2, 3, 4, 2))
    target = np.random.randn(*layer(x).shape)

    def loss():
        return ((layer(x).tanh() - cpu.Tensor(target, requires_grad=False)) ** 2).mean()

    x.zero_grad(); layer.zero_grad()
    loss().backward()
    for name, t in (("x", x), ("weight", layer.weight), ("bias", layer.bias)):
        numeric = numeric_gradient(lambda: float(loss().data), t.data)
        assert np.allclose(t.grad, numeric, atol=1e-6), f"gradient mismatch for {name}"


def test_conv_transpose_learns_to_upsample():
    """A ×2 transposed convolution learns bilinear-like upsampling from examples.

    Targets are smooth images; inputs are their 2×2 averages. A ConvTranspose2d
    (k = 4, S = 2, P = 1) must learn to put the detail back; its error must end
    below that of repeating each pixel (nearest-neighbour upsampling), and well
    below it away from the border — at the border an output pixel is covered by
    fewer windows, since the neighbour that would complete it lies outside.
    """
    cpu.set_seed(16)
    rng = np.random.RandomState(0)

    def smooth(n):
        yy, xx = np.mgrid[0:8, 0:8] / 8.0
        a, b, c, d = rng.randn(4, n, 1, 1)
        return (a * np.sin(2 * xx + b) + c * np.cos(2 * yy + d))[..., None]

    def downsample(img):
        return img.reshape(-1, 4, 2, 4, 2, 1).mean(axis=(2, 4))

    layer = ConvTranspose2d(1, 1, 4, stride=2, padding=1)
    opt = Adam(layer.parameters(), lr=0.03)
    for _ in range(300):
        target = smooth(32)
        opt.zero_grad()
        loss = ((layer(cpu.Tensor(downsample(target))) - cpu.Tensor(target, requires_grad=False)) ** 2).mean()
        loss.backward()
        opt.step()
    test = smooth(64)
    small = downsample(test)
    err_learned = layer(cpu.Tensor(small)).data - test
    err_nearest = np.repeat(np.repeat(small, 2, axis=1), 2, axis=2) - test
    assert (err_learned ** 2).mean() < (err_nearest ** 2).mean()
    inner = (slice(None), slice(2, -2), slice(2, -2))
    assert (err_learned[inner] ** 2).mean() < 0.7 * (err_nearest[inner] ** 2).mean()
