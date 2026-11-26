"""Software tests for the bundled dataset loaders."""

import numpy as np
import pytest

import datasets


def test_load_adult_train_shapes_and_labels():
    """The train split loads with the documented size and clean {0,1} labels."""
    ds = datasets.load_adult("train")

    # 32561 training instances (datasets/adult/adult.names), column-oriented X.
    assert ds.n_samples == 32561
    assert ds.X.shape == (ds.n_features, ds.n_samples)
    assert ds.y.shape == (ds.n_samples,)
    assert len(ds) == ds.n_samples

    # Labels are integers in {0, 1}; the positive rate matches the known ~24%.
    assert set(np.unique(ds.y).tolist()) == {0, 1}
    assert 0.23 < ds.y.mean() < 0.25

    # Every row of X is named, and X is finite (no NaNs from bad parsing).
    assert len(ds.feature_names) == ds.n_features
    assert np.isfinite(ds.X).all()


def test_continuous_features_are_standardised_on_train():
    """The 6 continuous features are z-scored: ~0 mean, ~1 std on train."""
    ds = datasets.load_adult("train")
    continuous = ds.X[:6]                       # continuous block comes first
    assert np.allclose(continuous.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(continuous.std(axis=1), 1.0, atol=1e-9)


def test_test_split_matches_train_feature_space():
    """Test uses the *same* encoder as train, so feature space is identical."""
    train = datasets.load_adult("train")
    test = datasets.load_adult("test")

    assert test.n_samples == 16281            # documented test size
    assert test.n_features == train.n_features
    assert test.feature_names == train.feature_names
    assert set(np.unique(test.y).tolist()) == {0, 1}
    # Test standardisation uses train stats, so its mean is near — but not
    # exactly — zero (that's the point: no leakage from fitting on test).
    assert np.isfinite(test.X).all()


def test_missing_value_is_its_own_category():
    """The '?' missing token is kept as an explicit one-hot column."""
    ds = datasets.load_adult("train")
    assert "?" in ds.categories["workclass"]
    assert "workclass=?" in ds.feature_names


def test_one_hot_blocks_sum_to_one_per_sample():
    """Each categorical field contributes exactly one active one-hot column."""
    ds = datasets.load_adult("train")
    offset = 6                                  # skip the 6 continuous rows
    # categories is ordered the same way the one-hot blocks are laid out in X.
    for field, vocab in ds.categories.items():
        width = len(vocab)
        block = ds.X[offset:offset + width]     # (width, n_samples)
        assert np.allclose(block.sum(axis=0), 1.0), field
        offset += width
    # The continuous block + every one-hot block together fill all of X.
    assert offset == ds.n_features


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        datasets.load_adult("validation")


# ====================================================================== #
# Flatland — the text corpus
# ====================================================================== #
def test_flatland_loads_text_paragraphs_sentences():
    """The corpus loads with non-empty text, paragraphs, and sentences."""
    flat = datasets.load_flatland()

    assert isinstance(flat.text, str) and len(flat.text) > 10000
    assert flat.n_paragraphs > 100
    assert flat.n_sentences > flat.n_paragraphs      # paragraphs split into more
    assert len(flat) == flat.n_sentences             # len() counts sentences
    assert all(isinstance(s, str) and s for s in flat.sentences)


def test_flatland_strips_gutenberg_boilerplate():
    """The Gutenberg header/footer is removed from the cleaned text."""
    flat = datasets.load_flatland()

    # The marker lines and licensing/catalogue text must be gone...
    assert "START OF THE PROJECT GUTENBERG" not in flat.text
    assert "END OF THE PROJECT GUTENBERG" not in flat.text
    assert "www.gutenberg.org" not in flat.text
    # ...while the actual novella is present.
    assert "Flatland" in flat.text


def test_flatland_paragraphs_have_no_illustration_markers():
    """Whitespace is collapsed and pure ``[Illustration]`` markers are dropped."""
    flat = datasets.load_flatland()

    for para in flat.paragraphs:
        assert "\n" not in para                      # internal newlines collapsed
        assert para == para.strip()
        assert para != "[Illustration]"


def test_flatland_feeds_tokenizer():
    """The sentences train the WordPiece tokenizer end to end.

    Uses a slice + small vocab so the (deliberately un-optimised, clarity-first)
    WordPiece training loop stays fast in the test suite.
    """
    from dlcb import Tokenizer

    flat = datasets.load_flatland()
    tok = Tokenizer()
    tok.build_vocab(flat.sentences[:150], max_size=300)

    assert tok.vocab_size <= 300
    # A common word from the book round-trips through encode/decode (decode
    # re-merges any ## pieces, so this holds even if "square" is split).
    ids = tok.encode("the square", max_len=8)
    assert ids[0] == tok.cls_id
    assert tok.decode(ids) == "the square"


# ---------------------------------------------------------------------- #
# MNIST
# ---------------------------------------------------------------------- #
def test_mnist_splits_have_the_documented_sizes_and_layout():
    train, test = datasets.load_mnist("train"), datasets.load_mnist("test")
    assert len(train) == 60000 and len(test) == 10000
    for ds in (train, test):
        assert ds.images.shape == (len(ds), 28, 28, 1)          # channels-last
        assert ds.images.dtype == np.float64
        assert 0.0 <= ds.images.min() and ds.images.max() <= 1.0
        assert ds.labels.dtype == np.int64
        assert set(np.unique(ds.labels).tolist()) == set(range(10))


def test_mnist_known_first_labels_and_class_balance():
    """The first training labels are famous (5, 0, 4, 1, 9); classes are ~10 % each."""
    train = datasets.load_mnist("train")
    assert train.labels[:5].tolist() == [5, 0, 4, 1, 9]
    assert datasets.load_mnist("test").labels[:5].tolist() == [7, 2, 1, 0, 4]
    share = np.bincount(train.labels) / len(train)
    assert share.min() > 0.09 and share.max() < 0.12


def test_mnist_images_look_like_digits():
    """Background is black and the ink sits in the middle (the digits are centred).

    A few digits touch the edge, so the border is not *always* black — but ink
    there is rare, and the 12 × 12 centre holds about three times the average.
    """
    images = datasets.load_mnist("test", limit=1000).images[..., 0]
    border = np.concatenate([images[:, :4].ravel(), images[:, -4:].ravel()])
    assert (border > 0).mean() < 0.05                           # 4-pixel frame: ~2 % inked
    assert images[:, 8:20, 8:20].mean() > 2.5 * images.mean()   # ink is concentrated
    assert 0.08 < images.mean() < 0.18                          # ~12 % of the pixel mass


def test_mnist_limit_and_flat_view():
    ds = datasets.load_mnist("train", limit=100)
    full = datasets.load_mnist("train")
    assert len(ds) == 100
    assert np.array_equal(ds.images, full.images[:100])
    assert ds.X.shape == (784, 100)
    assert np.array_equal(ds.X[:, 7], ds.images[7].ravel())     # one image per column
    assert ds.y is ds.labels
    with pytest.raises(ValueError):
        datasets.load_mnist("validation")
    with pytest.raises(ValueError):
        datasets.load_mnist("train", limit=0)


def test_mnist_feeds_conv2d_directly():
    from dlcb import engine as cpu
    from dlcb.conv import Conv2d

    ds = datasets.load_mnist("test", limit=4)
    y = Conv2d(1, 8, 5)(cpu.Tensor(ds.images))
    assert y.shape == (4, 24, 24, 8)


# ---------------------------------------------------------------------- #
# H-DIBCO 2018
# ---------------------------------------------------------------------- #
from datasets.loaders import _read_bmp  # noqa: E402

DIBCO_SHAPES = [(811, 1504), (629, 1973), (511, 1013), (289, 1504), (890, 1682),
                (582, 2544), (922, 3933), (286, 1212), (961, 3216), (434, 1777)]


def _write_bmp(rgb: np.ndarray, top_down: bool = False) -> bytes:
    """A minimal 24-bit BMP writer, independent of the reader under test."""
    import struct

    h, w, _ = rgb.shape
    row_bytes = (3 * w + 3) // 4 * 4
    rows = rgb if top_down else rgb[::-1]
    body = b"".join(r[:, ::-1].tobytes() + b"\0" * (row_bytes - 3 * w) for r in rows)
    header = struct.pack("<2sIHHI", b"BM", 54 + len(body), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, w, -h if top_down else h, 1, 24, 0, len(body), 0, 0, 0, 0)
    return header + info + body


@pytest.mark.parametrize("top_down", [False, True])
def test_bmp_reader_handles_padding_row_order_and_bgr(top_down):
    """Width 5 -> 15 bytes per row, padded to 16; distinct values in every channel."""
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    assert np.array_equal(_read_bmp(_write_bmp(rgb, top_down)), rgb)


def test_bmp_reader_rejects_other_formats():
    raw = bytearray(_write_bmp(np.zeros((2, 2, 3), dtype=np.uint8)))
    raw[28] = 8                                               # claim 8 bits per pixel
    with pytest.raises(ValueError):
        _read_bmp(bytes(raw))
    with pytest.raises(ValueError):
        _read_bmp(b"PK" + bytes(raw[2:]))


@pytest.fixture(scope="module")
def dibco():
    return datasets.load_dibco2018()


def test_dibco_pages_sizes_and_types(dibco):
    assert len(dibco) == 10 and dibco.names == [str(k) for k in range(1, 11)]
    for (image, mask), shape in zip((dibco[i] for i in range(10)), DIBCO_SHAPES):
        assert image.shape == shape + (3,) and image.dtype == np.float64
        assert 0.0 <= image.min() and image.max() <= 1.0
        assert mask.shape == shape and set(np.unique(mask)) == {0, 1}


def test_dibco_mask_marks_the_ink(dibco):
    """Flipped correctly and aligned: on every page, mask-1 pixels are darker than mask-0
    ones, and ink is the minority class (3–17 % of the pixels)."""
    for i in range(len(dibco)):
        grey, mask = dibco.grey(i)[..., 0], dibco.masks[i]
        assert 0.03 < mask.mean() < 0.17
        assert grey[mask == 1].mean() < grey[mask == 0].mean() - 0.1


def test_dibco_patches(dibco):
    X, Y = dibco.patches(64, pages=[3])                       # page "4": 289 × 1504
    assert X.shape == ((289 // 64) * (1504 // 64), 64, 64, 3) and Y.shape == X.shape[:3] + (1,)
    assert np.array_equal(X[1], dibco.images[3][0:64, 64:128])     # row-major order
    assert np.array_equal(Y[1, :, :, 0], dibco.masks[3][0:64, 64:128])
    Xg, _ = dibco.patches(64, stride=32, pages=[3], grey=True)
    assert Xg.shape == (((289 - 64) // 32 + 1) * ((1504 - 64) // 32 + 1), 64, 64, 1)
    with pytest.raises(ValueError):
        dibco.patches(1000, pages=[3])


def test_dibco_patches_feed_conv_and_transposed_conv(dibco):
    """Down by 2 with Conv2d, back up by 2 with ConvTranspose2d: the shape of the mask."""
    from dlcb import engine as cpu
    from dlcb.conv import Conv2d, ConvTranspose2d

    X, Y = dibco.patches(32, pages=[7])
    with cpu.no_grad():
        h = Conv2d(3, 4, 4, stride=2, padding=1)(cpu.Tensor(X[:2]))
        out = ConvTranspose2d(4, 1, 4, stride=2, padding=1)(h)
    assert out.shape == Y[:2].shape
