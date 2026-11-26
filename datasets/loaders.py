"""Loaders for the example datasets bundled with DL ChalkBoard.

Four datasets live here, for the kinds of data this project trains on:

- **UCI Adult / Census-Income** (``load_adult``) — a *tabular* classification set
  (predict whether a person earns ``>50K`` a year from 14 demographic features).
  The raw files live in ``datasets/adult/`` — ``adult.data`` (train) and
  ``adult.test`` (test).
- **Flatland** (``load_flatland``) — a *text* corpus, the full public-domain
  novella *Flatland: A Romance of Many Dimensions* (Edwin Abbott, Project
  Gutenberg eBook #97), in ``datasets/flatland/corpus.txt``. This is the corpus
  for the tokenizer and masked-language-model side of the project: raw prose to
  learn a vocabulary from and to mask-and-predict over.
- **MNIST** (``load_mnist``) — 70 000 *images* of handwritten digits, 28 × 28
  greyscale, for the convolutional networks. The four original IDX files live in
  ``datasets/mnist/`` (downloaded once, MD5-checked, if missing).
- **H-DIBCO 2018** (``load_dibco2018``) — 10 scanned *handwritten pages* with a
  pixel-level ink/background mask each, for *segmentation*. The two zip files
  of the competition live in ``datasets/dibco2018/`` (downloaded once,
  MD5-checked, if missing); the BMP images are decoded with NumPy alone.

Everything below the Adult section (encoding, standardisation, one-hot) is
specific to that tabular set; the Flatland loader at the bottom is much simpler —
it just cleans the Gutenberg boilerplate and hands back text.

The loader returns data that is **ready to feed the engine**: every feature is
numeric and the layout follows the project's *column-oriented* convention
(``X`` is ``(n_features, n_samples)``, the same orientation ``nn.Linear``
expects), and labels are integers in ``{0, 1}``. Nothing here depends on
``dlcb`` — it produces plain NumPy arrays, so you decide when to wrap them in
a ``Tensor``.

Encoding choices (all standard, kept deliberately simple and inspectable):

- **Continuous** features are *standardised* (z-scored): ``(x - mean) / std``.
- **Categorical** features are *one-hot* encoded; the missing-value token ``"?"``
  is treated as its own category, so no rows are dropped.
- The **label** ``>50K`` -> ``1``, ``<=50K`` -> ``0``.

Crucially, the encoder (the continuous means/stds and the category vocabularies)
is **fit on the training split only** and then *applied* to whichever split you
load. Fitting on test would leak information from the evaluation set into the
features — a classic mistake this loader avoids by construction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# The raw files sit in datasets/adult/ regardless of the caller's working dir.
_ADULT_DIR = Path(__file__).resolve().parent / "adult"

# Column schema, in file order (from datasets/adult/adult.names). 14 features
# followed by the income label.
_COLUMNS: List[str] = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]
_CONTINUOUS: List[str] = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]
_CATEGORICAL: List[str] = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
# The label field (last column). ">50K" is the positive class; the test file
# writes the labels with a trailing period (">50K."), stripped below.
_POSITIVE_LABEL = ">50K"

_SPLIT_FILES = {"train": "adult.data", "test": "adult.test"}


class AdultDataset:
    """A loaded, fully-numeric split of the Adult dataset.

    Attributes
    ----------
    X : np.ndarray
        Feature matrix, shape ``(n_features, n_samples)`` — *column-oriented*,
        matching ``nn.Linear`` (features down axis 0, samples across axis 1).
        Continuous columns are standardised; categoricals are one-hot.
    y : np.ndarray
        Integer labels, shape ``(n_samples,)``, values in ``{0, 1}``
        (``1`` == income ``>50K``).
    feature_names : list of str
        Name of each of the ``n_features`` rows of ``X``. One-hot columns are
        named ``"field=value"`` (e.g. ``"sex=Male"``), so a row of ``X`` is easy
        to trace back to its meaning.
    categories : dict[str, list[str]]
        The vocabulary (fit on train) used to one-hot each categorical field.
    split : str
        Which split this is (``"train"`` or ``"test"``).
    """

    def __init__(self, X, y, feature_names, categories, split) -> None:
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.categories = categories
        self.split = split

    @property
    def n_features(self) -> int:
        return self.X.shape[0]

    @property
    def n_samples(self) -> int:
        return self.X.shape[1]

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (f"AdultDataset(split={self.split!r}, "
                f"n_features={self.n_features}, n_samples={self.n_samples})")


def _read_rows(path: Path) -> List[List[str]]:
    """Parse a raw Adult file into a list of ``[field, ...]`` string rows.

    Skips blank lines and comment lines (the test file opens with a
    ``|1x3 Cross validator`` header), and any row that does not have the
    expected 15 comma-separated fields.
    """
    expected = len(_COLUMNS) + 1                      # 14 features + label
    rows: List[List[str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("|"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != expected:
                continue
            rows.append(parts)
    return rows


def _fit_encoder(train_rows: List[List[str]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """Learn the encoding *from the training split only*.

    Returns the per-feature mean and std for the continuous columns and the
    (sorted) category vocabulary for each categorical column. These are applied,
    unchanged, to whatever split is later encoded — so the test set never
    influences the transformation.
    """
    cont_idx = [_COLUMNS.index(c) for c in _CONTINUOUS]
    cont = np.array([[float(r[i]) for i in cont_idx] for r in train_rows], dtype=np.float64)
    means = cont.mean(axis=0)
    stds = cont.std(axis=0)
    stds[stds == 0.0] = 1.0                            # guard constant columns

    categories: Dict[str, List[str]] = {}
    for c in _CATEGORICAL:
        j = _COLUMNS.index(c)
        categories[c] = sorted({r[j] for r in train_rows})   # includes "?" if present
    return means, stds, categories


def _encode(
    rows: List[List[str]],
    means: np.ndarray,
    stds: np.ndarray,
    categories: Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Turn parsed string rows into a numeric ``(X_rows_by_features, y, names)``.

    ``X`` is built sample-major ``(n_samples, n_features)`` here; the public
    loader transposes it to the column-oriented ``(n_features, n_samples)``.
    """
    n = len(rows)

    # Continuous block: standardise with the train statistics.
    cont_idx = [_COLUMNS.index(c) for c in _CONTINUOUS]
    cont = np.array([[float(r[i]) for i in cont_idx] for r in rows], dtype=np.float64)
    cont = (cont - means) / stds

    # Categorical blocks: one-hot against the train vocabulary. A value unseen in
    # training (rare) maps to an all-zero row for that field rather than erroring.
    feature_names: List[str] = list(_CONTINUOUS)
    blocks = [cont]
    for c in _CATEGORICAL:
        j = _COLUMNS.index(c)
        vocab = categories[c]
        index = {v: k for k, v in enumerate(vocab)}
        block = np.zeros((n, len(vocab)), dtype=np.float64)
        for row_i, r in enumerate(rows):
            k = index.get(r[j])
            if k is not None:
                block[row_i, k] = 1.0
        blocks.append(block)
        feature_names.extend(f"{c}={v}" for v in vocab)

    X = np.hstack(blocks)                              # (n_samples, n_features)

    # Label: strip the test file's trailing "." then map to {0, 1}.
    y = np.array(
        [1 if r[-1].rstrip(".").strip() == _POSITIVE_LABEL else 0 for r in rows],
        dtype=np.int64,
    )
    return X, y, feature_names


def load_adult(split: str = "train") -> AdultDataset:
    """Load a split of the Adult dataset as model-ready NumPy arrays.

    Parameters
    ----------
    split : {"train", "test"}
        Which split to return. The encoder is always fit on ``"train"`` and
        applied to the requested split (see the module docstring).

    Returns
    -------
    AdultDataset
        With ``.X`` ``(n_features, n_samples)``, ``.y`` ``(n_samples,)`` in
        ``{0, 1}``, plus ``.feature_names`` and ``.categories``.

    Examples
    --------
    >>> import datasets
    >>> train = datasets.load_adult("train")
    >>> train
    AdultDataset(split='train', n_features=108, n_samples=32561)
    >>> train.X.shape, train.y.shape
    ((108, 32561), (32561,))
    """
    if split not in _SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(_SPLIT_FILES)}, got {split!r}")

    # Fit on train, then encode the requested split with those same statistics.
    train_rows = _read_rows(_ADULT_DIR / _SPLIT_FILES["train"])
    means, stds, categories = _fit_encoder(train_rows)

    rows = train_rows if split == "train" else _read_rows(_ADULT_DIR / _SPLIT_FILES[split])
    X, y, feature_names = _encode(rows, means, stds, categories)

    # Transpose to the project's column-oriented (n_features, n_samples) layout.
    return AdultDataset(
        X=X.T, y=y, feature_names=feature_names, categories=categories, split=split
    )


# ====================================================================== #
# Flatland — a plain-text corpus for the tokenizer / masked-LM
# ====================================================================== #

# The novella sits next to this module regardless of the caller's working dir.
_FLATLAND_FILE = Path(__file__).resolve().parent / "flatland" / "corpus.txt"

# Project Gutenberg wraps the actual book between these two marker lines. Anything
# before the first / after the second is licensing and catalogue boilerplate.
_GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG"
_GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG"

# A deliberately *naive* sentence splitter: break after ., ! or ? when followed by
# whitespace. It has no idea about abbreviations ("Mr.", "e.g."), so it will
# occasionally over-split — fine for a didactic corpus, and documented as such.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class FlatlandDataset:
    """The Flatland corpus, cleaned of Gutenberg boilerplate, in three forms.

    The same text is offered at three granularities so the caller picks whatever
    a task needs — the whole string, paragraphs, or sentences. A ``Tokenizer``
    trains on any list of strings, so ``build_vocab(ds.sentences)`` or
    ``build_vocab(ds.paragraphs)`` both work directly.

    Attributes
    ----------
    text : str
        The full book text with the Gutenberg header/footer removed.
    paragraphs : list of str
        Blank-line-separated paragraphs, each with internal whitespace collapsed
        to single spaces (illustration/section markers like ``[Illustration]`` are
        dropped).
    sentences : list of str
        The paragraphs split into sentences by a naive ``.!?`` rule.
    """

    def __init__(self, text: str, paragraphs: List[str], sentences: List[str]) -> None:
        self.text = text
        self.paragraphs = paragraphs
        self.sentences = sentences

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    def __len__(self) -> int:
        # Sentences are the natural training unit, so len() counts them.
        return len(self.sentences)

    def __repr__(self) -> str:
        return (f"FlatlandDataset(n_paragraphs={self.n_paragraphs}, "
                f"n_sentences={self.n_sentences}, chars={len(self.text)})")


def _strip_gutenberg(raw: str) -> str:
    """Return only the text between the Gutenberg START and END marker lines.

    Falls back to the whole input if the markers are absent, so a plain text file
    without Gutenberg wrapping still loads.
    """
    lines = raw.splitlines()
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        if line.startswith(_GUTENBERG_START):
            start = i + 1
        elif line.startswith(_GUTENBERG_END):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def _split_paragraphs(text: str) -> List[str]:
    """Split cleaned text into paragraphs on blank lines.

    Each paragraph's internal newlines and runs of spaces are collapsed to single
    spaces, and pure markup markers (a line that is entirely ``[...]``, e.g.
    ``[Illustration]``) are dropped.
    """
    paragraphs: List[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        para = " ".join(chunk.split())          # collapse all whitespace
        if not para:
            continue
        if para.startswith("[") and para.endswith("]"):
            continue                            # illustration / section marker
        paragraphs.append(para)
    return paragraphs


def _split_sentences(paragraphs: List[str]) -> List[str]:
    """Break each paragraph into sentences with the naive ``.!?`` rule."""
    sentences: List[str] = []
    for para in paragraphs:
        for sent in _SENTENCE_BOUNDARY.split(para):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences


def load_flatland() -> FlatlandDataset:
    """Load the Flatland text corpus as cleaned paragraphs and sentences.

    Reads ``datasets/flatland/corpus.txt``, strips the Project Gutenberg
    header/footer, and returns a :class:`FlatlandDataset` exposing ``.text``,
    ``.paragraphs`` and ``.sentences``. Pure text in, plain Python strings out —
    nothing here depends on ``dlcb``.

    Returns
    -------
    FlatlandDataset
        With ``.text`` (str), ``.paragraphs`` (list of str) and ``.sentences``
        (list of str).

    Examples
    --------
    >>> import datasets
    >>> flat = datasets.load_flatland()
    >>> flat.sentences[0]
    'I call our world Flatland, not because we call it so, ...'
    >>> from dlcb import Tokenizer
    >>> tok = Tokenizer()
    >>> tok.build_vocab(flat.sentences, max_size=2000)   # learn WordPiece from it
    """
    raw = _FLATLAND_FILE.read_text(encoding="utf-8")
    text = _strip_gutenberg(raw)
    paragraphs = _split_paragraphs(text)
    sentences = _split_sentences(paragraphs)
    return FlatlandDataset(text=text, paragraphs=paragraphs, sentences=sentences)


# ====================================================================== #
# MNIST — handwritten digits for the convolutional networks
# ====================================================================== #

# The four original IDX files sit in datasets/mnist/, gzipped, exactly as
# published. If one is missing, it is downloaded once (standard library only)
# and checked against its MD5 before use.
_MNIST_DIR = Path(__file__).resolve().parent / "mnist"
_MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_MNIST_FILES = {
    # split: (images file, labels file)
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}
_MNIST_MD5 = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
# IDX magic numbers: 0x00000803 = unsigned bytes, 3 dimensions (images);
# 0x00000801 = unsigned bytes, 1 dimension (labels).
_IDX_IMAGES, _IDX_LABELS = 2051, 2049


class MNISTDataset:
    """A split of MNIST: 28 × 28 greyscale digits and their labels 0–9.

    Attributes
    ----------
    images : np.ndarray
        Shape ``(n_samples, 28, 28, 1)``, float64 in ``[0, 1]`` (pixel / 255).
        **Channels-last**, the ``(batch, H, W, C)`` layout of ``dlcb.conv`` and
        the ``H × W × C`` of Lecture V — feed it to ``Conv2d`` as is.
        Background is 0, ink is close to 1.
    labels : np.ndarray
        Shape ``(n_samples,)``, int64 in ``{0, …, 9}``.
    split : str
        ``"train"`` (60 000 digits) or ``"test"`` (10 000).

    ``X`` gives the same pixels as a *column-oriented* matrix
    ``(784, n_samples)`` — one flattened image per column, row-major — for a
    dense ``nn.Linear`` / ``MLP`` baseline, matching ``AdultDataset.X``.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, split: str) -> None:
        self.images = images
        self.labels = labels
        self.split = split

    @property
    def X(self) -> np.ndarray:
        """Pixels as ``(784, n_samples)``: one flattened image per column."""
        return self.images.reshape(len(self), -1).T

    @property
    def y(self) -> np.ndarray:
        """Alias of ``labels``, the name ``AdultDataset`` uses."""
        return self.labels

    @property
    def n_samples(self) -> int:
        return self.images.shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return f"MNISTDataset(split={self.split!r}, n_samples={self.n_samples}, shape=(28, 28, 1))"


def _mnist_file(name: str) -> Path:
    """Path to an MNIST file, downloading (and MD5-checking) it if missing."""
    import hashlib

    path = _MNIST_DIR / name
    if not path.exists():
        import urllib.request

        _MNIST_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(_MNIST_MIRROR + name, tmp)
        except OSError as err:
            raise RuntimeError(
                f"{path} is missing and could not be downloaded ({err}). Download "
                f"{_MNIST_MIRROR + name} by hand into {_MNIST_DIR}/."
            ) from err
        tmp.replace(path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != _MNIST_MD5[name]:
        raise RuntimeError(f"{path} is corrupt (MD5 {digest}); delete it to re-download")
    return path


def _read_idx(path: Path, magic: int) -> np.ndarray:
    """Parse a gzipped IDX file: a big-endian header, then raw unsigned bytes.

    The format (LeCun et al.) is a 4-byte magic number, one 4-byte size per
    dimension, then the data in row-major order. Ten lines of standard library
    and ``np.frombuffer`` — no image library needed.
    """
    import gzip

    raw = gzip.decompress(path.read_bytes())
    found = int.from_bytes(raw[0:4], "big")
    if found != magic:
        raise ValueError(f"{path.name}: magic number {found}, expected {magic}")
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(split: str = "train", limit: Optional[int] = None) -> MNISTDataset:
    """Load a split of MNIST as NumPy arrays ready for ``dlcb.conv``.

    Parameters
    ----------
    split : {"train", "test"}
        60 000 training or 10 000 test digits. The test digits were written by
        different people from the training ones, so it measures generalisation
        to new handwriting, not to new copies of familiar writers.
    limit : int, optional
        Keep only the first ``limit`` digits. A pure-NumPy CNN processes a few
        hundred images per second, so a few thousand training digits is the
        practical size for an exercise. The original order is kept (it is not
        sorted by label), so the first ``limit`` digits contain every class.

    Returns
    -------
    MNISTDataset
        ``.images`` ``(n, 28, 28, 1)`` floats in ``[0, 1]``, ``.labels`` ``(n,)``
        ints, and ``.X`` ``(784, n)`` for dense baselines.

    Examples
    --------
    >>> import datasets
    >>> train = datasets.load_mnist("train", limit=5000)
    >>> train
    MNISTDataset(split='train', n_samples=5000, shape=(28, 28, 1))
    >>> train.images.shape, train.labels[:5]
    ((5000, 28, 28, 1), array([5, 0, 4, 1, 9]))
    """
    if split not in _MNIST_FILES:
        raise ValueError(f"split must be one of {sorted(_MNIST_FILES)}, got {split!r}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit}")
    image_file, label_file = _MNIST_FILES[split]
    images = _read_idx(_mnist_file(image_file), _IDX_IMAGES)
    labels = _read_idx(_mnist_file(label_file), _IDX_LABELS)
    if limit is not None:
        images, labels = images[:limit], labels[:limit]
    images = (images.astype(np.float64) / 255.0)[..., None]      # (n, 28, 28, 1)
    return MNISTDataset(images=images, labels=labels.astype(np.int64), split=split)


# ====================================================================== #
# H-DIBCO 2018 — handwritten document binarisation (pixel segmentation)
# ====================================================================== #

# The two archives of the competition's benchmark page sit in
# datasets/dibco2018/, exactly as published. If one is missing it is
# downloaded once (standard library only) and checked against its MD5. The
# images are read straight out of the zip files; nothing is extracted.
_DIBCO_DIR = Path(__file__).resolve().parent / "dibco2018"
_DIBCO_URL = "https://vc.ee.duth.gr/h-dibco2018/benchmark/"
_DIBCO_MD5 = {
    "dibco2018_Dataset.zip": "c335d66ca0b4c226e6c81d41613410fe",   # 10 colour scans
    "dibco2018-GT.zip": "1fc2cffbbc0d61be864cb23519fb4b03",        # their ground truth
}
_DIBCO_N = 10


class DIBCODataset:
    """H-DIBCO 2018: ten handwritten pages and, for each, which pixels are ink.

    **Task.** *Binarisation*: label every pixel of a scanned page as **ink**
    (text) or **background** (paper, stains, bleed-through from the other side
    of the sheet, shadows). It is semantic segmentation with two classes: the
    input is an image and the output is an image of the same height and width.

    Attributes
    ----------
    images : list of np.ndarray
        One per page, ``(H, W, 3)`` float64 RGB in ``[0, 1]``, channels-last
        like ``dlcb.conv``. **The pages have different sizes** (from 286 × 1212
        to 922 × 3933 pixels), so this is a list, not one array.
    masks : list of np.ndarray
        One per page, ``(H, W)`` int64, the **target**: ``1`` = ink, ``0`` =
        background. (The competition's files draw ink black, 0, on white, 255;
        the loader flips them so that 1 marks the class of interest, the usual
        convention for a segmentation target.)
    names : list of str
        ``"1"`` … ``"10"``, the competition's file names.

    Ink is rare — between 3 % and 16 % of the pixels of a page — so a model that
    answers "background" everywhere already gets about 92 % of the pixels right.
    Pixel accuracy is therefore a poor score here; the competition ranks by
    F-measure on the ink class (with PSNR and others alongside).

    ``patches`` cuts the pages into equal squares, which *can* be stacked into a
    batch for a network.
    """

    def __init__(self, images: List[np.ndarray], masks: List[np.ndarray], names: List[str]) -> None:
        self.images = images
        self.masks = masks
        self.names = names

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(image (H, W, 3), mask (H, W))`` of page ``i``."""
        return self.images[i], self.masks[i]

    def grey(self, i: int) -> np.ndarray:
        """Page ``i`` in grey levels, ``(H, W, 1)`` in ``[0, 1]`` (ITU-R BT.601 luma)."""
        return (self.images[i] @ np.array([0.299, 0.587, 0.114]))[..., None]

    def patches(
        self, size: int, stride: Optional[int] = None, pages: Optional[List[int]] = None,
        grey: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cut pages into ``size × size`` squares, ``stride`` pixels apart.

        Returns ``X`` ``(n, size, size, C)`` (``C`` = 3, or 1 with ``grey=True``)
        and ``Y`` ``(n, size, size, 1)`` with the mask of each square. Squares
        that would cross the right or bottom edge are skipped. ``stride``
        defaults to ``size`` (no overlap). ``pages`` selects which pages to cut
        (indices into the dataset) — keep the pages you evaluate on out of the
        ones you train on.
        """
        stride = stride or size
        xs, ys = [], []
        for i in (range(len(self)) if pages is None else pages):
            image = self.grey(i) if grey else self.images[i]
            mask = self.masks[i]
            h, w = mask.shape
            for top in range(0, h - size + 1, stride):
                for left in range(0, w - size + 1, stride):
                    xs.append(image[top:top + size, left:left + size])
                    ys.append(mask[top:top + size, left:left + size, None])
        if not xs:
            raise ValueError(f"no {size}×{size} patch fits in the selected pages")
        return np.stack(xs), np.stack(ys)

    def __repr__(self) -> str:
        return f"DIBCODataset(n_pages={len(self)}, sizes from {min(m.size for m in self.masks):,} to {max(m.size for m in self.masks):,} pixels)"


def _dibco_file(name: str) -> Path:
    """Path to a DIBCO archive, downloading (and MD5-checking) it if missing."""
    import hashlib

    path = _DIBCO_DIR / name
    if not path.exists():
        import urllib.request

        _DIBCO_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(_DIBCO_URL + name, tmp)
        except OSError as err:
            raise RuntimeError(
                f"{path} is missing and could not be downloaded ({err}). Download "
                f"{_DIBCO_URL + name} by hand into {_DIBCO_DIR}/."
            ) from err
        tmp.replace(path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != _DIBCO_MD5[name]:
        raise RuntimeError(f"{path} is corrupt (MD5 {digest}); delete it to re-download")
    return path


def _read_bmp(raw: bytes) -> np.ndarray:
    """Decode an uncompressed 24-bit Windows BMP into ``(H, W, 3)`` uint8 RGB.

    The format is a 14-byte file header (``"BM"``, size, offset of the pixels)
    and a 40-byte info header (width, height, bits per pixel, compression),
    then the pixels: **bottom row first** when the height is positive, each
    pixel stored **B, G, R**, and each row padded to a multiple of 4 bytes.
    Undoing those three quirks is all it takes — no image library needed.
    """
    import struct

    if raw[:2] != b"BM":
        raise ValueError("not a BMP file")
    offset = struct.unpack("<I", raw[10:14])[0]
    width, height = struct.unpack("<ii", raw[18:26])
    bits, compression = struct.unpack("<HI", raw[28:34])
    if bits != 24 or compression != 0:
        raise ValueError(f"only uncompressed 24-bit BMP is supported, got {bits}-bit, compression {compression}")
    rows = abs(height)
    row_bytes = (3 * width + 3) // 4 * 4                       # padded to 4 bytes
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=offset, count=row_bytes * rows)
    pixels = pixels.reshape(rows, row_bytes)[:, :3 * width].reshape(rows, width, 3)
    pixels = pixels[:, :, ::-1]                                # BGR -> RGB
    return pixels[::-1] if height > 0 else pixels              # bottom-up -> top-down


def load_dibco2018() -> DIBCODataset:
    """Load the H-DIBCO 2018 benchmark: 10 handwritten pages and their ink masks.

    These ten pages are the competition's *test* set; it has no training split.
    For an exercise, cut them with ``DIBCODataset.patches`` and keep some pages
    out for evaluation.

    Returns
    -------
    DIBCODataset
        ``.images[i]`` ``(H, W, 3)`` floats in ``[0, 1]`` — the input;
        ``.masks[i]`` ``(H, W)`` ints, 1 = ink, 0 = background — the target.

    Examples
    --------
    >>> import datasets
    >>> dibco = datasets.load_dibco2018()
    >>> image, mask = dibco[0]
    >>> image.shape, mask.shape, mask.mean().round(3)
    ((811, 1504, 3), (811, 1504), 0.051)
    >>> X, Y = dibco.patches(64, pages=[0, 1, 2])
    >>> X.shape[1:], Y.shape[1:]
    ((64, 64, 3), (64, 64, 1))
    """
    import zipfile

    images, masks, names = [], [], []
    with zipfile.ZipFile(_dibco_file("dibco2018_Dataset.zip")) as scans, \
            zipfile.ZipFile(_dibco_file("dibco2018-GT.zip")) as truth:
        for k in range(1, _DIBCO_N + 1):
            image = _read_bmp(scans.read(f"dataset/{k}.bmp"))
            gt = _read_bmp(truth.read(f"gt/{k}_gt.bmp"))
            if gt.shape != image.shape:
                raise ValueError(f"page {k}: image {image.shape} and ground truth {gt.shape} differ")
            images.append(image.astype(np.float64) / 255.0)
            masks.append((gt[:, :, 0] == 0).astype(np.int64))  # black (0) = ink -> 1
            names.append(str(k))
    return DIBCODataset(images, masks, names)
