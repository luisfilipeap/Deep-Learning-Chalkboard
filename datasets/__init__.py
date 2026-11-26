"""Example datasets bundled with DL ChalkBoard.

Import this module and call a loader. There are four, one per kind of data:

**Adult** (tabular classification) — model-ready NumPy arrays::

    import datasets

    train = datasets.load_adult("train")
    test  = datasets.load_adult("test")

    train.X            # (n_features, n_samples) float array — column-oriented
    train.y            # (n_samples,) int labels in {0, 1}  (1 == income >50K)
    train.feature_names
    train.categories

The arrays are plain NumPy, so wrap them in the engine when you want to train::

    from dlcb import engine as cpu
    X = cpu.Tensor(train.X)            # already (features, batch)

**Flatland** (text corpus) — cleaned prose for the tokenizer / masked-LM::

    flat = datasets.load_flatland()

    flat.text          # the whole book (str), Gutenberg boilerplate removed
    flat.paragraphs    # list[str], blank-line-separated
    flat.sentences     # list[str], naive .!? split

    from dlcb import Tokenizer
    tok = Tokenizer()
    tok.build_vocab(flat.sentences, max_size=2000)   # learn WordPiece from it

**MNIST** (images) — handwritten digits for the convolutional layers::

    train = datasets.load_mnist("train", limit=5000)   # first 5 000 of 60 000
    test  = datasets.load_mnist("test")                # 10 000

    train.images       # (n, 28, 28, 1) floats in [0, 1] — channels-last, for Conv2d
    train.labels       # (n,) ints in 0..9
    train.X            # (784, n) — one flattened image per column, for a dense MLP

**H-DIBCO 2018** (segmentation) — handwritten pages and which pixels are ink::

    dibco = datasets.load_dibco2018()                  # 10 pages, sizes differ

    image, mask = dibco[0]
    image              # (H, W, 3) floats in [0, 1] — the INPUT, an RGB scan
    mask               # (H, W) ints — the TARGET: 1 = ink, 0 = background
    X, Y = dibco.patches(64, pages=[0, 1, 2])          # (n, 64, 64, 3), (n, 64, 64, 1)
"""

from datasets.loaders import (
    AdultDataset,
    DIBCODataset,
    FlatlandDataset,
    MNISTDataset,
    load_adult,
    load_dibco2018,
    load_flatland,
    load_mnist,
)

__all__ = [
    "load_adult", "AdultDataset",
    "load_flatland", "FlatlandDataset",
    "load_mnist", "MNISTDataset",
    "load_dibco2018", "DIBCODataset",
]
