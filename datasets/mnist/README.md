# MNIST

The four files here are the original MNIST distribution, unmodified (gzipped IDX format):

| file | contents | MD5 |
|---|---|---|
| `train-images-idx3-ubyte.gz` | 60 000 training images, 28 × 28, uint8 | `f68b3c2dcbeaaa9fbdd348bbdeb94873` |
| `train-labels-idx1-ubyte.gz` | 60 000 training labels, 0–9 | `d53e105ee54ea40749a09fcbcd1e9432` |
| `t10k-images-idx3-ubyte.gz` | 10 000 test images | `9fb629c4189551a2d022fa330f9573f3` |
| `t10k-labels-idx1-ubyte.gz` | 10 000 test labels | `ec29112dd5afa0611ce80d1b7f02629c` |

Load them with `datasets.load_mnist("train" | "test", limit=None)`. If a file is missing,
the loader downloads it once from `https://ossci-datasets.s3.amazonaws.com/mnist/` and
checks its MD5.

**Source.** Y. LeCun, C. Cortes and C. J. C. Burges, *The MNIST database of handwritten
digits*. Built from NIST Special Databases 1 and 3; described in Y. LeCun, L. Bottou,
Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition",
*Proc. IEEE* 86(11), 1998. Made available under the Creative Commons Attribution-Share
Alike 3.0 licence.
