# H-DIBCO 2018

The two archives here are the benchmark of the ICFHR 2018 Handwritten Document Image
Binarization Contest, unmodified, as published at
<https://vc.ee.duth.gr/h-dibco2018/benchmark/>:

| file | contents | MD5 |
|---|---|---|
| `dibco2018_Dataset.zip` | `dataset/1.bmp` … `10.bmp`: 10 colour scans of handwritten pages, 24-bit BMP | `c335d66ca0b4c226e6c81d41613410fe` |
| `dibco2018-GT.zip` | `gt/1_gt.bmp` … `10_gt.bmp`: ground truth, ink black (0) on white (255); `weights/*.dat`: weights for the competition's pseudo-F-measure (not used by the loader) | `1fc2cffbbc0d61be864cb23519fb4b03` |

Load them with `datasets.load_dibco2018()`. If a file is missing, the loader downloads it
once from the page above and checks its MD5 (the MD5s are those of the files as served on
2026-11-26). The loader flips the ground truth to **1 = ink, 0 = background**.

The same page also offers the competition's evaluation program (`dibco_metrics.zip`,
needs the MATLAB runtime) and the tool that produces the weights (`BinEvalWeights.zip`,
a Windows executable); neither is needed here.

**Source.** I. Pratikakis, K. Zagoris, P. Kaddas, B. Gatos, "ICFHR 2018 Competition on
Handwritten Document Image Binarization (H-DIBCO 2018)", *16th International Conference on
Frontiers in Handwriting Recognition (ICFHR)*, 2018. The images are distributed by the
organisers for research on document binarisation; cite the paper above when you use them.
