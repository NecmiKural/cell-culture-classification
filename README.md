# Cell Culture Image Classification

Fine-grained, heavily imbalanced multi-class image classification of microscope images of cell cultures in petri dishes. Built as a university deep-learning assignment (COMPX525), iterated across three full experimental versions to push validation accuracy from **60.7% → 83.6%**.

> **Note on data:** The image dataset is the property of a third party and may not be redistributed, so no images are included in this repository. The code, methodology, results, and full report are provided. To run the notebooks you would supply your own image directory in the same `class/` folder layout.

## Problem

- 16 active classes (a–i, k–q), real-world "messy" microscopy data.
- Severe class imbalance: 320:1 between the largest class (1,283 images) and the smallest (4 images).
- High inter-class similarity between visually near-identical cell morphologies — the main source of error.
- 4,695 labelled training images, 512 unlabelled test images at 1024×1024.

## Approach

Three modelling families were compared, each with two architectures, as required by the brief:

1. **Trained from scratch** — a custom SE-ResNet CNN and a Vision Transformer (ViT-Small).
2. **Fine-tuning pretrained networks** — EfficientNet-B0 (v1) and ConvNeXt-Tiny (v2/v3).
3. **Embeddings + classical ML** — features from ResNet-50 / ConvNeXt-Base fed to SVM, XGBoost, and Random Forest.

Final predictions come from a weighted soft-voting **ensemble** of the strongest architectures.

### Preprocessing pipeline (`preprocess.py`)
- Background removal via a grayscale threshold mask (removes the dark region outside the petri dish).
- Resize 1024→224 with INTER_AREA (matches ImageNet input size).
- CLAHE contrast enhancement on the L channel in LAB space.
- Offline augmentation for minority classes to a 200-sample floor, with `aug_`-prefixed filenames kept **out** of validation to avoid leakage.

### Iteration story
| Version | Key change | Val accuracy |
|---|---|---|
| v1 | Baseline (custom CNN + EfficientNet-B0 + embeddings) | 60.69% |
| v2 | Leakage fixes, ConvNeXt-Tiny, better validation split | 76.60% |
| v3 | 5-fold cross-validation ensemble of ConvNeXt-Tiny | **83.55%** |

Each version's failures explicitly informed the next — the methodology section of the report walks through the three V1 flaws that were corrected.

## Repository layout

```
REPORT.md / REPORT.pdf        Full technical report (dataset analysis, methodology, results)
CODE_EXPLANATION_NOTES.md     Walkthrough of the implementation
preprocess.py                 Preprocessing pipeline
notebooks/                    Training notebooks (outputs cleared)
  01_custom_cnn_efficientnet_v1.ipynb
  02_convnext_v2.ipynb
  03_convnext_5fold_v3.ipynb
  04_vit_small.ipynb
figures/                      Result charts (confusion matrices, curves, class distribution)
results/                      Per-version metrics (results_v1..v3, vit) as JSON
submission_final.csv          Final per-image test predictions
```

## Results at a glance

Best single model: ConvNeXt-Tiny (fine-tuned). Best overall: 5-fold ConvNeXt ensemble at **83.55%** validation accuracy. See `figures/` for confusion matrices and training curves, and `REPORT.pdf` for the full discussion.

## Tech

Python · PyTorch · timm · scikit-learn · XGBoost · OpenCV · trained on Kaggle GPU.

---
*University coursework (COMPX525 Deep Learning). Shared as a portfolio sample. Dataset not included by agreement with the data owner.*
