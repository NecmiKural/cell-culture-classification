# COMPX525-26A Assignment 1: Petri Dish Cell Culture Classification

## 1. Introduction

This report presents a comprehensive multi-class image classification system for cell culture images captured in petri dishes. The task involves classifying 512 test images into 16 active classes (a–i, k–q), where class 'j' is excluded due to having zero training samples. The dataset exhibits extreme class imbalance (ranging from 4 to 1,283 samples per class) and contains visually similar classes, making it a challenging fine-grained classification problem.

Three distinct modelling approaches were explored as required by the assignment, each comparing two different architectures: (1) training from scratch using a custom SE-ResNet (CNN) and a Vision Transformer (ViT-Small), (2) fine-tuning pretrained models using EfficientNet-B0 (V1) and ConvNeXt-Tiny (V2/V3), and (3) classical ML classifiers applied to embeddings from ResNet-50 (V1) and ConvNeXt-Base (V2/V3). A weighted soft-voting ensemble of the best-performing architectures produces the final test predictions.

Three full experimental iterations were conducted. V1 served as a baseline to identify methodological issues, V2 incorporated systematic fixes that yielded a 16-point improvement (60.69% → 76.60%), and V3 introduced 5-fold cross-validation ensembling of ConvNeXt-Tiny models, pushing the final ensemble to **83.55%** validation accuracy. This iterative approach was deliberate: each iteration's failures and successes directly informed the next version's design.

## 2. Dataset Analysis

### 2.1 Class Distribution

The original training set contains 4,695 images across 17 labelled classes. Class 'j' has zero training samples and was excluded, leaving 16 active classes. The distribution is severely imbalanced with an imbalance ratio (largest to smallest) of 320:1 between class 'a' (1,283 images) and class 'b' (4 images).

| Class | Original Count | After Augmentation | Proportion (%) |
|-------|---------------|-------------------|----------------|
| a | 1,283 | 1,283 | 27.33 |
| g | 720 | 720 | 15.34 |
| l | 660 | 660 | 14.06 |
| m | 543 | 543 | 11.57 |
| c | 475 | 475 | 10.12 |
| d | 266 | 266 | 5.66 |
| i | 199 | 200 | 4.24 |
| o | 197 | 200 | 4.20 |
| e | 117 | 200 | 2.49 |
| n | 91 | 200 | 1.94 |
| f | 59 | 200 | 1.26 |
| p | 35 | 200 | 0.75 |
| q | 29 | 200 | 0.62 |
| k | 12 | 200 | 0.26 |
| h | 5 | 200 | 0.11 |
| b | 4 | 200 | 0.09 |

Total after preprocessing: 5,947 training images (4,695 original + 1,252 augmented). The test set contains 512 unlabelled images.

### 2.2 Visual Characteristics and Challenges

Manual inspection of the dataset revealed several challenges:

**Dark background artifacts:** Many images contain dark regions outside the petri dish boundary that could confuse classifiers if not removed. These artifacts vary across images and do not carry diagnostic information.

**Inter-class similarity:** Classes 'a', 'l', and 'm' share similar cell density and morphological features — confluent or semi-confluent monolayers with comparable texture. This makes them particularly hard to distinguish and is the primary source of classification errors across all models.

**Intra-class variability:** Some classes (e.g., 'd', 'i') show significant variation in imaging conditions including brightness, focus, and zoom level, requiring robust augmentation strategies.

**Extreme minority classes:** Classes 'b' (4 images) and 'h' (5 images) provide almost no training signal. While they also have very few test samples (1 each), any classifier must still handle them.

### 2.3 Validation Strategy

A 15% stratified holdout split was used for validation (approximately 705 validation images from original data only). Critically, augmented images (identifiable by the `aug_` filename prefix) were **excluded from the validation set** in V2, ensuring that performance estimates reflect generalization to real images rather than near-duplicates of training samples. This was one of the most important methodological corrections from V1 (see Section 4.1).

## 3. Preprocessing Pipeline

Preprocessing was applied uniformly to all images (train and test), ensuring consistency between training and inference.

### 3.1 Background Removal

A binary threshold mask was computed on the grayscale image (threshold=15) to identify dark background pixels. These were zeroed out using bitwise AND, removing background artifacts that could otherwise introduce spurious features. The threshold of 15 was chosen empirically — low enough to preserve dim cell regions while eliminating the near-black background.

### 3.2 Resize

All images were resized from their original 1024×1024 resolution to 224×224 pixels using INTER_AREA interpolation (best for downsampling as it avoids aliasing). The 224×224 size was chosen for direct compatibility with ImageNet-pretrained model architectures, where this is the standard input size. While the README suggested 256×256, using 224×224 eliminates the need for a separate centre crop or resize within the model pipeline.

### 3.3 CLAHE Enhancement

Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied to the L (lightness) channel in LAB colour space with clipLimit=2.0 and tileGridSize=8×8. Working in LAB space ensures only luminance contrast is enhanced without distorting colour information. CLAHE provides locally adaptive contrast enhancement, which is particularly important for cell culture images where subtle texture differences between classes are diagnostically significant but may be visually subtle in raw images. The clip limit of 2.0 prevents over-amplification of noise in homogeneous regions.

### 3.4 Offline Augmentation for Minority Classes

Classes with fewer than 200 samples received synthetic augmentations to establish a minimum baseline. Augmentation operations included: random rotation (±20°), horizontal/vertical flips, brightness/contrast jitter (alpha ∈ [0.8, 1.2], beta ∈ [-20, 20]), random zoom (0.85–1.0×), and occasional Gaussian blur (15% probability, 3×3 kernel). This offline augmentation was applied during preprocessing (not during training) and generated images with the `aug_` filename prefix, enabling clean separation during validation.

The target of 200 samples per minority class was a conservative choice — enough to provide some learning signal without overwhelming the dataset with synthetic images that might reduce overall quality.

## 4. Experimental Methodology

### 4.1 V1 Baseline and Lessons Learned

The first iteration (V1) employed a custom ResNet-style CNN (~11.2M parameters), EfficientNet-B0 fine-tuning, and ResNet-50 embeddings with SVM/XGBoost/Random Forest. The best ensemble achieved only **60.69%** validation accuracy. Post-hoc analysis revealed three critical methodological flaws:

**Augmented data leakage into validation:** Augmented images were included in the validation set. Since offline augmentations are near-copies of their source images (differing only in geometric/photometric transforms), validation metrics for minority classes were artificially inflated. Classes 'b', 'h', and 'k' all showed ~100% validation accuracy in V1 — not because the model generalized well, but because it had seen near-identical images during training.

**Over-aggressive class rebalancing:** V1 applied both inverse-frequency weighted sampling in the DataLoader *and* inverse-frequency weighted cross-entropy loss simultaneously. This double rebalancing caused the model to over-correct, heavily biasing predictions toward minority classes at the expense of majority class accuracy. The effect was particularly damaging because majority classes ('a', 'g', 'l') collectively represent over 56% of the data.

**Insufficient model capacity for fine-tuning:** EfficientNet-B0 (5.3M parameters), while efficient, lacked the representational capacity to capture the fine-grained distinctions needed for this dataset's visually similar classes. Its compact architecture was designed for mobile deployment, not for maximizing accuracy on challenging fine-grained problems.

### 4.2 V2 Design Improvements

V2 systematically addressed each V1 issue:

**Clean validation split:** Augmented files (identified by `aug_` prefix) are used exclusively for training; validation uses only original images. This provides honest, unbiased performance estimates.

**Softer class rebalancing via square-root weighting:** Instead of pure inverse frequency, V2 uses square-root inverse frequency weighting: w_c = √(N_total / (N_classes × N_c)), normalized to mean=1. This moderates the rebalancing effect — minority classes still receive higher weight, but the correction is less extreme, preserving majority class performance. This approach is well-established in the imbalanced learning literature as a compromise between uniform and fully balanced weighting.

**Focal Loss with label smoothing:** Focal Loss (Lin et al., 2017) with γ=2.0 dynamically down-weights easy examples and focuses learning on hard-to-classify samples. The focal modulating factor (1 − p_t)^γ reduces the loss contribution of well-classified examples exponentially. Label smoothing (ε=0.1) replaces hard targets (0/1) with soft targets (ε/K, 1−ε+ε/K), preventing overconfident predictions and acting as implicit regularization against overfitting.

**Stronger pretrained backbone:** ConvNeXt-Tiny (Liu et al., 2022) pretrained on ImageNet-22K replaced EfficientNet-B0. ImageNet-22K contains 14M images across 21,841 classes — 14× more classes than ImageNet-1K — providing substantially richer feature representations that transfer better to novel domains.

**Albumentations augmentation pipeline:** The training-time augmentation was upgraded to use the Albumentations library with stronger transforms: ElasticTransform, OpticalDistortion, CoarseDropout (simulating occlusions), MotionBlur, and ShiftScaleRotate. These provide more diverse training views than simple flips and rotations, improving generalization.

**Layer-wise discriminative learning rates:** During fine-tuning, earlier layers receive proportionally lower learning rates via linear scaling from 10% to 100% of the base LR across network depth. This preserves transferable low-level features (edges, textures) while allowing task-specific higher-level features to adapt more aggressively. Formally, for layer at position p out of L total layers: lr_p = base_lr × (0.1 + 0.9 × p/L).

## 5. Approach 1: Training from Scratch (SE-ResNet vs ViT-Small)

Two fundamentally different architectures were trained from scratch to evaluate whether CNN inductive biases (locality, translation equivariance) or Transformer global attention is more effective on this small dataset.

### 5.1 Architecture 1: SE-ResNet (CNN)

A custom SE-ResNet architecture was designed incorporating Squeeze-and-Excitation (SE) attention blocks (Hu et al., 2018). The SE mechanism adaptively recalibrates channel-wise feature responses by modelling inter-channel dependencies through a "squeeze" (global average pooling) and "excitation" (two FC layers with ReLU and sigmoid) pathway. This allows the network to emphasize informative feature channels and suppress less useful ones — particularly valuable when subtle texture differences distinguish classes.

**Network structure:**

- **Stem:** 7×7 convolution (stride 2, 64 filters) → BatchNorm → ReLU → 3×3 MaxPool (stride 2)
- **Stage 1:** 3 SE-Residual blocks, 64 channels
- **Stage 2:** 4 SE-Residual blocks, 128 channels (stride 2 downsample)
- **Stage 3:** 6 SE-Residual blocks, 256 channels (stride 2 downsample)
- **Stage 4:** 3 SE-Residual blocks, 512 channels (stride 2 downsample)
- **Classifier:** AdaptiveAvgPool(1×1) → Dropout(0.4) → FC(512, 256) → ReLU → Dropout(0.2) → FC(256, 16)

Each SE-Residual block consists of: Conv3×3 → BN → ReLU → Conv3×3 → BN → SE(reduction=16) → residual addition → ReLU. The SE reduction ratio of 16 keeps the computational overhead minimal (~2.5% additional parameters) while providing channel attention.

Total parameters: ~21.5M, all trainable. The stage configuration [3, 4, 6, 3] follows the proven ResNet-50 pattern.

### 5.2 Training Configuration

- **Optimizer:** AdamW (lr=1×10⁻³, weight_decay=5×10⁻⁴). AdamW decouples weight decay from gradient updates, providing more effective regularization than L2-penalized Adam.
- **Scheduler:** CosineAnnealingWarmRestarts (T_0=15, T_mult=2). Periodic learning rate resets help escape local minima and explore different loss basins. The warm restart period doubles after each cycle (15 → 30 → 60 epochs).
- **Loss:** Focal Loss (γ=2.0, label_smoothing=0.1) with sqrt-balanced class weights.
- **Mixed precision (AMP):** Automatic Mixed Precision with GradScaler for reduced GPU memory usage and faster computation.
- **Early stopping:** Patience of 20 epochs monitoring validation accuracy.
- **Batch size:** 32 with 4 gradient accumulation steps (effective batch size 128).

### 5.3 Results and Analysis

**Validation accuracy: 52.06% (V3) / 43.97% (V2)**

In V3, the custom SE-Net achieved 52.06% validation accuracy — an 8-point improvement over V2, attributable to the stronger Albumentations augmentation pipeline applied during V3 training. Nevertheless, the model still exhibits the fundamental limitation of training from scratch on a small dataset. The train-val gap indicates moderate overfitting despite regularization.

**Per-class analysis reveals a clear pattern:**

| Performance Tier | Classes | Recall Range |
|-----------------|---------|-------------|
| Strong (>75%) | b (100%), h (100%), q (100%), p (100%), e (82%) | Very distinctive visual features |
| Moderate (40-75%) | a (63%), i (53%), k (50%), d (43%), c (42%) | Some unique features but overlap exists |
| Weak (<40%) | o (30%), n (29%), l (22%), m (1%) | High visual similarity with other classes |

The catastrophic failure on class 'm' (1.2% recall — only 1 of 81 correctly classified) demonstrates a fundamental limitation: with only ~4,700 original images across 16 classes, there is simply insufficient data to learn the fine-grained features needed to distinguish visually similar classes from scratch. The model defaults to predicting majority classes ('a', 'g') for ambiguous inputs.

### 5.4 Hyperparameters Explored

| Parameter | Values Tried | Best | Rationale |
|-----------|-------------|------|-----------|
| Learning rate | 5×10⁻⁴, 1×10⁻³, 2×10⁻³ | 1×10⁻³ | Lower rates underfit; higher rates unstable |
| Weight decay | 1×10⁻⁴, 5×10⁻⁴, 1×10⁻³ | 5×10⁻⁴ | Balances regularization vs. underfitting |
| Dropout | 0.3, 0.4, 0.5 | 0.4 | Higher dropout reduced training signal too much |
| Focal gamma | 1.0, 2.0, 3.0 | 2.0 | γ=3.0 over-penalized easy examples |
| SE reduction | 8, 16 | 16 | Reduction=8 added parameters without benefit |

### 5.5 Architecture 2: Vision Transformer (ViT-Small)

To compare a fundamentally different architecture family, a small Vision Transformer was trained from scratch. The ViT splits each 224×224 image into 196 non-overlapping 16×16 patches, projects each patch to a 256-dimensional embedding, prepends a learnable [CLS] token, adds learnable positional embeddings, and processes the sequence through 6 Transformer encoder blocks (4 attention heads each, MLP ratio 4×). The final classification uses the [CLS] token output. Total parameters: ~5.0M.

**ViT-Small configuration:**

- **Patch embedding:** 16×16 patches → 196 tokens per image, projected to 256 dimensions
- **Transformer blocks:** 6 layers, each with multi-head self-attention (4 heads, head_dim=64) and MLP (256→1024→256) with GELU activation
- **Regularization:** Dropout(0.1) in attention and MLP, Dropout(0.3) before classifier
- **Initialization:** Truncated normal (std=0.02) for all weights, following ViT conventions

The same training pipeline was used (Focal Loss, sqrt-balanced weights, Albumentations augmentation, AdamW with lr=5×10⁻⁴, CosineAnnealingLR).

**Validation accuracy: 31.06%**

As expected, the ViT significantly underperformed the SE-ResNet. This result is consistent with established findings in the literature: Vision Transformers lack the inductive biases that CNNs possess — specifically, locality (each convolutional filter operates on a small spatial neighbourhood) and translation equivariance (the same filter is applied across all spatial positions). These biases act as strong priors that reduce the amount of training data needed to learn useful visual features. Without them, the ViT must learn spatial relationships entirely from data, which requires datasets orders of magnitude larger than our ~4,700 images. Dosovitskiy et al. (2021) showed that ViTs only match or exceed CNNs when pretrained on datasets of 14M+ images (ImageNet-21K) or 300M+ images (JFT-300M).

### 5.6 Approach 1 Summary

| Architecture | Type | Parameters | Val Accuracy | Key Insight |
|---|---|---|---|---|
| SE-ResNet | CNN | ~21.5M | 52.06% | CNN inductive biases provide an advantage on small datasets |
| ViT-Small | Transformer | ~5.0M | 31.06% | Transformers need much larger datasets to learn spatial structure |

The SE-ResNet's superior performance confirms that for small-dataset image classification, CNN architectures with their built-in spatial priors remain the better choice when training from scratch. The ViT's poor performance is not a failure of the architecture itself, but a demonstration of the data requirements inherent to attention-based models.

## 6. Approach 2: Fine-tuning Pretrained Models (EfficientNet-B0 vs ConvNeXt-Tiny)

Two pretrained architectures were fine-tuned to compare how model capacity and pretraining data scale affect transfer learning performance on this dataset.

### 6.1 Architecture 1: EfficientNet-B0 (V1)

EfficientNet-B0 (Tan & Le, 2019) was the first pretrained model attempted, using ImageNet-1K weights (5.3M parameters). A custom classifier head replaced the original: Dropout(0.3) → Linear(1280, 256) → ReLU → Dropout(0.2) → Linear(256, 16). Training followed a two-phase strategy: Phase 1 froze the backbone and trained only the head (10 epochs, lr=1×10⁻³), then Phase 2 unfroze all layers with discriminative learning rates (backbone lr=1×10⁻⁵, head lr=5×10⁻⁴) for 30 epochs. CrossEntropyLoss with inverse-frequency class weights was used.

**Validation accuracy: 52.30%** — The relatively modest accuracy was attributed to: (1) EfficientNet-B0's compact architecture (designed for mobile efficiency, not maximum accuracy), (2) ImageNet-1K pretraining providing narrower feature diversity than ImageNet-22K, and (3) V1's methodological issues (augmented data in validation, over-aggressive rebalancing).

### 6.2 Architecture 2: ConvNeXt-Tiny (V2/V3)

ConvNeXt-Tiny (Liu et al., 2022) modernizes the ResNet design by incorporating techniques from Vision Transformers into a pure CNN framework: a patchify stem (4×4 non-overlapping convolutions replacing the 7×7+maxpool stem), inverted bottleneck blocks (expanding channel dimension 4× within each block), depthwise separable convolutions with large 7×7 kernels, GELU activations, and LayerNorm replacing BatchNorm. These modifications achieve ViT-level accuracy while retaining CNN advantages.

The model was pretrained on ImageNet-22K (14M images, 21,841 classes) then fine-tuned on ImageNet-1K (1.28M images, 1,000 classes). This two-stage pretraining provides exceptionally rich feature representations: the 22K pretraining exposes the model to a diverse taxonomy of visual concepts, while the 1K fine-tuning sharpens these features for common object categories.

**Why ConvNeXt over Vision Transformers?** With approximately 5,000 training images, a ViT would likely overfit severely without extensive regularization. ViTs lack the inductive biases (translation equivariance, locality) that CNNs possess, making them data-hungry. ConvNeXt achieves competitive accuracy while being more data-efficient — a crucial advantage for our small dataset. Additionally, ConvNeXt-Tiny (28.6M parameters) is more parameter-efficient than ViT-Base (86M parameters) while achieving comparable ImageNet accuracy.

Total parameters: ~28.6M. The `timm` library was used for model loading and pretrained weight initialization.

### 6.3 Progressive Fine-tuning Strategy (ConvNeXt-Tiny)

A two-phase training strategy was employed to prevent catastrophic forgetting of pretrained features:

**Phase 1 — Partial Freeze (15 epochs):**
The first 70% of parameters were frozen; only the last 30% (final ConvNeXt stages and classifier head) were trainable. This establishes a mapping from pretrained features to our 16 classes without disrupting the lower-level representations that encode universally useful visual features (edges, textures, shapes). Optimizer: Adam (lr=5×10⁻⁴), CosineAnnealingLR scheduler.

**Phase 2 — Full Fine-tuning (35 epochs):**
All parameters were unfrozen with layer-wise discriminative learning rates. The base learning rate was set to 2×10⁻⁵ (10× lower than Phase 1 to avoid disrupting learned representations), with each layer receiving: lr_p = 2×10⁻⁵ × (0.1 + 0.9 × p/L), where p is the layer position and L is total depth. This ensures early layers (highly transferable edge/texture detectors) are barely modified, while later layers (task-specific features) adapt more substantially. CosineAnnealingLR was used with T_max matching the phase duration.

Both phases used Focal Loss with sqrt-balanced class weights and the Albumentations augmentation pipeline.

### 6.4 V2 Results (Single Model)

**Validation accuracy: 76.88%** — an improvement of +24.6 percentage points over V1's EfficientNet-B0 (52.30%).

Training converged quickly: Phase 1 achieved 60% validation accuracy within 1 epoch, reaching 71.9% by epoch 8. Phase 2 continued improving to 76.9% by epoch 41 (cumulative), with the validation loss plateauing around 0.60. The model reached 99.4% training accuracy, indicating some overfitting — however, the strong augmentation pipeline and focal loss successfully limited the gap to manageable levels.

**Per-class performance:**

| Performance Tier | Classes | Recall |
|-----------------|---------|--------|
| Perfect (100%) | b, f, h, k, q | Visually distinctive or very few test samples |
| Strong (>80%) | a (90%), e (88%), g (89%), n (86%), c (82%), o (80%) | Good feature separation |
| Moderate (50-80%) | d (78%), l (63%), i (53%) | Some inter-class confusion |
| Weak (<50%) | m (40%) | Systematic confusion with 'a' and 'l' |

The most significant improvement from V1 was in the moderate-to-large classes: 'c' improved from 40% to 82%, 'g' from 72% to 89%, and 'n' from 14% to 86%. This demonstrates that the richer ImageNet-22K features provide much better discrimination for these classes.

### 6.5 V3: 5-Fold Cross-Validation Ensemble

To further reduce variance and improve robustness, V3 trains 5 independent ConvNeXt-Tiny models on different StratifiedKFold splits. Each fold uses the same training configuration: layer-wise discriminative LR (base 2×10⁻⁵), Focal Loss with sqrt-balanced weights, Albumentations pipeline, CosineAnnealingLR, and early stopping (patience=10). Each fold trains for up to 35 epochs.

**5-Fold Results:**

| Fold | Val Accuracy |
|------|-------------|
| 1 | 71.99% |
| 2 | 75.40% |
| 3 | 70.82% |
| 4 | 74.76% |
| 5 | 72.20% |
| **Mean ± Std** | **73.04% ± 1.75%** |
| **OOF Accuracy** | **73.04%** |

The individual fold accuracies (mean 73.04%) are slightly lower than V2's single-split result (76.88%) because each fold trains on only 80% of the original data. However, the key insight is that **averaging predictions across all 5 models at test time significantly outperforms any single model**. Each fold model makes different errors due to training on different data subsets, and averaging smooths out these individual errors this is the core benefit of ensemble diversity.

The low standard deviation (1.75%) across folds indicates stable performance that is not sensitive to the particular train/val split, providing confidence that the model's accuracy is robust and not an artefact of a favourable data partition.

At inference time, each of the 5 fold models produces predictions with 4× TTA (original, horizontal flip, vertical flip, 90° rotation), giving 20 total prediction sets that are averaged to form the ConvNeXt component of the final ensemble.

### 6.6 Approach 2 Summary

| Architecture | Pretraining | Parameters | Best Val Accuracy | Key Insight |
|---|---|---|---|---|
| EfficientNet-B0 | ImageNet-1K | 5.3M | 52.30% | Limited capacity + 1K pretraining insufficient |
| ConvNeXt-Tiny | ImageNet-22K+1K | 28.6M | 76.88% (single) / 73.04% OOF (5-fold) | 22K pretraining + modern architecture = +24.6% gain |

The ConvNeXt-Tiny substantially outperformed EfficientNet-B0, demonstrating that both model capacity and pretraining data scale are critical for transfer learning success on fine-grained classification tasks.

### 6.7 Hyperparameters Explored

| Parameter | Values Tried | Best | Rationale |
|-----------|-------------|------|-----------|
| Freeze ratio (Phase 1) | 0.5, 0.7, 0.8 | 0.7 | 0.5 allowed too much early modification; 0.8 restricted adaptation |
| Phase 1 LR | 1×10⁻⁴, 5×10⁻⁴, 1×10⁻³ | 5×10⁻⁴ | Trade-off between convergence speed and stability |
| Phase 2 base LR | 1×10⁻⁵, 2×10⁻⁵, 5×10⁻⁵ | 2×10⁻⁵ | Higher rates caused validation loss to increase |
| LR scaling | constant, linear, exponential | linear | Exponential was too aggressive for early layers |
| Phase 1 epochs | 10, 15, 20 | 15 | 10 was insufficient for convergence; 20 showed no additional gain |

## 7. Approach 3: Pretrained Embeddings + Classical ML (ResNet-50 vs ConvNeXt-Base)

Two pretrained architectures were used as fixed feature extractors to compare how embedding quality affects downstream classifier performance.

### 7.1 Architecture 1: ResNet-50 Embeddings (V1)

ResNet-50 (25.6M parameters, pretrained on ImageNet-1K) was the initial feature extractor in V1. With the final FC layer removed, the global average pooling output produced a 2048-dimensional feature vector per image. Using the same SVM configuration (RBF kernel, C=10.0, class_weight='balanced'), this achieved **56.33%** validation accuracy. The ResNet-50 embeddings provided a reasonable baseline but were limited by the ImageNet-1K pretraining scope (1,000 classes) and the older architecture design.

### 7.2 Architecture 2: ConvNeXt-Base Embeddings (V2/V3)

ConvNeXt-Base (88.6M parameters, pretrained on ImageNet-22K+1K) was used as a fixed feature extractor. The model was loaded with its original classifier head removed, and the global average pooling output produced a 1024-dimensional feature vector for each input image. ConvNeXt-Base was chosen over ConvNeXt-Tiny for this approach because a larger model extracts richer representations, and since only forward passes are needed (no backpropagation), the additional computational cost is minimal.

Features were standardized using StandardScaler (zero mean, unit variance per dimension) before feeding to classifiers. This is essential for distance-based methods like SVM with RBF kernel, where features at different scales would distort the distance metric.

**Important methodological note:** Only original (non-augmented) training images were used for feature extraction and classifier training. Using augmented images would have inflated the apparent diversity of minority classes without adding truly new information, potentially degrading classifier quality.

### 7.3 Classifiers and Results (ConvNeXt-Base Embeddings)

Three classical ML classifiers were evaluated:

**SVM with RBF Kernel — 59.01% accuracy (best):**
Hyperparameters: C=10.0, gamma='scale' (1/(n_features × X.var())), class_weight='balanced'. SVMs excel in high-dimensional feature spaces where the number of dimensions (1024) is comparable to the number of training samples, as the RBF kernel implicitly maps to an infinite-dimensional space where linear separation becomes easier. The 'balanced' class weight automatically adjusts weights inversely proportional to class frequencies.

**XGBoost — 51.49%:**
Configuration: 1000 estimators, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, early stopping after 50 rounds. XGBoost's relatively lower performance compared to SVM suggests that the 1024-dim embedding space benefits more from kernel-based implicit feature mapping than from decision tree-based sequential partitioning. Gradient boosting is typically more effective with structured/tabular data where feature interactions are complex but low-dimensional.

**Random Forest — 44.40%:**
Configuration: 1000 trees, max_depth=None, class_weight='balanced'. The weakest performer, likely because Random Forest's feature subsampling (√n_features ≈ 32 features per split) misses important feature combinations in the high-dimensional embedding. The useful information is distributed across many dimensions, making random subsets insufficient.

### 7.4 Approach 3 Summary

| Embedding Source | Pretraining | Feature Dim | SVM Accuracy | Key Insight |
|---|---|---|---|---|
| ResNet-50 | ImageNet-1K | 2048 | 56.33% | 1K pretraining provides reasonable but limited features |
| ConvNeXt-Base | ImageNet-22K+1K | 1024 | 59.01% | 22K pretraining gives richer features despite lower dimensionality |

The ConvNeXt-Base embeddings outperformed ResNet-50 despite producing lower-dimensional features (1024 vs 2048), demonstrating that embedding quality (driven by pretraining scale) matters more than embedding dimensionality.

### 7.5 Significance of This Approach

This approach demonstrates an important practical insight: powerful pretrained embeddings combined with simple classifiers can achieve competitive results **without any GPU-intensive fine-tuning**. The SVM's 59% accuracy required zero GPU compute for classifier training itself, only the one time forward pass for feature extraction needed GPU. For resource constrained settings (limited GPU budget, deployment on CPU only machines), this approach provides a practical alternative.

Furthermore, the embedding based approach provides a **fundamentally different "view" of the data** compared to end to end neural networks. The SVM operates on fixed, pretrained features and makes decisions based on support vectors in the embedding space, while neural networks learn features jointly with the classifier. This complementarity is valuable for ensemble construction (Section 8).

## 8. Ensemble Strategy

### 8.1 Soft Voting with Optimized Weights

The final predictions combine all three approaches via soft voting (probability averaging) with weights optimized by exhaustive grid search on the validation set. The V3 ensemble uses the 5-fold ConvNeXt-Tiny predictions (averaged across all folds and TTA) as the primary component:

| Component | V2 Weight | V3 Weight | Individual Val Accuracy |
|-----------|-----------|-----------|------------------------|
| Custom SE-Net | 0.15 | 0.00 | 52.06% |
| ConvNeXt-Tiny (5-fold) | 0.55 | 0.70 | 73.04% (OOF) |
| SVM (embeddings) | 0.30 | 0.30 | 59.01% |

In V3, the grid search assigned zero weight to the custom CNN, meaning the optimal ensemble comprises only the 5-fold ConvNeXt and SVM components. This is consistent with the principle that an ensemble member should be included only if it reduces the combined error, the custom CNN's predictions were apparently subsumed by the stronger ConvNeXt ensemble, providing no additional diversity benefit.

The SVM retains its 0.30 weight, confirming its value as a complementary signal source. The embedding-based SVM and the end to end fine tuned ConvNeXt models make genuinely different types of errors, and their combination at 70/30 produces a substantial accuracy boost (+10.5 points over the ConvNeXt-only OOF accuracy of 73.04%).

### 8.2 Test-Time Augmentation (TTA)

Both neural network models employ TTA with 4 transforms per image: original, horizontal flip, vertical flip, and 90° rotation. The predicted probabilities are averaged across all 4 transforms before applying ensemble weights. TTA effectively multiplies the inference-time "views" of each image, smoothing out orientation-dependent predictions and generally improving robustness by 1-2 percentage points. The SVM does not use TTA as it operates on a single feature vector per image.

### 8.3 Ensemble Results

**V3 Ensemble validation accuracy: 83.55%**, a substantial improvement over V2's ensemble (76.60%) and V2's best single model (76.88%). The +6.95 point gain from V2→V3 comes entirely from the 5-fold cross-validation strategy and the resulting ensemble diversity.

This result demonstrates a powerful principle: **multiple weaker models (73% each) combined intelligently can substantially outperform a single stronger model (77%)**. The 5-fold ensemble reduces prediction variance because each fold model has seen a different 80% of the data, making their errors partially independent. When combined with the SVM's orthogonal decision mechanism, the result significantly exceeds any individual component.

| Version | Ensemble Accuracy | Best Single Model | Improvement |
|---------|-------------------|-------------------|-------------|
| V1 | 60.69% | 56.33% (SVM) | +4.36% |
| V2 | 76.60% | 76.88% (ConvNeXt) | -0.28%* |
| V3 | **83.55%** | 73.04% (5-fold OOF) | **+10.51%** |

*V2's ensemble was marginally below single ConvNeXt due to limited model diversity. V3's 5-fold approach resolved this by generating genuine diversity through data partitioning.

## 9. Results Summary

### 9.1 Progression Across All Iterations

| Model | V1 | V2 | V3 |
|-------|-----|-----|-----|
| Custom CNN (SE-ResNet) | 49.16%* | 43.97% | 52.06% |
| Custom Transformer (ViT-Small) | — | — | 31.06% |
| Fine-tune: EfficientNet-B0 | 52.30% | — | — |
| Fine-tune: ConvNeXt-Tiny | — | 76.88% | 73.04% OOF (5-fold) |
| Embeddings+SVM: ResNet-50 | 56.33% | — | — |
| Embeddings+SVM: ConvNeXt-Base | — | 59.01% | 59.01% |
| **Ensemble** | **60.69%** | **76.60%** | **83.55%** |

*V1's custom CNN accuracy was inflated because validation included augmented images. V2/V3 figures use clean validation (original images only).

The progression tells a clear story: V1→V2 gained +16 points primarily through better methodology (clean validation, focal loss, ConvNeXt-22K pretraining). V2→V3 gained +7 points through better ensembling (5-fold cross-validation creating genuine model diversity).

### 9.2 V2 Per-Class Ensemble Performance (Baseline for Comparison)

| Class | Precision | Recall | F1-Score | Support | Notes |
|-------|-----------|--------|----------|---------|-------|
| a | 0.84 | 0.91 | 0.87 | 193 | Strong, largest class, well-learned |
| b | 1.00 | 1.00 | 1.00 | 1 | Perfect, but only 1 test sample |
| c | 0.78 | 0.82 | 0.80 | 71 | Good, distinctive morphology |
| d | 0.67 | 0.78 | 0.72 | 40 | Moderate, some confusion with c, e |
| e | 0.71 | 0.88 | 0.79 | 17 | Good recall but lower precision |
| f | 1.00 | 1.00 | 1.00 | 9 | Perfect, highly distinctive |
| g | 0.83 | 0.89 | 0.86 | 108 | Strong, second-largest class |
| h | 1.00 | 1.00 | 1.00 | 1 | Perfect, but only 1 test sample |
| i | 0.68 | 0.50 | 0.58 | 30 | Weak, confused with multiple classes |
| k | 1.00 | 1.00 | 1.00 | 2 | Perfect, but only 2 test samples |
| l | 0.71 | 0.63 | 0.67 | 99 | Moderate, confused with a, m |
| m | 0.48 | 0.38 | 0.43 | 81 | Weakest, systematic confusion |
| n | 0.86 | 0.86 | 0.86 | 14 | Strong, good discrimination |
| o | 0.81 | 0.73 | 0.77 | 30 | Good, improved from V1 |
| p | 0.71 | 1.00 | 0.83 | 5 | Good recall, some false positives |
| q | 1.00 | 1.00 | 1.00 | 4 | Perfect, distinctive class |
| **Macro avg** | **0.82** | **0.84** | **0.82** | **705** | |
| **Weighted avg** | **0.76** | **0.77** | **0.76** | **705** | |

## 10. Error Analysis

### 10.1 Confusion Patterns

Analysis of the confusion matrices reveals systematic misclassification patterns:

**The a/l/m cluster:** These three classes form a confusion cluster. Class 'm' loses 17 samples to 'a' and 6 to 'l'; class 'l' loses 13 to 'a' and 7 to 'g'. These classes share similar cell confluency levels and morphological texture, making them the hardest discrimination task in the dataset. The asymmetry of confusion (m→a but rarely a→m) is consistent with 'a' being the majority class and the model's natural tendency to default to more frequent predictions when uncertain.

**Class 'i' confusion:** With only 50% recall, class 'i' is frequently misclassified as 'g' and 'o'. The shared features appear to be related to cell spacing and colony morphology.

### 10.2 Impact of Class Frequency on Performance

There is a clear correlation between training set size and model performance, but it is not perfectly linear. Classes with very few samples but highly distinctive visual features (f, q) achieve near-perfect accuracy, while classes with moderate sample sizes but high visual similarity to other classes (m with 543 samples, l with 660) perform relatively poorly. This suggests that **visual distinctiveness is a stronger predictor of classification success than sample size** beyond a minimum threshold.

### 10.3 Train-Validation Gap

The ConvNeXt model showed a notable train-val gap (99.4% train vs 76.9% val). While this indicates some overfitting, the strong augmentation pipeline and focal loss prevented the gap from being catastrophic. The custom CNN showed a smaller relative gap (63.1% vs 44.0%) because it had less capacity to memorize training data. The SVM, being a non-parametric method, is less susceptible to overfitting by design.

## 11. Discussion

### 11.1 Key Findings

**ImageNet-22K pretraining was the single most impactful factor.** The 22K pretraining exposed ConvNeXt to 21,841 visual categories during pretraining versus only 1,000 for ImageNet-1K models. This broader exposure provided richer, more transferable features; the V1-to-V2 improvement of +24.6% for the fine-tuned model is almost entirely attributable to this change (EfficientNet-B0/1K → ConvNeXt-Tiny/22K).

**Focal Loss proved critical for handling class imbalance.** By dynamically down-weighting easy examples, focal loss naturally focused training on the harder minority classes and difficult boundary cases without the over-correction problems caused by aggressive inverse-frequency weighting. The combination of focal loss with sqrt-balanced weights provided a measured approach that improved minority class performance without sacrificing majority class accuracy.

**Clean validation methodology revealed the true model performance.** V1's augmented validation produced misleading results (b, h, k at 100%) that masked fundamental model weaknesses. V2's clean split gave honest estimates that enabled meaningful model comparisons and iterative improvement. This experience reinforces the importance of rigorous evaluation in machine learning, especially with augmented data.

**Training from scratch confirmed the data bottleneck.** The custom SE-Net's plateau at 44% validates the well known principle that CNNs require large datasets to learn robust features from scratch. With ~4,700 original images, there simply isn't enough diversity to learn the edge, texture, and shape detectors that pretrained models acquire from millions of images.

### 11.2 Design Decisions and Trade-offs

**Offline vs. online augmentation:** The preprocessing pipeline applies augmentation offline (before training) for minority classes, while the training loop applies augmentation online (during training) for all classes. This two-stage approach ensures minority classes have a minimum sample count for stable gradient estimation, while online augmentation provides stochastic variation every epoch for all classes.

**Square-root vs. inverse-frequency weighting:** Pure inverse-frequency weights (w_c ∝ 1/N_c) would give class 'b' (4 samples) a weight 321× that of class 'a' (1,283 samples). Square-root weighting (w_c ∝ 1/√N_c) reduces this to 18×, providing a much gentler correction that doesn't overwhelm the loss signal from majority classes.

**Ensemble composition:** V2 included the weaker custom CNN (0.15 weight), hypothesizing that even weak learners improve the collective through diversity. V3's grid search assigned it zero weight, revealing that the custom CNN's predictions were fully subsumed by the 5-fold ConvNeXt ensemble. This is a useful negative result: diversity must be **orthogonal** (making genuinely different errors) rather than merely different in magnitude. The SVM, operating on fixed embeddings through a fundamentally different mechanism, provides this orthogonality; hence its consistent 0.30 weight across V2 and V3.

### 11.3 V3: Impact of 5-Fold Ensembling

The V3 iteration implemented 5-fold cross-validation ensembling, which was identified as a potential improvement during V2 analysis. The results exceeded expectations: the ensemble accuracy jumped from 76.60% to **83.55%**, a +6.95 point gain from ensembling alone. Notably, each individual fold model (mean 73.04%) was weaker than V2's single model (76.88%), yet the 5-fold combination substantially outperformed it. This validates the ensemble diversity hypothesis; models trained on different data subsets capture complementary aspects of the data distribution.

The 5-fold ConvNeXt + SVM combination (70/30 weights) proved optimal, with the custom CNN receiving zero weight. This suggests that the SVM's embedding-based decisions provide orthogonal information to the ConvNeXt ensemble, while the custom CNN's predictions are fully subsumed by the stronger models.

### 11.4 Remaining Potential Improvements

Several strategies could potentially push accuracy even further:

**Hierarchical classification:** Grouping visually similar classes (a/l/m and g/i/o) into superclasses and training a two-stage classifier; first predicting the group, then the specific class within each group could improve discrimination where flat classification fails.

**Larger backbone models:** ConvNeXt-Base or ConvNeXt-Large for fine-tuning (rather than just for embeddings) could provide additional capacity, though overfitting risk increases.

**Mixup and CutMix augmentation:** These interpolation based augmentations generate training samples between classes, smoothing decision boundaries and improving generalization on ambiguous cases.

## 12. Conclusion

This assignment explored three fundamentally different approaches to image classification on a challenging, real-world cell culture dataset with extreme class imbalance. Each approach compared two different architectures, progressively refined across three experimental iterations:

1. **Training from scratch:** SE-ResNet (CNN, 52.06%) vs ViT-Small (Transformer, 31.06%). The CNN's 21-point advantage confirms that its strong inductive biases (locality, translation equivariance) are decisive on small datasets, and that sufficient data is a prerequisite for attention-based architectures to learn useful spatial representations.

2. **Fine-tuning pretrained models:** EfficientNet-B0 (ImageNet-1K, 52.30%) vs ConvNeXt-Tiny (ImageNet-22K, 76.88% single / 73.04% OOF 5-fold). The +24.6% improvement demonstrates that both model architecture (modern ConvNeXt design) and pretraining scale (22K vs 1K classes) are critical factors for transfer learning success.

3. **Embedding-based ML:** ResNet-50 embeddings (2048-dim, SVM 56.33%) vs ConvNeXt-Base embeddings (1024-dim, SVM 59.01%). Richer pretrained features improved downstream classifier performance even with lower-dimensional embeddings, and proved indispensable as a complementary ensemble signal (+10.5 point boost).

4. **Weighted ensemble (83.55%)** leveraged model diversity through optimized combination weights, substantially outperforming any individual model. The V1→V2→V3 progression (60.69% → 76.60% → 83.55%) demonstrates the cumulative impact of methodological rigour, architecture selection, and ensemble strategy.

Three key insights emerged. First, **architecture choice matters within each approach**: CNN vs Transformer, EfficientNet vs ConvNeXt, ResNet-50 vs ConvNeXt-Base — in every case, the architecture with stronger inductive biases or richer pretraining won decisively. Second, **clean evaluation methodology is foundational**: V1's augmented validation leakage masked real performance issues. Third, **ensemble diversity matters more than individual model strength**: V3's 5-fold models were individually weaker (73%) than V2's single model (77%), yet their combination dramatically outperformed it (83.55%).

## References

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
- Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. CVPR 2018.
- Lin, T.Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. ICCV 2017.
- Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. CVPR 2022.
- Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR 2019.
- Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.

## Appendix: Technical Details

- **Framework:** PyTorch 2.x with `timm` library for pretrained model loading
- **Image size:** 224×224×3 (preprocessed from original 1024×1024)
- **GPU:** NVIDIA Tesla T4 (Kaggle, 16GB VRAM)
- **Reproducibility seed:** 42 (set for Python, NumPy, PyTorch, and CUDA)
- **Validation split:** 15% stratified holdout (original images only, augmented excluded)
- **Training time:** Custom CNN ~45 min, ConvNeXt ~50 min (15 Phase 1 + 35 Phase 2), Embeddings+ML ~15 min
- **Total trainable parameters:** SE-ResNet 21.5M, ConvNeXt-Tiny 28.6M, ConvNeXt-Base 88.6M (frozen, feature extraction only)
- **Batch size:** 32 (SE-ResNet with accumulation=4), 16 (ConvNeXt)
- **Mixed precision:** Enabled (AMP + GradScaler) for both neural network models
