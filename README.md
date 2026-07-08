# Stall Detection with a 1D Convolutional Neural Network

> **University project** — a graded individual assignment from my studies at TU Berlin. This repository is published **for demonstration purposes only**. See [Disclaimer & License](#disclaimer--license) — no permission is granted to use or copy any of its contents.

## Task

Binary classification of univariate sensor time series: each sample is a signal of 1,400 time steps, labeled either **stable** (0) or **stall** (1).

- 548 labeled samples, with only ~17% belonging to the *stall* class (strong class imbalance)
- Evaluation metric: **F1 score**
- In addition to the trained model, the assignment required reporting an **estimated F1 score with an uncertainty margin** for unseen data, plus the chosen decision threshold — the submission was then graded against a hidden hold-out set

## Approach

1. **Data exploration** — dimensions, class distribution, visual comparison of stable vs. stall signals
2. **Stratified 5-fold cross-validation** — within each fold, an inner 80/20 train/validation split is used for early stopping and threshold selection; the outer fold serves as an untouched test set
3. **Class imbalance handling** — class weights computed per training fold
4. **Regularization / training control** — `EarlyStopping` (restoring best weights) and `ReduceLROnPlateau` callbacks, Dropout, BatchNormalization
5. **Decision-threshold optimization** — thresholds from 0.01 to 0.99 are swept on validation predictions to maximize F1, in two variants:
   - [`cross_validation.py`](src/cross_validation.py): a threshold per fold, with the median reported
   - [`cross_validation_pooled_threshold.py`](src/cross_validation_pooled_threshold.py): one global threshold chosen from the pooled out-of-fold validation predictions, then evaluated on the outer test folds
6. **Stability check** — re-runs with different random seeds
7. **Final model** — trained on all data with an 80/20 train/validation split ([`train_final_model.py`](src/train_final_model.py))
8. **Project documentation** as a 1 min overview video [`presentation/project_presentation.mp4`](presentation/project_presentation.mp4) 

To avoid data leakage, the `Normalization` layer is adapted **only on the respective training split** in every setting.

## Model architecture

A compact 1D CNN (Keras / TensorFlow):

```
Input (1400, 1)
→ Normalization (adapted on training data only)
→ Conv1D(16, kernel 7) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
→ Conv1D(32, kernel 5) → BatchNorm → ReLU → MaxPool(2) → Dropout(0.2)
→ Conv1D(64, kernel 3) → BatchNorm → ReLU
→ GlobalAveragePooling1D
→ Dense(32, ReLU) → Dropout(0.3)
→ Dense(1, sigmoid)
```

## Results

| Metric | Value |
|---|---|
| Estimated F1 on unseen data (cross-validation) | **0.90 ± 0.10** |
| Chosen decision threshold | **0.51** |
| F1 on the hidden hold-out set (grading) | **1.00** (no misclassifications) |

The hold-out result lies inside the reported interval; the margin was chosen conservatively because of the F1 variance across folds (± 0.06) and the high variance of the fold-optimal thresholds (± 0.3 around a mean of 0.51).

Identified possible improvements (see presentation): a smaller architecture reaching the same F1 with fewer parameters, and `RepeatedStratifiedKFold` for a more robust performance estimate.

## Repository contents

| Path | Description |
|---|---|
| [`src/data_overview.py`](src/data_overview.py) | Data exploration: shapes, class distribution, example signal plots |
| [`src/data_preprocessing.py`](src/data_preprocessing.py) | Input shape preparation and naive baseline |
| [`src/cross_validation.py`](src/cross_validation.py) | Stratified 5-fold CV with per-fold threshold optimization |
| [`src/cross_validation_pooled_threshold.py`](src/cross_validation_pooled_threshold.py) | Stratified 5-fold CV with one global threshold from pooled out-of-fold predictions |
| [`src/train_final_model.py`](src/train_final_model.py) | Trains the final model on all data and saves it |
| [`presentation/project_presentation.mp4`](presentation/project_presentation.mp4) | Short video presentation of approach and results |
| [`requirements.txt`](requirements.txt) | Python dependencies (Python 3.12, TensorFlow 2.19, Keras 3.11) |

## Dataset — intentionally not included

The dataset was provided by the instructors and is **not part of this repository** and will not be shared. The scripts are therefore not runnable as-is.

## Disclaimer & License

- This repository documents **my own coursework** from my studies at TU Berlin. It is not affiliated with or endorsed by TU Berlin.
- **All rights reserved.** No permission is granted to use, copy, modify, distribute, or build upon any content of this repository — see [LICENSE](LICENSE).
- In particular, submitting this work (in whole or in part) as your own in any academic context is prohibited and constitutes plagiarism.
