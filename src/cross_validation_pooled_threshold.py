import numpy as np
import keras
from keras import layers
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score

# reproducibility
seed = 111322422
np.random.seed(seed)
keras.utils.set_random_seed(seed)

# load data
data = np.load("data/student_task_adl_prepared_dataset.npz")
X = data["X"].astype("float32")  # (548, 1400)
y = data["y"].astype("int32")    # (548,)
X = X[..., np.newaxis]           # (548, 1400, 1)

print(X.shape, y.shape)
print(f"Random baseline (to beat): {y.mean():.4f}")

# CV setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

thresholds = np.linspace(0.01, 0.99, 99)

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(1400, 1)),
        layers.Normalization(),

        layers.Conv1D(16, 7, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(32, 5, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(
            1, activation="sigmoid",
            name="franzDOTwendelATcampusDOTtuMINUSberlinDOTde"
        ),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
    )
    return model

def compute_class_weight(y_train):
    n0 = int(np.sum(y_train == 0))
    n1 = int(np.sum(y_train == 1))
    return {0: 1.0, 1: n0 / max(n1, 1)}

def adapt_normalization(model, X_adapt):
    for layer in model.layers:
        if isinstance(layer, keras.layers.Normalization):
            layer.adapt(X_adapt)
            return

# --- storage for pooled validation predictions and fold test predictions
oof_val_probs = []
oof_val_true = []

fold_test_probs = []
fold_test_true = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
    print(f"\nFold {fold}/{n_splits}")

    # split outer fold
    X_test_fold = X[test_index]
    y_test_fold = y[test_index]
    X_train_full = X[train_index]
    y_train_full = y[train_index]

    # inner split for early stopping + threshold selection
    X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
        X_train_full, y_train_full,
        test_size=0.20,
        shuffle=True,
        stratify=y_train_full,
        random_state=seed + fold
    )

    class_weight = compute_class_weight(y_train_fold)
    print(f"Class weights: {class_weight}")

    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                     restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=8,
                                         factor=0.5, min_lr=1e-5, verbose=0),
    ]

    model = build_model()

    # adapt normalization on inner training
    adapt_normalization(model, X_train_fold)

    model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=300,
        batch_size=32,
        class_weight=class_weight,
        callbacks=cb,
        verbose=0
    )

    # collect pooled (out-of-fold) validation preds
    p_val = model.predict(X_val_fold, verbose=0).reshape(-1)
    oof_val_probs.append(p_val)
    oof_val_true.append(y_val_fold)

    # collect outer test preds (evaluation happens AFTER we choose global threshold)
    p_test = model.predict(X_test_fold, verbose=0).reshape(-1)
    fold_test_probs.append(p_test)
    fold_test_true.append(y_test_fold)

# ---- 1) pick ONE global threshold using pooled validation predictions
p_val_all = np.concatenate(oof_val_probs)
y_val_all = np.concatenate(oof_val_true)

best_threshold = 0.5
best_f1_val = -1.0
for t in thresholds:
    f1v = f1_score(y_val_all, (p_val_all >= t).astype("int32"))
    if f1v > best_f1_val:
        best_f1_val = f1v
        best_threshold = float(t)

print(f"\nChosen global threshold (from pooled CV validation): {best_threshold:.3f}")
print(f"OOF-val F1 at that threshold (optimistic): {best_f1_val:.4f}")

# ---- 2) estimate unseen-data F1: evaluate each outer test fold using THAT ONE threshold
fold_f1_scores = []
for p_test, y_test in zip(fold_test_probs, fold_test_true):
    y_hat = (p_test >= best_threshold).astype("int32")
    fold_f1_scores.append(f1_score(y_test, y_hat))

f1_mean = float(np.mean(fold_f1_scores))
f1_std = float(np.std(fold_f1_scores, ddof=1))  # sample std
print(f"Estimated F1 on unseen data (outer test): mean={f1_mean:.4f}, std={f1_std:.4f}")


print("\nFinal deployment rule:")
print(f"predict positive if p >= {best_threshold:.3f}")
