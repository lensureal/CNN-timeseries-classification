import numpy as np
import keras
import tensorflow as tf
from keras import layers, callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score

# reproducibility
seed = 12223432
np.random.seed(seed)
keras.utils.set_random_seed(seed)

# load data
data = np.load("data/student_task_adl_prepared_dataset.npz")
X = data["X"].astype("float32") # (548,1400)
y = data["y"].astype("int32")   # (548,)

X = X[..., np.newaxis]  # (548,1400,1) add dimension for 1D CNN
print(X.shape, y.shape)

baseline = y.mean()
print(f"Random baseline (to beat): {baseline}")

# Stratified KFold CV
n_splits = 5

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

fold_f1_scores = []

# define thresholds to test
thresholds = np.linspace(0.01, 0.99, 99)

# treshholds
best_thresholds = []


for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{n_splits}")

    # callbacks
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=8, factor=0.5, min_lr=1e-5, verbose=1),
    ]

    # define test data
    X_test_fold = X[test_index]
    y_test_fold = y[test_index]
    
    # split train_full into train and validation
    X_train_full_fold = X[train_index]
    y_train_full_fold = y[train_index]

    X_train_fold, X_validation_fold, y_train_fold, y_validation_fold = train_test_split(
    X_train_full_fold,
    y_train_full_fold,
    test_size=0.20,
    shuffle=True,
    stratify=y_train_full_fold,
    random_state=seed + fold)

    # class weights

    n0 = int(np.sum(y_train_fold == 0))
    n1 = int(np.sum(y_train_fold == 1))
    class_weight = {0: 1.0, 1: n0 / max(n1,1)}
    print(f"Class weights: {class_weight}")

    # model

    model = keras.Sequential([
        layers.Input(shape=(1400, 1)),

        layers.Normalization(),

        # first Conv block

        layers.Conv1D(16, 7, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        # second Conv block

        layers.Conv1D(32, 5, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        # third Conv block

        layers.Conv1D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(
            1,
            activation="sigmoid",
            name="franzDOTwendelATcampusDOTtuMINUSberlinDOTde"
        ),
    ])
    
    # normalization on training fold to prevent data leakage
    for layer in model.layers:
        if isinstance(layer, keras.layers.Normalization):
            layer.adapt(X_train_fold)
            break
        
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
    )

    model.fit(
        X_train_fold,
        y_train_fold,
        validation_data=(X_validation_fold, y_validation_fold),
        epochs=300,
        batch_size=32,
        class_weight=class_weight,
        callbacks=cb,
        verbose=1
    )

    # predict on validation and choose fold-specific threshold on validation
    p_val_fold = model.predict(X_validation_fold, verbose=0).reshape(-1)

    best_t_fold = 0.5
    best_f1_val = 0
    for t in thresholds:
        y_hat_val = (p_val_fold >= t).astype("int32")
        f1_val = f1_score(y_validation_fold, y_hat_val)
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            best_t_fold = float(t)

    best_thresholds.append(best_t_fold)

    # evaluate probability on test data
    p_test_fold = model.predict(X_test_fold, verbose=0).reshape(-1)
    
    # test fold F1 at the chosen (validation) threshold
    y_hat_test = (p_test_fold >= best_t_fold).astype("int32")
    f1_test = f1_score(y_test_fold, y_hat_test)
    fold_f1_scores.append(f1_test)


# calculate single treshhold across folds
best_threshold = float(np.median(best_thresholds))

# report results
f1_mean = float(np.mean(fold_f1_scores))
f1_std  = float(np.std(fold_f1_scores))
print(f"Reported threshold (median of folds): {best_threshold:.3f}")
print(f"Estimated F1 (test across folds): mean={f1_mean:.4f}, std={f1_std:.4f}")