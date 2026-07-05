import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split

seed = 21031998
np.random.seed(seed)
keras.utils.set_random_seed(seed)

# callbacks
callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True), # changed to patience=8, cause of slight overfitting with CV patience 20
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=8, factor=0.5, min_lr=1e-5),
    ]

data = np.load("data/student_task_adl_prepared_dataset.npz")
X = data["X"].astype("float32")  # (548, 1400)
y = data["y"].astype("int32")    # (548,)

X = X[..., np.newaxis]           # (548, 1400, 1)

X_train_final, X_validation_final, y_train_final, y_validation_final = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=seed
)

# compute class_weight on X_train_final
n0 = int(np.sum(y_train_final == 0))
n1 = int(np.sum(y_train_final == 1))
class_weight = {0: 1.0, 1: n0 / max(n1, 1)}

final_model = keras.Sequential([
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

# adapt normalization on training only
for layer in final_model.layers:
    if isinstance(layer, keras.layers.Normalization):
        layer.adapt(X_train_final)
        break

final_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="binary_crossentropy",
)

final_model.fit(
    X_train_final, y_train_final,
    validation_data=(X_validation_final, y_validation_final),
    epochs=300, batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)
final_model.save("franzwendel_model.keras")