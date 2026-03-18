# train_emotion_keras.py  (replace your existing file with this)
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils

# --- Config ---
IMG_SIDE = 48
CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 50
RANDOM_STATE = 42
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load data ---
print("Loading processed_data/X.npy and processed_data/y.npy ...")
X = np.load("processed_data/X.npy", allow_pickle=False)
y = np.load("processed_data/y.npy", allow_pickle=False)
print("Loaded shapes:", X.shape, y.shape)

# --- Normalize & reshape robustly ---
# If X is already shaped (n, H, W) or (n, H, W, C) handle those. If it's flat (n, 2304) reshape.
n = X.shape[0]
rest = X.shape[1:]

# If flat (n, 2304) or (n, 2304, 1), collapse remaining dims to pixels
pixels = int(np.prod(rest))
expected_pixels = IMG_SIDE * IMG_SIDE

if pixels != expected_pixels:
    raise ValueError(f"Unexpected pixel count per sample: {pixels}. "
                     f"Expected {expected_pixels} (for {IMG_SIDE}x{IMG_SIDE}).")

# reshape to (n, 48, 48)
if X.ndim == 2:
    X = X.reshape((n, IMG_SIDE, IMG_SIDE))
elif X.ndim == 3 and X.shape[1] == IMG_SIDE and X.shape[2] == IMG_SIDE:
    # already (n,48,48)
    pass
elif X.ndim == 3 and X.shape[1] * X.shape[2] == pixels:
    X = X.reshape((n, IMG_SIDE, IMG_SIDE))
elif X.ndim == 4 and X.shape[1] == IMG_SIDE and X.shape[2] == IMG_SIDE:
    # already (n,48,48,C)
    pass
else:
    # fallback: flatten remaining dims then reshape
    X = X.reshape((n, pixels)).reshape((n, IMG_SIDE, IMG_SIDE))

# ensure channel dim
if X.ndim == 3:
    X = np.expand_dims(X, -1)  # (n,48,48,1)

# cast and scale to 0..1
X = X.astype("float32")
if X.max() > 1.0:
    X /= 255.0

print("Final X shape for model:", X.shape)

# --- Train/val/test split ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)
print("Splits -> train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

# --- Class weights to handle imbalance ---
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced",
                                     classes=classes,
                                     y=y_train)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)

# --- Simple CNN model ---
num_classes = len(np.unique(y))
def make_model(input_shape=(IMG_SIDE, IMG_SIDE, CHANNELS), num_classes=num_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    return model

model = make_model(input_shape=X_train.shape[1:])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# --- Callbacks ---
best_path = os.path.join(MODEL_DIR, "best_emotion_model.h5")
final_path = os.path.join(MODEL_DIR, "final_emotion_model.h5")
cb = [
    callbacks.ModelCheckpoint(best_path, save_best_only=True, monitor="val_loss"),
    callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
]

# --- Fit ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=cb,
    verbose=2
)

# --- Save final model & history ---
model.save(final_path)
np.save("history.npy", history.history)
print("Saved best model to", best_path)
print("Saved final model to", final_path)
print("Saved history.npy")

# --- Evaluate on test set ---
print("Evaluating on test set...")
res = model.evaluate(X_test, y_test, verbose=2)
print("Test loss, Test acc:", res)
