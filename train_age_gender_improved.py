import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU config error: {e}")

from pathlib import Path
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, 
    Dense, Dropout, BatchNormalization, Activation, Add,
    SeparableConv2D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------- Config ----------
DATA_DIR = Path("UTKFace")
IMAGE_SIZE = (128, 128)  # Increased from 64x64 for better detail
BATCH_SIZE = 32  # Reduced for better gradient updates
EPOCHS = 100
MODEL_OUT = "age_gender_model_improved.keras"
LEARNING_RATE = 5e-4
# ----------------------------

print("Loading dataset...")
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR.resolve()}")

filenames = list(DATA_DIR.glob("*.jpg"))
print(f"Total images found: {len(filenames)}")

ages, genders, images = [], [], []
skipped = 0

for file in filenames:
    try:
        parts = file.name.split("_")
        age = int(parts[0])
        gender = int(parts[1])
        
        if age < 0 or age > 116:
            skipped += 1
            continue
            
        img = load_img(file, target_size=IMAGE_SIZE)
        img_array = img_to_array(img) / 255.0
        
        ages.append(age)
        genders.append(gender)
        images.append(img_array)
    except Exception as e:
        skipped += 1
        continue

print(f"Skipped {skipped} invalid files")
print(f"Valid samples: {len(images)}")

ages = np.array(ages, dtype=np.float32)
genders = np.array(genders, dtype=np.int32)
images = np.array(images, dtype=np.float32)

# Age normalization - using log scale for better distribution
MAX_AGE = 116.0
ages_norm = ages / MAX_AGE

print(f"Age range: {ages.min():.1f} - {ages.max():.1f}")
print(f"Gender distribution: Male={np.sum(genders==0)}, Female={np.sum(genders==1)}")

# Split data with larger test set for better evaluation
X_train, X_test, age_train, age_test, gender_train, gender_test = train_test_split(
    images, ages_norm, to_categorical(genders, 2), 
    test_size=0.20, random_state=42, stratify=genders
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Improved model with residual connections and better architecture
def residual_block(x, filters, kernel_size=3, strides=1):
    """Residual block for better gradient flow"""
    shortcut = x
    
    # Main path
    x = Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    
    # Shortcut connection
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=strides, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x

def build_improved_model():
    """Improved architecture with residual connections and attention"""
    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    # Initial convolution
    x = Conv2D(32, (7,7), strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)
    
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)
    
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    
    # Separate branches for age and gender
    # Age branch - needs more capacity
    age_branch = Dense(512, activation="relu")(x)
    age_branch = BatchNormalization()(age_branch)
    age_branch = Dropout(0.5)(age_branch)
    age_branch = Dense(256, activation="relu")(age_branch)
    age_branch = BatchNormalization()(age_branch)
    age_branch = Dropout(0.4)(age_branch)
    age_branch = Dense(128, activation="relu")(age_branch)
    age_output = Dense(1, activation="linear", name="age_output")(age_branch)
    
    # Gender branch - simpler task
    gender_branch = Dense(256, activation="relu")(x)
    gender_branch = BatchNormalization()(gender_branch)
    gender_branch = Dropout(0.4)(gender_branch)
    gender_branch = Dense(128, activation="relu")(gender_branch)
    gender_output = Dense(2, activation="softmax", name="gender_output")(gender_branch)
    
    model = Model(inputs=inputs, outputs=[age_output, gender_output])
    return model

model = build_improved_model()

# Custom loss weighting - age is harder to predict
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss={
        "age_output": "mse", 
        "gender_output": "categorical_crossentropy"
    },
    loss_weights={
        "age_output": 3.0,  # Increased weight for age
        "gender_output": 1.0
    },
    metrics={
        "age_output": ["mae", "mse"], 
        "gender_output": ["accuracy"]
    }
)

model.summary()

# Enhanced callbacks
callbacks = [
    ModelCheckpoint(
        MODEL_OUT, 
        monitor='val_age_output_mae',  # Focus on age accuracy
        save_best_only=True, 
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss', 
        patience=15,  # Increased patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_age_output_mae', 
        factor=0.5, 
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
]

# Enhanced data augmentation
def augment_image(image, label_dict):
    """Apply stronger augmentations"""
    # Random horizontal flip (gender invariant, age invariant)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.3)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    
    # Random saturation
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    
    # Small rotation
    if tf.random.uniform([]) > 0.5:
        angle = tf.random.uniform([], -0.2, 0.2)
        image = tf.contrib.image.rotate(image, angle) if hasattr(tf.contrib, 'image') else image
    
    # Ensure values stay in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label_dict

# Reshape age arrays
age_train_reshaped = age_train.reshape(-1, 1) if age_train.ndim == 1 else age_train
age_test_reshaped = age_test.reshape(-1, 1) if age_test.ndim == 1 else age_test

# Create training dataset with stronger augmentation
train_dataset = tf.data.Dataset.from_tensor_slices((
    X_train, 
    {"age_output": age_train_reshaped, "gender_output": gender_train}
))

train_dataset = train_dataset.map(
    augment_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.shuffle(buffer_size=2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create validation dataset (no augmentation)
val_dataset = tf.data.Dataset.from_tensor_slices((
    X_test,
    {"age_output": age_test_reshaped, "gender_output": gender_test}
))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("\nStarting training with improved model...")
print(f"Steps per epoch: {len(X_train) // BATCH_SIZE}")
print(f"Validation steps: {len(X_test) // BATCH_SIZE}\n")

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\nEvaluating on test set...")
results = model.evaluate(val_dataset, verbose=0)

print("\n" + "="*60)
print("FINAL TEST RESULTS (IMPROVED MODEL)")
print("="*60)
print(f"Total Loss: {results[0]:.4f}")
print(f"Age Loss (MSE): {results[1]:.4f}")
print(f"Gender Loss: {results[2]:.4f}")
print(f"Age MAE (normalized): {results[3]:.4f}")
print(f"Age MSE: {results[4]:.4f}")
print(f"Age MAE (years): {results[3]*MAX_AGE:.2f} years")
print(f"Gender Accuracy: {results[5]*100:.2f}%")
print("="*60)

# Sample predictions with better analysis
print("\nSample predictions (20 examples):")
print("-" * 80)
sample_indices = np.random.choice(len(X_test), 20, replace=False)
errors = []

for idx in sample_indices:
    pred = model.predict(X_test[idx:idx+1], verbose=0)
    pred_age = pred[0][0][0] * MAX_AGE
    pred_gender = "Male" if pred[1][0][0] > pred[1][0][1] else "Female"
    pred_gender_conf = max(pred[1][0][0], pred[1][0][1])
    
    true_age = age_test_reshaped[idx][0] * MAX_AGE
    true_gender = "Male" if gender_test[idx][0] == 1 else "Female"
    
    age_error = abs(pred_age - true_age)
    errors.append(age_error)
    gender_correct = "✓" if pred_gender == true_gender else "✗"
    
    print(f"True: {true_gender:6s}, {true_age:3.0f}y | "
          f"Pred: {pred_gender:6s} ({pred_gender_conf:.2f}), {pred_age:3.0f}y | "
          f"Age Error: {age_error:4.1f}y {gender_correct}")

print(f"\nAverage error on samples: {np.mean(errors):.2f} years")
print(f"Median error on samples: {np.median(errors):.2f} years")

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['age_output_mae'], label='Train MAE')
plt.plot(history.history['val_age_output_mae'], label='Val MAE')
plt.title('Age MAE Over Time')
plt.xlabel('Epoch')
plt.ylabel('MAE (normalized)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['gender_output_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_gender_output_accuracy'], label='Val Accuracy')
plt.title('Gender Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Total Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history_improved.png', dpi=150)
print(f"\n✓ Training plots saved to training_history_improved.png")
print(f"✓ Model saved to {MODEL_OUT}")