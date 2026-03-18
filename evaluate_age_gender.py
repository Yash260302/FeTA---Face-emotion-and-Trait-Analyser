import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Configuration
DATA_DIR = Path("UTKFace")
MODEL_PATH = "age_gender_model_improved.keras"  # Change to your model
IMAGE_SIZE = (128, 128)  # Must match training
MAX_AGE = 116.0

print("="*70)
print("AGE & GENDER MODEL EVALUATION")
print("="*70)

# Load test data
print("\nLoading test dataset...")
filenames = list(DATA_DIR.glob("*.jpg"))
print(f"Total images found: {len(filenames)}")

ages, genders, images = [], [], []
for file in filenames[:1000]:  # Limit to 1000 for faster evaluation
    try:
        parts = file.name.split("_")
        age = int(parts[0])
        gender = int(parts[1])
        
        if age < 0 or age > 116:
            continue
            
        img = load_img(file, target_size=IMAGE_SIZE)
        img_array = img_to_array(img) / 255.0
        
        ages.append(age)
        genders.append(gender)
        images.append(img_array)
    except:
        continue

ages = np.array(ages, dtype=np.float32)
genders = np.array(genders, dtype=np.int32)
images = np.array(images, dtype=np.float32)

ages_norm = ages / MAX_AGE

# Split data (using same random state as training)
_, X_test, _, age_test, _, gender_test = train_test_split(
    images, ages_norm, to_categorical(genders, 2), 
    test_size=0.20, random_state=42, stratify=genders
)

age_test_raw = age_test * MAX_AGE  # Denormalized ages

print(f"Test samples: {len(X_test)}")

# Load model
print(f"\nLoading model from {MODEL_PATH}...")
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(X_test, batch_size=32, verbose=1)

if isinstance(predictions, (list, tuple)):
    age_pred, gender_pred = predictions
else:
    age_pred = predictions[:, 0:1]
    gender_pred = predictions[:, 1:]

# Denormalize age predictions
age_pred_years = age_pred.ravel() * MAX_AGE
age_pred_years = np.clip(age_pred_years, 0, 116)

# Gender predictions
gender_pred_classes = np.argmax(gender_pred, axis=1)
gender_true_classes = np.argmax(gender_test, axis=1)

# ============= AGE EVALUATION =============
print("\n" + "="*70)
print("AGE PREDICTION RESULTS")
print("="*70)

age_mae = mean_absolute_error(age_test_raw, age_pred_years)
age_rmse = np.sqrt(np.mean((age_test_raw - age_pred_years) ** 2))
age_errors = np.abs(age_test_raw - age_pred_years)

print(f"\nMean Absolute Error (MAE): {age_mae:.2f} years")
print(f"Root Mean Squared Error (RMSE): {age_rmse:.2f} years")
print(f"Median Error: {np.median(age_errors):.2f} years")
print(f"Max Error: {np.max(age_errors):.2f} years")
print(f"Min Error: {np.min(age_errors):.2f} years")

# Age error distribution
print(f"\nError Distribution:")
print(f"  Within 5 years:  {np.sum(age_errors <= 5) / len(age_errors) * 100:.1f}%")
print(f"  Within 10 years: {np.sum(age_errors <= 10) / len(age_errors) * 100:.1f}%")
print(f"  Within 15 years: {np.sum(age_errors <= 15) / len(age_errors) * 100:.1f}%")

# Age group analysis
age_groups = [(0, 18, "Children"), (18, 30, "Young Adults"), 
              (30, 50, "Adults"), (50, 116, "Seniors")]

print(f"\nPerformance by Age Group:")
for start, end, label in age_groups:
    mask = (age_test_raw >= start) & (age_test_raw < end)
    if np.sum(mask) > 0:
        group_mae = mean_absolute_error(age_test_raw[mask], age_pred_years[mask])
        print(f"  {label:15s} ({start:3d}-{end:3d}): MAE = {group_mae:.2f} years (n={np.sum(mask)})")

# ============= GENDER EVALUATION =============
print("\n" + "="*70)
print("GENDER PREDICTION RESULTS")
print("="*70)

gender_accuracy = np.mean(gender_pred_classes == gender_true_classes)
print(f"\nOverall Accuracy: {gender_accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(gender_true_classes, gender_pred_classes)
print(f"\nConfusion Matrix:")
print(f"              Predicted")
print(f"              Male  Female")
print(f"Actual Male   {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Female {cm[1,0]:4d}  {cm[1,1]:4d}")

# Per-class accuracy
print(f"\nPer-class Performance:")
for i, label in enumerate(["Male", "Female"]):
    precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
    recall = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  {label:6s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# ============= SAMPLE PREDICTIONS =============
print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

n_samples = min(20, len(X_test))
sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

print(f"\nShowing {n_samples} random predictions:")
print("-" * 80)

for idx in sample_indices:
    pred_age = age_pred_years[idx]
    pred_gender = "Male" if gender_pred_classes[idx] == 0 else "Female"
    pred_gender_conf = np.max(gender_pred[idx])
    
    true_age = age_test_raw[idx]
    true_gender = "Male" if gender_true_classes[idx] == 0 else "Female"
    
    age_error = abs(pred_age - true_age)
    gender_correct = "✓" if pred_gender == true_gender else "✗"
    
    print(f"True: {true_gender:6s} {true_age:3.0f}y | "
          f"Pred: {pred_gender:6s} ({pred_gender_conf:.2f}) {pred_age:3.0f}y | "
          f"Age Err: {age_error:4.1f}y {gender_correct}")

# ============= VISUALIZATIONS =============
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Age Error Distribution
axes[0, 0].hist(age_errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(age_mae, color='red', linestyle='--', label=f'MAE: {age_mae:.2f}')
axes[0, 0].set_xlabel('Absolute Error (years)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Age Prediction Error Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. True vs Predicted Age Scatter
axes[0, 1].scatter(age_test_raw, age_pred_years, alpha=0.4, s=10)
axes[0, 1].plot([0, 116], [0, 116], 'r--', label='Perfect Prediction')
axes[0, 1].set_xlabel('True Age (years)')
axes[0, 1].set_ylabel('Predicted Age (years)')
axes[0, 1].set_title('True vs Predicted Age')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Error by True Age
axes[0, 2].scatter(age_test_raw, age_errors, alpha=0.4, s=10, c='coral')
axes[0, 2].axhline(age_mae, color='red', linestyle='--', label=f'MAE: {age_mae:.2f}')
axes[0, 2].set_xlabel('True Age (years)')
axes[0, 2].set_ylabel('Absolute Error (years)')
axes[0, 2].set_title('Prediction Error by Age')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Gender Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Male', 'Female'], 
            yticklabels=['Male', 'Female'],
            ax=axes[1, 0], cbar_kws={'label': 'Count'})
axes[1, 0].set_xlabel('Predicted Gender')
axes[1, 0].set_ylabel('True Gender')
axes[1, 0].set_title(f'Gender Confusion Matrix\nAccuracy: {gender_accuracy*100:.2f}%')

# 5. Age Error by Gender
male_mask = gender_true_classes == 0
female_mask = gender_true_classes == 1
axes[1, 1].boxplot([age_errors[male_mask], age_errors[female_mask]], 
                    labels=['Male', 'Female'])
axes[1, 1].set_ylabel('Absolute Age Error (years)')
axes[1, 1].set_title('Age Error Distribution by Gender')
axes[1, 1].grid(True, alpha=0.3)

# 6. Cumulative Error Distribution
sorted_errors = np.sort(age_errors)
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
axes[1, 2].plot(sorted_errors, cumulative, linewidth=2, color='green')
axes[1, 2].axvline(5, color='orange', linestyle='--', alpha=0.7, label='5 years')
axes[1, 2].axvline(10, color='red', linestyle='--', alpha=0.7, label='10 years')
axes[1, 2].set_xlabel('Absolute Error (years)')
axes[1, 2].set_ylabel('Cumulative Percentage (%)')
axes[1, 2].set_title('Cumulative Error Distribution')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved evaluation plots to 'model_evaluation_results.png'")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)