# evaluate_emotion.py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# load data and model
X_test = np.load("processed_data/X.npy", allow_pickle=False)
y = np.load("processed_data/y.npy", allow_pickle=False)

# we used split in training; recreate test split the same way:
from sklearn.model_selection import train_test_split
_, X_temp, _, y_temp = train_test_split(X_test, y, test_size=0.2, random_state=42, stratify=y)
_, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ensure shape and scale like training script expects
if X_test.ndim == 3:
    X_test = X_test[..., None]
if X_test.max() > 1.0:
    X_test = X_test.astype("float32") / 255.0

model = load_model("models/best_emotion_model.h5")
preds = model.predict(X_test, verbose=0)
y_pred = preds.argmax(axis=1)

print("Test accuracy (loaded model):", (y_pred == y_test).mean())
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, digits=4))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = sorted(list(set(y_test)))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")
