import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix
)
import joblib
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ================================
# 1. Load dataset
# ================================
DATA_PATH = r"D:/yul/mitigasi/Mitigasi_Bencana/data/processed/train_clean.csv"

print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Validasi kolom
REQUIRED_COLS = ["clean_text", "target"]
for col in REQUIRED_COLS:
    if col not in df.columns:
        raise ValueError(f"[ERROR] Kolom '{col}' tidak ditemukan di dataset!")

# ================================
# 2. Kolom yang digunakan
# ================================
TEXT_COL = "clean_text"
LABEL_COL = "target"

X = df[TEXT_COL]
y = df[LABEL_COL]

print(f"[INFO] Total Data: {len(df)} baris")

# ================================
# 3. Split training / validation
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("[INFO] Data berhasil dibagi menjadi train/test")

# ================================
# 4. TF-IDF Vectorizer
# ================================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("[INFO] TF-IDF Vectorizer selesai dibuat")

# ================================
# 5. Train Logistic Regression
# ================================
print("[INFO] Training model Logistic Regression...")
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# ================================
# 6. Evaluation Model
# ================================
print("/n=== MODEL EVALUATION ===")

y_pred = model.predict(X_test_vec)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"/n[RESULT] Accuracy: {acc:.4f}")

# Classification report
report = classification_report(y_test, y_pred)
print("/n=== Classification Report ===")
print(report)

# ================================
# 6.1 Save Evaluation Report ke CSV
# ================================
eval_report = classification_report(y_test, y_pred, output_dict=True)
eval_df = pd.DataFrame(eval_report).transpose()

EVAL_PATH = r"D:/yul/mitigasi/Mitigasi_Bencana/data/processed/evaluation_report.csv"
eval_df.to_csv(EVAL_PATH, index=True)

print(f"[INFO] Evaluation disimpan ke: {EVAL_PATH}")

# ================================
# 6.2 Plot Confusion Matrix
# ================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Non-Bencana', 'Bencana'],
    yticklabels=['Non-Bencana', 'Bencana']
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()

CM_PATH = r"D:/yul/mitigasi/Mitigasi_Bencana/models/confusion_matrix.png"
plt.savefig(CM_PATH)
plt.close()

print(f"[INFO] Confusion matrix disimpan ke: {CM_PATH}")

# ================================
# 7. Save model dan vectorizer
# ================================
MODEL_DIR = r"D:/MITIGASI BENCANA/Mitigasi_Bencana/models"
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "logreg_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("[INFO] Model dan TF-IDF vectorizer berhasil disimpan di folder 'models/'")
