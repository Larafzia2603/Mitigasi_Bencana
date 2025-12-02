import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ================================
# 1. Load dataset yang sudah dipreprocessing
# ================================
DATA_PATH = os.path.join("data", "processed", "train_clean.csv")

print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Pastikan kolom sesuai
TEXT_COL = "clean_text"
LABEL_COL = "label"

X = df[TEXT_COL]
y = df[LABEL_COL]

# ================================
# 2. Split train & valid
# ================================
print("[INFO] Splitting dataset...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 3. TF-IDF Vectorization
# ================================
print("[INFO] Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_valid_vec = vectorizer.transform(X_valid)

# ================================
# 4. Train Logistic Regression
# ================================
print("[INFO] Training Logistic Regression model...")
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# ================================
# 5. Evaluation
# ================================
print("[INFO] Evaluating model...")
y_pred = model.predict(X_valid_vec)

print("\nAccuracy:", accuracy_score(y_valid, y_pred))
print("\nClassification Report:")
print(classification_report(y_valid, y_pred))

# ================================
# 6. Save model & vectorizer
# ================================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print(f"\n[✔] Model saved to: {MODEL_PATH}")
print(f"[✔] Vectorizer saved to: {VECTORIZER_PATH}")
