import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ================================
# 1. Load dataset
# ================================
DATA_PATH = r"D:\BENCANA NIH\Mitigasi_Bencana\data\processed\train_clean.csv"

print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ================================
# 2. Kolom yang digunakan
# ================================
TEXT_COL = "clean_text"      # teks input
LABEL_COL = "target"         # label (0/1 di dataset kamu)

X = df[TEXT_COL]
y = df[LABEL_COL]

# ================================
# 3. Split training / validation
# ================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 4. TF-IDF Vectorizer
# ================================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# ================================
# 5. Train Logistic Regression
# ================================
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# ================================
# 6. Evaluation
# ================================
y_pred = model.predict(X_val_vec)
print("[INFO] Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# ================================
# 7. Save model dan vectorizer
# ================================
MODEL_DIR = r"D:\MITIGASI BENCANA\Mitigasi_Bencana\models"
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "logreg_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("[INFO] Model saved in folder 'models/'")
