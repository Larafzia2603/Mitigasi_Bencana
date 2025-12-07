import pandas as pd
from joblib import load

# ===== 1. Load model & vectorizer (PAKAI JOBLIB — PALING AMAN) =====
tfidf = load("D:/yul/mitigasi/Mitigasi_Bencana/models/tfidf_vectorizer.pkl")
model = load("D:/yul/mitigasi/Mitigasi_Bencana/models/logreg_model.pkl")

# ===== 2. Load tweet yang sudah dipreprocessing =====
df = pd.read_csv("D:/yul/mitigasi/Mitigasi_Bencana/data/processed/tweets_processed.csv")

# Pastikan kolomnya ada
if "clean_text" not in df.columns:
    raise ValueError("Kolom 'clean_text' tidak ditemukan di tweets_processed.csv!")

# Transform ke TF-IDF
X_api = tfidf.transform(df["clean_text"])

# ===== 3. Prediksi =====
preds = model.predict(X_api)
probas = model.predict_proba(X_api)

df["prediction"] = preds
df["prob_nonbencana"] = probas[:, 0]
df["prob_bencana"] = probas[:, 1]

# ===== 4. Simpan hasil =====
output_path = "D:/yul/mitigasi/Mitigasi_Bencana/data/tweets_predicted.csv"
df.to_csv(output_path, index=False)

print("✔ Prediksi selesai, disimpan ke:", output_path)
print("=== Hasil Prediksi Tweet Real-Time ===")
print(df[["text", "clean_text", "prediction"]].head())