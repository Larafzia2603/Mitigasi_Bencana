import joblib
import os

# path absolut file
BASE_PATH = "D:/Ahh miitgasi/Mitigasi_Bencana/models"
TFIDF_PATH = os.path.join(BASE_PATH, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_PATH, "logreg_model.pkl")

# load tfidf
tfidf = joblib.load(TFIDF_PATH)

# load model
clf = joblib.load(MODEL_PATH)

# contoh input
text = ["banjir besar melanda kota sejak dini hari"]

# vectorize
vec = tfidf.transform(text)

# prediksi
pred = clf.predict(vec)

print("Prediksi:", pred[0])

