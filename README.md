ALUR PROYEK
1. Pengambilan Data
Data real-time diperoleh dari Twitter API (X Developer) melalui endpoint search tweet.
Data historical diambil dari dataset Kaggle “Real or Not? Disaster Tweets”.

2. Preprocessing
Membersihkan teks (casefolding, removing punctuation, removing URLs, stopwords, stemming/lemmatization).
Preprocessing dilakukan secara terpisah untuk:
- data tweet API
- data Kaggle

Output preprocessing berupa file teks bersih (clean_text) yang siap diproses oleh model.

3. Training Model
Model dilatih menggunakan:
- TF-IDF Vectorizer sebagai text feature extractor
- Logistic Regression sebagai model klasifikasi bencana

File model disimpan dalam format .pkl:
- tfidfvectorized.pkl
- model_logreg.pkl

4. Prediksi Tweet Real-Time
Data tweet terbaru dari API diproses oleh:
- TF-IDF
- model Logistic Regression

Hasil prediksi disimpan dalam predicted_tweets.csv dengan label 0 (non-bencana) dan 1 (bencana).

5. Evaluasi Model
Evaluasi dilakukan pada data testing dari Kaggle.
Metrik yang digunakan:
- accuracy
- precision
- recall
- f1-score
- confusion matrix

6. Visualisasi
Hasil prediksi data tweet divisualisasikan dalam bentuk:
- bar chart distribusi prediksi
- pie chart kategori tweet (opsional)

Visualisasi disimpan dalam folder visualizations/.


PEMBAGIAN TUGAS ANGGOTA:
Yulianingsi_F52124004 :
- Preprocessing data tweet API
- Menyiapkan dan mengonfigurasi Bearer Token untuk akses API Twitter

Gaida Muthmainnah_F52124023 :
- Preprocessing dataset Kaggle
- Visualisasi hasil prediksi tweet API

Rayhan Aliffinsi_F52124010 :
- Melakukan input dan pembagian data train/test dari dataset Kaggle
- Melakukan testing dan evaluasi terhadap model

Nur Ainun_ F52124024 :
- Menyusun struktur folder proyek
- Menyusun alur pengerjaan proyek
- Melakukan revisi dan penyelarasan struktur dokumentasi

Lara Fauzia_F52124015 :
- Melakukan training model menggunakan Logistic Regression
- Menyimpan model dan vectorizer dalam bentuk .pkl