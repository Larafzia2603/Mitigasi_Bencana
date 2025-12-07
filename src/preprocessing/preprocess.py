import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os


# Download stopwords & wordnet
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text):
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def run_preprocessing():
    # Tentukan base directory → folder project utama
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    input_path = os.path.join(BASE_DIR, "D:/yul/mitigasi/Mitigasi_Bencana/data/raw/tweets_banjir.csv")
    output_dir = os.path.join(BASE_DIR, "data/processed")
    output_path = os.path.join(output_dir, "D:/yul/mitigasi/Mitigasi_Bencana/data/processed/tweets_processed.py")

    # Buat folder processed jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    print("Membaca data...")
    df = pd.read_csv(input_path)

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    print("Preprocessing text...")
    df["preprocessed_text"] = df["clean_text"].apply(preprocess_text)

    print("Menyimpan hasil ke:", output_path)
    df.to_csv(output_path, index=False)

    print("✔ Selesai! File tersimpan.")


if __name__ == "__main__":
    run_preprocessing()