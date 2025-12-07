import pandas as pd
import matplotlib.pyplot as plt

# baca hasil prediksi
df = pd.read_csv(r'D:\BENCANA NIH\Mitigasi_Bencana\data\tweets_predicted.csv')

# hitung jumlah tiap kategori
counts = df['prediction'].value_counts()

# bar chart
plt.figure(figsize=(6,4))
counts.plot(kind='bar')
plt.title('Distribusi Prediksi Tweet')
plt.xlabel('Kategori (0 = tidak bencana, 1 = bencana)')
plt.ylabel('Jumlah Tweet')
plt.tight_layout()
plt.savefig('D:\BENCANA NIH\Mitigasi_Bencana\models\Visualisasi.PNG')
plt.close()