import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# 1. Load hasil prediksi
# ==============================
csv_path = R"D:\BENCANA NIH\Mitigasi_Bencana\data\tweets_predicted.csv"
df = pd.read_csv(csv_path)

# ==============================
# 2. Hitung jumlah prediksi
# ==============================
counts = df["prediction"].value_counts()

# Beri label aman
labels = ["Tidak Bencana (0)", "Bencana (1)"]

# Pastikan urutannya konsisten
sizes = [
    counts.get(0, 0),
    counts.get(1, 0)
]

# ==============================
# 3. Pastikan folder visualisasi ada
# ==============================
vis_folder = r"D:\BENCANA NIH\Mitigasi_Bencana\models"
os.makedirs(vis_folder, exist_ok=True)

save_path = os.path.join(vis_folder, "prediction_pie_chart.png")

# ==============================
# 4. Buat PIE CHART
# ==============================
plt.figure(figsize=(6, 6))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)
plt.title("Distribusi Prediksi Tweet (Pie Chart)")
plt.tight_layout()

# Simpan
plt.savefig(save_path)
plt.close()

print("✔ Pie chart berhasil dibuat:", save_path)
