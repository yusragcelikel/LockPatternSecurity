import pandas as pd

# Dosya yolunu belirle
data_path = "data/CICMalDroid_2020.csv"

# Veri setini yükle
df = pd.read_csv(data_path)

# Sütun isimlerini küçük harfe çevir
df.columns = df.columns.str.lower()

# Sütun isimlerini göster
print("Sütun Adları:", df.columns)

# 'class' sütunu kullanılarak malware ve benign ayrımı yapalım
malware_classes = [1, 2, 3, 4]  # Adware, Banking, SMS malware, Riskware
benign_classes = [5]  # Benign

# Malware ve benign verileri ayır
malware = df[df['class'].isin(malware_classes)]
benign = df[df['class'].isin(benign_classes)]

# Dengeli veri seti oluştur (her iki sınıfın eşit sayıda örnek içermesini sağla)
sample_size = min(len(malware), len(benign))
malware = malware.sample(sample_size, random_state=42)
benign = benign.sample(sample_size, random_state=42)

# İşlenmiş veriyi kaydet
processed_data_path = "data/CICMalDroid_Processed.csv"
processed_df = pd.concat([malware, benign])
processed_df.to_csv(processed_data_path, index=False)

print(f"CICMalDroid verisi işlendi ve {processed_data_path} dosyasına kaydedildi!")
