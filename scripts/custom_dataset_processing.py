import pandas as pd

# Özel veri setini yükle
custom_data_path = "data/custom_dataset.csv"
df = pd.read_csv(custom_data_path)

# Eksik değerleri temizle
df.dropna(inplace=True)

# Zaman damgalarını normalize et
df["Time"] = df["Time"] / df["Time"].max()

# İşlenmiş veriyi kaydet
processed_custom_data_path = "data/custom_dataset_processed.csv"
df.to_csv(processed_custom_data_path, index=False)

print(f"Özel veri seti işlendi ve {processed_custom_data_path} dosyasına kaydedildi!")
