import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from PIL import Image

print("Veri seti yükleniyor...")

# Veri setini yükle
data_path = "data/CICMalDroid_Processed.csv"
df = pd.read_csv(data_path)

print("Veri seti yüklendi. İlk 5 satır:")
print(df.head())

# MobileNetV3 modelini yükle
print("MobileNetV3 modeli yükleniyor...")
mobilenet_v3 = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# Modelin son katmanını değiştirerek Binary Classification için ayarla
mobilenet_v3.classifier[3] = torch.nn.Linear(mobilenet_v3.classifier[3].in_features, 2)

print("Model hazır. Veriler işleniyor...")

# Veriyi dönüştürme işlemleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# Veriyi Tensor haline getirme
X_raw = df.drop(columns=['class']).values
y = df['class'].apply(lambda x: 0 if x in [5] else 1).values

print("Özellikler sayısı:", X_raw.shape[1])
print("Toplam veri sayısı:", X_raw.shape[0])

# Özellikleri 20x20 boyutuna yeniden şekillendirme ve grayscale bir görüntü oluşturma
def reshape_features_to_image(features):
    img_array = np.zeros((20, 20), dtype=np.uint8)
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
    
    min_val, max_val = features.min(), features.max()
    if max_val - min_val == 0:
        norm_features = np.zeros_like(features)
    else:
        norm_features = (features - min_val) / (max_val - min_val)

    num_features = min(400, len(features))
    img_array.flat[:num_features] = (norm_features[:num_features] * 255).astype(np.uint8)

    img_pil = Image.fromarray(img_array, mode="L")
    return transform(img_pil)

print("Özelliklerden görüntüler oluşturuluyor...")
X_images = torch.stack([reshape_features_to_image(features) for features in X_raw])
y = torch.tensor(y, dtype=torch.long)

print("Görüntüler oluşturuldu. DataLoader hazırlanıyor...")

# PyTorch DataLoader ile veri kümesi oluşturma
dataset = TensorDataset(X_images, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Eğitime başlanıyor...")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mobilenet_v3.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs} başladı.")
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_X = batch_X.repeat(1, 3, 1, 1)
        outputs = mobilenet_v3(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Batch {batch_idx}, Loss: {loss.item():.4f}")

print("Model eğitimi tamamlandı. Model kaydediliyor...")

# Eğitilmiş modeli kaydetme
torch.save(mobilenet_v3.state_dict(), "models/mobilenet_v3.pth")
print("MobileNetV3 modeli başarıyla eğitildi ve kaydedildi!")
