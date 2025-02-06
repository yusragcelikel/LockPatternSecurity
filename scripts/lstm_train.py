import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Veri setini yükle
data_path = "data/custom_dataset_processed.csv"
df = pd.read_csv(data_path)

# Zaman serisi verisini tensor formatına getir
X = df.drop(columns=['Time']).values  # Zaman sütunu hariç tüm sütunlar
X = torch.tensor(X, dtype=torch.float32)

# PyTorch DataLoader oluştur
dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# LSTM Autoencoder Modeli
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

# Modeli oluştur ve parametreleri ayarla
input_size = X.shape[1]  # Giriş boyutu (özellik sayısı)
hidden_size = 64
num_layers = 2

model = LSTMAutoencoder(input_size, hidden_size, num_layers)

# Loss fonksiyonu ve optimizasyon yöntemi
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Modeli eğitme
num_epochs = 10
for epoch in range(num_epochs):
    for batch_X in dataloader:
        batch_X = batch_X[0].unsqueeze(1)  # LSTM için uygun forma sok
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_X)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Eğitilmiş modeli kaydetme
torch.save(model.state_dict(), "models/lstm_autoencoder.pth")
print("LSTM Autoencoder modeli başarıyla eğitildi ve kaydedildi!")
