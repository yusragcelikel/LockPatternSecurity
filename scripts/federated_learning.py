import torch
import syft as sy
from torch import nn
import pandas as pd
from lstm_train import LSTMAutoencoder
from mobilenet_train import mobilenet_v3  # MobileNetV3 modelini al
import torch.optim as optim

# PySyft Hook başlat
hook = sy.TorchHook(torch)

# Sanal istemciler (Client'lar) oluştur
client1 = sy.VirtualWorker(hook, id="client1")
client2 = sy.VirtualWorker(hook, id="client2")
client3 = sy.VirtualWorker(hook, id="client3")

# Global Model Oluştur
print("Global model oluşturuluyor...")

# MobileNetV3 yükle
mobilenet_model = mobilenet_v3
mobilenet_model.load_state_dict(torch.load("models/mobilenet_v3.pth"))

# LSTM Autoencoder yükle
input_size = 470  # Özellik sayısına göre ayarla
hidden_size = 64
num_layers = 2
lstm_model = LSTMAutoencoder(input_size, hidden_size, num_layers)
lstm_model.load_state_dict(torch.load("models/lstm_autoencoder.pth"))

# Modeli istemcilere gönder
mobilenet_client1 = mobilenet_model.send(client1)
mobilenet_client2 = mobilenet_model.send(client2)
mobilenet_client3 = mobilenet_model.send(client3)

lstm_client1 = lstm_model.send(client1)
lstm_client2 = lstm_model.send(client2)
lstm_client3 = lstm_model.send(client3)

print("Model istemcilere gönderildi.")

# Federated Learning Eğitimi
def train_federated(model, clients, num_epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for client in clients:
            model_client = model.send(client)  # Modeli istemciye gönder
            optimizer.zero_grad()

            # Örnek veri oluştur
            data = torch.randn(32, 3, 224, 224)  # MobileNet için giriş verisi
            labels = torch.randint(0, 2, (32,))  # 0 = Benign, 1 = Malware

            # Eğitim adımları
            outputs = model_client(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.get()  # Kayıp değerini istemciden geri al

            model_client = model_client.get()  # Modeli geri al

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(clients):.4f}")

# MobileNetV3 için Federated Learning eğitimi
print("Federated Learning başlatılıyor: MobileNetV3")
train_federated(mobilenet_model, [client1, client2, client3])

# LSTM Autoencoder için Federated Learning eğitimi
print("Federated Learning başlatılıyor: LSTM Autoencoder")
train_federated(lstm_model, [client1, client2, client3])

# Modeli kaydet
torch.save(mobilenet_model.state_dict(), "models/federated_mobilenet.pth")
torch.save(lstm_model.state_dict(), "models/federated_lstm.pth")

print("Federated Learning modeli başarıyla eğitildi ve kaydedildi!")
