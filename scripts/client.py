import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import flwr as fl
import os

# 📌 Mobilenet Modeli (Özelleştirilmiş Küçük Model)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 📌 LSTM Autoencoder Modeli (Düzeltilmiş Giriş Boyutu!)
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=28, hidden_size=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

# 📌 Veriyi yükleme fonksiyonu
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    return trainloader

# 📌 Eğitim fonksiyonu
def train(model, trainloader, lstm=False, epochs=1):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() if not lstm else nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()

            if lstm:
                images = images.view(images.shape[0], 28, 28)  # 📌 LSTM için giriş boyutu 28x28
                outputs = model(images)
                loss = criterion(outputs, images)  # 🔥 LSTM Autoencoder için giriş=çıkış
            else:
                outputs = model(images.view(images.shape[0], -1))
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

# 📌 Modeli kaydetme fonksiyonu
def save_model(model, filename):
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), filename)
    print(f"✅ Model başarıyla kaydedildi: {filename}")

# 📌 Flower İstemci Tanımlama
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, mobilenet_model, lstm_model):
        self.mobilenet_model = mobilenet_model
        self.lstm_model = lstm_model
        self.trainloader = load_data()

    def get_parameters(self, config):
        mobilenet_params = [val.cpu().numpy() for val in self.mobilenet_model.state_dict().values()]
        lstm_params = [val.cpu().numpy() for val in self.lstm_model.state_dict().values()]
        return mobilenet_params + lstm_params

    def set_parameters(self, parameters):
        mobilenet_params = parameters[:len(list(self.mobilenet_model.state_dict()))]
        lstm_params = parameters[len(list(self.mobilenet_model.state_dict())):]

        mobilenet_state_dict = {k: torch.tensor(v) for k, v in zip(self.mobilenet_model.state_dict().keys(), mobilenet_params)}
        self.mobilenet_model.load_state_dict(mobilenet_state_dict, strict=True)

        lstm_state_dict = {k: torch.tensor(v) for k, v in zip(self.lstm_model.state_dict().keys(), lstm_params)}
        self.lstm_model.load_state_dict(lstm_state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.mobilenet_model, self.trainloader, lstm=False)
        train(self.lstm_model, self.trainloader, lstm=True)  # 🔥 LSTM için düzeltilmiş eğitim
        save_model(self.mobilenet_model, "models/federated_mobilenet.pth")
        save_model(self.lstm_model, "models/federated_lstm.pth")
        return self.get_parameters(config), len(self.trainloader.dataset), {}

# 📌 İstemciyi Başlatma
def start_client():
    mobilenet_model = Net()
    lstm_model = LSTMAutoencoder()

    fl.client.start_client(
        server_address="localhost:8080",
        client=FlowerClient(mobilenet_model, lstm_model).to_client()
    )

if __name__ == "__main__":
    start_client()
