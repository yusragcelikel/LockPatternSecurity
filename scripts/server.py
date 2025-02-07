import flwr as fl
import torch
import os

# Sunucu tarafında modeli kaydetme fonksiyonu
def save_model(parameters, filename="models/federated_model.pth"):
    os.makedirs("models", exist_ok=True)
    state_dict = {f"param_{i}": torch.tensor(param) for i, param in enumerate(parameters)}
    torch.save(state_dict, filename)
    print(f"Federated Learning sonrası model kaydedildi: {filename}")

# Yeni Federated Learning sunucu başlatma fonksiyonu
def start_server():
    strategy = fl.server.strategy.FedAvg()

    # **Güncellenmiş start_server() kullanımı**
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

if __name__ == "__main__":
    start_server()
