import subprocess
import time
import socket

# Sunucunun belirli bir porta bağlanıp bağlanmadığını kontrol eden fonksiyon
def is_server_running(host="localhost", port=5000):
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except (socket.timeout, ConnectionRefusedError):
        return False

# Sunucuyu başlat
def start_server():
    print("[INFO] Sunucu başlatılıyor...")
    server_process = subprocess.Popen(["python", "scripts/server.py"])
    
    # Sunucunun gerçekten başladığını kontrol et
    for _ in range(10):
        if is_server_running():
            print("[INFO] Sunucu başarıyla başlatıldı!")
            return server_process
        time.sleep(1)
    
    print("[ERROR] Sunucu başlatılamadı!")
    server_process.terminate()
    return None

# İstemcileri başlat
def start_clients(num_clients=2):
    print(f"[INFO] {num_clients} istemci başlatılıyor...")
    processes = []
    
    for i in range(num_clients):
        p = subprocess.Popen(["python", "scripts/client.py"])
        processes.append(p)
        time.sleep(1)  # İstemcileri sırayla başlat
    
    for p in processes:
        p.wait()

if __name__ == "__main__":
    server_process = start_server()
    
    if server_process:
        start_clients(num_clients=2)
        server_process.terminate()  # Sunucuyu kapat (isteğe bağlı)
    else:
        print("[ERROR] Sunucu başlamadığı için istemciler çalıştırılmadı!")
