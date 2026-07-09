import socket

HOST = "0.0.0.0"   # listen on all interfaces
PORT = 5000

# Create socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Waiting for ESP connection...")

conn, addr = server_socket.accept()
print("Connected by", addr)

while True:
    try:
        data = conn.recv(1024).decode()

        if data:
            print(data.strip())

    except:
        print("Connection lost")
        break