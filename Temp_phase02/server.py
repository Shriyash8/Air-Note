import socket
import numpy as np
import joblib
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)


model = joblib.load("airnote_model.pkl")

BUFFER_SIZE = 90
buffer = []


SOCKET_HOST = "0.0.0.0"
SOCKET_PORT = 5001   

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((SOCKET_HOST, SOCKET_PORT))
sock.listen(1)

print("Waiting for ESP connection on port 5001...")
conn, addr = sock.accept()
print("Connected by", addr)


def extract_features(sample):
    features = []

    for col in range(6):
        data = sample[:, col]

        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.max(data))
        features.append(np.min(data))
        features.append(np.max(data) - np.min(data))
        features.append(np.sqrt(np.mean(data**2)))
        features.append(np.sum(data**2))

    return features


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict")
def predict():
    global buffer

    buffer = []

    print("\n--- NEW CAPTURE ---")

    while True:
        try:
            data = conn.recv(1024).decode()

            lines = data.strip().split("\n")

            for line in lines:
                values = list(map(float, line.split(",")))

                if len(values) == 6:
                    buffer.append(values)
                    print("Buffer:", len(buffer))

            if len(buffer) >= BUFFER_SIZE:
                print("--- PREDICTING ---")

                sample = np.array(buffer)

                features = extract_features(sample)
                prediction = model.predict([features])[0]

                return jsonify({"prediction": prediction})

        except:
            continue


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)