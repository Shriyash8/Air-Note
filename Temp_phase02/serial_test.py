import serial
import numpy as np
import joblib

# Load trained model
model = joblib.load("airnote_model.pkl")

# Serial setup
ser = serial.Serial("COM6", 115200)

buffer = []
BUFFER_SIZE = 90

print("Draw in air...\n")

# SAME feature extraction (DO NOT CHANGE)
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


while True:
    try:
        line = ser.readline().decode().strip()
        values = list(map(float, line.split(",")))

        if len(values) == 6:
            buffer.append(values)

        if len(buffer) >= BUFFER_SIZE:
            sample = np.array(buffer)

            # 🔥 Feature extraction
            features = extract_features(sample)

            # 🔥 Prediction
            prediction = model.predict([features])[0]

            print("\n======================")
            print("Predicted Letter:", prediction)
            print("======================\n")

            buffer = []

    except Exception as e:
        print("Error:", e)