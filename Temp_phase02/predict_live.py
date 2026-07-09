import numpy as np
import joblib

# Load trained model
model = joblib.load("airnote_model.pkl")

# Same feature extraction function (MUST BE IDENTICAL)
def extract_features(sample):
    features = []

    for col in range(6):  # ax, ay, az, gx, gy, gz
        data = sample[:, col]

        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.max(data))
        features.append(np.min(data))
        features.append(np.max(data) - np.min(data))
        features.append(np.sqrt(np.mean(data**2)))
        features.append(np.sum(data**2))

    return features

# Simulated buffer (replace with ESP data later)
buffer = []

print("Start drawing...")

while True:
    try:
        # 👇 Replace this with real serial/WiFi input later
        line = input("Enter 6 values (ax ay az gx gy gz): ")

        values = list(map(float, line.split()))
        buffer.append(values)

        # When enough data collected (~2 sec)
        if len(buffer) >= 50:   # adjust based on your sampling rate
            sample = np.array(buffer)

            features = extract_features(sample)
            prediction = model.predict([features])

            print("Predicted Letter:", prediction[0])

            buffer = []  # reset after prediction

    except Exception as e:
        print("Error:", e)