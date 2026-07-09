import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# STEP 1: Load Dataset
# ==============================
df = pd.read_csv("temp02_dataset.csv")

# Clean labels (remove spaces like 'G ')
df['label'] = df['label'].str.strip()

# ==============================
# STEP 2: Feature Extraction
# ==============================
def extract_features(sample):
    features = []

    for col in ['ax','ay','az','gx','gy','gz']:
        data = sample[col].values

        # Basic features
        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.max(data))
        features.append(np.min(data))

        # Strong features
        features.append(np.max(data) - np.min(data))  # range
        features.append(np.sqrt(np.mean(data**2)))    # RMS
        features.append(np.sum(data**2))              # energy

    return features

# ==============================
# STEP 3: Convert Dataset
# ==============================
feature_list = []
labels = []

for seq_id in df['seq'].unique():
    sample = df[df['seq'] == seq_id]

    features = extract_features(sample)
    label = sample['label'].iloc[0]

    feature_list.append(features)
    labels.append(label)

X = np.array(feature_list)
y = np.array(labels)

print("Feature shape:", X.shape)

# ==============================
# STEP 4: Train Model
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# STEP 5: Evaluate
# ==============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# STEP 6: Save Model
# ==============================
joblib.dump(model, "airnote_model.pkl")

print("\nModel saved as airnote_model.pkl")