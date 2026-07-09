import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("temp02_dataset.csv")


df['label'] = df['label'].str.strip()


def extract_features(sample):
    features = []

    for col in ['ax','ay','az','gx','gy','gz']:
        data = sample[col].values

        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.max(data))
        features.append(np.min(data))
        features.append(np.max(data) - np.min(data))
        features.append(np.sqrt(np.mean(data**2)))
        features.append(np.sum(data**2))

    return features



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


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Classification report (F1, precision, recall)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))