import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# =====================================
# LOAD DATASET
# =====================================

print("Loading Dataset...")

df = pd.read_csv("temp02_dataset.csv")

df["label"] = df["label"].str.strip()

# =====================================
# FEATURE EXTRACTION
# =====================================

def extract_features(sample):

    features = []

    for col in ["ax","ay","az","gx","gy","gz"]:

        data = sample[col].values

        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.max(data))
        features.append(np.min(data))
        features.append(np.max(data)-np.min(data))
        features.append(np.sqrt(np.mean(data**2)))
        features.append(np.sum(data**2))

    return features

# =====================================
# CREATE FEATURE MATRIX
# =====================================

feature_list = []
labels = []

for seq_id in df["seq"].unique():

    sample = df[df["seq"] == seq_id]

    feature_list.append(extract_features(sample))

    labels.append(sample["label"].iloc[0])

X = np.array(feature_list)
y = np.array(labels)

print("Feature Shape:", X.shape)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# TRAIN MODEL
# =====================================

print("Training Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# PREDICT
# =====================================

y_pred = model.predict(X_test)

# =====================================
# OVERALL METRICS
# =====================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n========== RESULTS ==========")

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

# =====================================
# CLASSIFICATION REPORT
# =====================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

print("Saved: confusion_matrix.png")

# =====================================
# CLASS DISTRIBUTION GRAPH
# =====================================

class_counts = df.groupby("label")["seq"].nunique()

plt.figure(figsize=(8,5))

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.title("Class Distribution")
plt.xlabel("Letters")
plt.ylabel("Samples")

plt.savefig("class_distribution.png")

print("Saved: class_distribution.png")

# =====================================
# PRECISION GRAPH
# =====================================

labels_list = []
precision_list = []

for label in model.classes_:
    labels_list.append(label)
    precision_list.append(report[label]["precision"])

plt.figure(figsize=(8,5))

plt.bar(labels_list, precision_list)

plt.title("Precision per Class")
plt.xlabel("Letters")
plt.ylabel("Precision")

plt.ylim(0,1.1)

plt.savefig("precision_graph.png")

print("Saved: precision_graph.png")

# =====================================
# RECALL GRAPH
# =====================================

recall_list = []

for label in model.classes_:
    recall_list.append(report[label]["recall"])

plt.figure(figsize=(8,5))

plt.bar(labels_list, recall_list)

plt.title("Recall per Class")
plt.xlabel("Letters")
plt.ylabel("Recall")

plt.ylim(0,1.1)

plt.savefig("recall_graph.png")

print("Saved: recall_graph.png")

# =====================================
# F1 GRAPH
# =====================================

f1_list = []

for label in model.classes_:
    f1_list.append(report[label]["f1-score"])

plt.figure(figsize=(8,5))

plt.bar(labels_list, f1_list)

plt.title("F1 Score per Class")
plt.xlabel("Letters")
plt.ylabel("F1 Score")

plt.ylim(0,1.1)

plt.savefig("f1_graph.png")

print("Saved: f1_graph.png")

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(
    model,
    "airnote_model.pkl"
)

print("\nModel Saved Successfully")

print("\nDONE")
