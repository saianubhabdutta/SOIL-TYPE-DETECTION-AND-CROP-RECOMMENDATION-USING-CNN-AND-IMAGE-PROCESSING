import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# =========================
# Load Dataset
# =========================

df = pd.read_csv("crop_model/crop_dataset.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)
print("Columns:", df.columns)

# =========================
# Features & Labels
# =========================

X = df.drop("label", axis=1)
y = df["label"]

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Feature Scaling (IMPORTANT)
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Model Training
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# =========================
# Save Model + Scaler
# =========================

pickle.dump(model, open("crop_model/crop_model.pkl", "wb"))
pickle.dump(scaler, open("crop_model/scaler.pkl", "wb"))

print("Model + Scaler saved successfully!")

# =========================
# Save Mean Values per Crop
# =========================

means = df.groupby("label").mean()

means.to_csv("crop_model/crop_means.csv")

print("Crop means saved!")
