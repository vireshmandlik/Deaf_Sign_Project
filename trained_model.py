import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("gesture_dataset.csv", header=None)

# -------------------------------
# Fix if all values are in a single column
# -------------------------------
if data.shape[1] == 1:
    data = data[0].str.split(",", expand=True)

# Remove any empty rows (if present)
data = data.dropna()

print("✅ Dataset Shape:", data.shape)
print("First row preview:", data.iloc[0].tolist())

# -------------------------------
# Separate features (X) and labels (y)
# -------------------------------
X = data.iloc[:, :-1].astype(float)  # All columns except last
y = data.iloc[:, -1]                 # Last column = label

print("✅ Features shape:", X.shape)
print("✅ Labels shape:", y.shape)

# -------------------------------
# Split dataset into training and test sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Random Forest Model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Evaluate Model Accuracy
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", accuracy)

# -------------------------------
# Save trained model as pickle
# -------------------------------
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model Saved Successfully as gesture_model.pkl")
