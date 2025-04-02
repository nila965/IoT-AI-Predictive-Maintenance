# model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Generate Random Training Data
def generate_data(n=1000):
    np.random.seed(42)
    data = {
        "Temperature": np.random.uniform(20, 50, n),
        "Humidity": np.random.uniform(30, 90, n),
        "Hall Effect": np.random.randint(0, 2, n),
        "Sound": np.random.uniform(10, 100, n),
        "Current": np.random.uniform(0, 15, n),
        "Gas": np.random.uniform(0, 200, n),
        "Flame": np.random.randint(0, 2, n),
        "Relay Status": np.random.randint(0, 2, n),
        "Maintenance Required": np.random.randint(0, 2, n)  # Target (1=Yes, 0=No)
    }
    return pd.DataFrame(data)


# Train and Save Model
def train_model():
    df = generate_data()
    X = df.drop(columns=["Maintenance Required"])
    y = df["Maintenance Required"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "predictive_maintenance_model.pkl")
    print("✅ Model Trained & Saved Successfully!")


if __name__ == "__main__":
    train_model()
