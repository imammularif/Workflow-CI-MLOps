import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os

# 1. Load Data
df = pd.read_csv('heart_disease_preprocessed.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Jalankan Training
# Gunakan nested=True agar sinkron dengan terminal
with mlflow.start_run(nested=True):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # --- INI KUNCINYA ---
    # Kita paksa log model ke folder bernama 'model'
    mlflow.sklearn.log_model(sk_model=rf, artifact_path="model") 
    
    # Log metrik manual
    accuracy = rf.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    print("-" * 30)
    print(f"Model berhasil disimpan secara manual!")
    print("-" * 30)