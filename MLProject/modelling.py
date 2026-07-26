"""
modelling.py (versi untuk MLflow Project / CI)
Melatih model Random Forest untuk prediksi Customer Churn.
Dipanggil oleh MLflow Project melalui file 'MLProject'.

CATATAN PENTING:
Saat dijalankan lewat 'mlflow run', MLflow SUDAH membuat run aktif secara
otomatis. Jadi di sini kita TIDAK perlu (dan TIDAK BOLEH) memanggil
mlflow.set_experiment() atau mlflow.start_run() secara manual lagi,
karena akan menyebabkan konflik run ID.
"""

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(data_path):
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="telco_churn_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # Aktifkan autolog -> otomatis mencatat parameter, metrik, dan model
    # ke run yang sudah aktif dari 'mlflow run'
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_data(args.data_path)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("=== Hasil Evaluasi Model (CI) ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")


if __name__ == "__main__":
    main()