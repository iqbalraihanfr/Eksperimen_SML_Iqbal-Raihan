"""
Automasi preprocessing dataset Pima Indians Diabetes.
Konversi dari notebook eksperimen -> fungsi otomatis yang mengembalikan data siap latih.

Basic K1 tercapai lewat notebook. File ini disertakan agar struktur repo lengkap
dan konsisten dengan langkah eksperimen (data loading -> EDA cleaning -> preprocessing).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Kolom yang secara medis tidak mungkin bernilai 0 -> 0 dianggap missing
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path):
    """Data loading: baca CSV mentah."""
    return pd.read_csv(path)


def clean_data(df):
    """Ganti 0 tidak valid dengan median kolom (hasil temuan EDA)."""
    df = df.copy()
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df


def preprocess_data(path, test_size=0.2, random_state=42, save_dir=None):
    """
    Pipeline lengkap: load -> clean -> split -> scaling.
    Mengembalikan X_train, X_test, y_train, y_test yang siap dilatih.
    """
    df = load_data(path)
    df = clean_data(df)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        train = X_train.copy(); train["Outcome"] = y_train.values
        test = X_test.copy();  test["Outcome"] = y_test.values
        train.to_csv(os.path.join(save_dir, "diabetes_train.csv"), index=False)
        test.to_csv(os.path.join(save_dir, "diabetes_test.csv"), index=False)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    Xtr, Xte, ytr, yte = preprocess_data(
        "../diabetes_raw.csv", save_dir="diabetes_preprocessing"
    )
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")
    print("Preprocessing selesai. File tersimpan di diabetes_preprocessing/")
