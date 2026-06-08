import warnings
import os
import joblib
import numpy as np
import pandas as pd
import json

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score
)
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_DIR = "saved_models"
RANDOM_STATE = 42
TARGET_COL = "Failure"
DROP_COLS = ["Record_ID", "Date", "Equipment_ID"]

FEATURE_COLS = [
    "Temperature", "Pressure", "Vibration",
    "Running_Hours", "Current", "Power_Consumption", "Oil_Level",
    "Equipment_Type"
]

ALERT_THRESHOLDS = {
    "Temperature":       {"warning": 90,  "critical": 105},
    "Vibration":         {"warning": 4.5, "critical": 5.5},
    "Running_Hours":     {"warning": 4000,"critical": 4800},
    "Oil_Level":         {"warning": 30,  "critical": 20},
    "Pressure":          {"warning": 8.5, "critical": 9.5},
    "Current":           {"warning": 35,  "critical": 39},
    "Power_Consumption": {"warning": 160, "critical": 175},
}


# ─────────────────────────────────────────────
# 1. Data Loading & Validation
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load Excel file and perform basic validation."""
    print(f"\n[1/6] Loading data from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"The file '{filepath}' was not found. "
            "Please ensure the file is uploaded to the Colab environment "
            "and the path is correct. "
            f"Current working directory: {os.getcwd()}"
        )
    df = pd.read_excel(filepath)
    print(f"      Loaded {len(df):,} rows × {df.shape[1]} columns")

    required = set(FEATURE_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# ─────────────────────────────────────────────
# 2. Feature Engineering & Preprocessing
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that improve failure prediction."""
    df = df.copy()

    df["Thermal_Stress"]   = df["Temperature"] * df["Running_Hours"] / 10_000
    df["Mech_Load"]        = df["Vibration"] * df["Current"]
    df["Power_Efficiency"] = df["Power_Consumption"] / (df["Current"] + 1e-6)
    df["Low_Oil_Flag"]     = (df["Oil_Level"] < 30).astype(int)
    df["High_Vib_Flag"]    = (df["Vibration"] > 4.5).astype(int)
    df["High_Temp_Flag"]   = (df["Temperature"] > 90).astype(int)
    df["Stress_Score"]     = df["High_Temp_Flag"] + df["High_Vib_Flag"] + df["Low_Oil_Flag"]
    df["Hours_Bin"]        = pd.cut(
        df["Running_Hours"],
        bins=[0, 1000, 3000, 5000],
        labels=[0, 1, 2]
    ).astype(int)

    return df


def preprocess(df: pd.DataFrame, le: LabelEncoder = None, fit: bool = True):
    """
    Encode categoricals, drop unused cols, split X / y.
    Returns (X, y, label_encoder, feature_names)
    """
    df = engineer_features(df)

    if fit:
        le = LabelEncoder()
        df["Equipment_Type"] = le.fit_transform(df["Equipment_Type"])
    else:
        df["Equipment_Type"] = le.transform(df["Equipment_Type"])

    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop, errors="ignore")

    y = None
    if TARGET_COL in df.columns:
        y = (df[TARGET_COL] == "Yes").astype(int)
        df = df.drop(columns=[TARGET_COL])

    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    feature_names = list(df.columns)
    return df, y, le, feature_names


# ─────────────────────────────────────────────
# 3. Model Training
# ─────────────────────────────────────────────
def train_model(X_train, y_train) -> RandomForestClassifier:
    """Train Random Forest with SMOTE for class imbalance."""
    print("\n[3/6] Handling class imbalance with SMOTE ...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"      Resampled: {y_res.value_counts().to_dict()}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\n[3/6] Training Random Forest ...", end=" ", flush=True)
    model.fit(X_res, y_res)
    print("done")

    return model


# ─────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────
def evaluate_model(model: RandomForestClassifier, X_test, y_test):
    """Evaluate the Random Forest model on the test set."""
    print("\n[4/6] Evaluating model on test set ...")
    print("=" * 72)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n  ▶ Random Forest")
    print(f"    ROC-AUC : {roc:.4f}")
    print(f"    Avg Prec: {ap:.4f}  |  F1: {f1:.4f}")
    print(f"    Confusion Matrix (TN FP / FN TP):")
    print(f"      {cm[0]}  /  {cm[1]}")
    print(classification_report(y_test, y_pred,
          target_names=["No Failure", "Failure"], digits=4))
    print("=" * 72)


# ─────────────────────────────────────────────
# 5. Feature Importance
# ─────────────────────────────────────────────
def feature_importance_report(model, X_test, y_test, feature_names):
    """Print top-15 most predictive features using permutation importance."""
    print("\n[5/6] Computing feature importance ...")
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    idx = np.argsort(perm.importances_mean)[::-1][:15]
    print("\n  Top 15 features (permutation importance):")
    print(f"  {'Feature':<25} {'Mean':<10} {'Std'}")
    print("  " + "-" * 45)
    for i in idx:
        print(f"  {feature_names[i]:<25} {perm.importances_mean[i]:.4f}     "
              f"± {perm.importances_std[i]:.4f}")


# ─────────────────────────────────────────────
# 6. Save / Load
# ─────────────────────────────────────────────
def save_artifacts(model, le, feature_names):
    os.makedirs(MODEL_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODEL_DIR, f"random_forest_{ts}.pkl")
    joblib.dump({
        "model": model,
        "label_encoder": le,
        "feature_names": feature_names,
        "trained_at": ts,
    }, path)
    print(f"\n[6/6] Saved model artifacts → {path}")
    return path


def load_artifacts(path: str):
    data = joblib.load(path)
    return data["model"], data["label_encoder"], data["feature_names"]


# ─────────────────────────────────────────────
# 7. Prediction + Alert Generation
# ─────────────────────────────────────────────
def generate_alerts(row: pd.Series) -> list[dict]:
    """Return list of human-readable alert strings for a single record."""
    alerts = []
    for col, thresholds in ALERT_THRESHOLDS.items():
        if col not in row:
            continue
        val = row[col]
        if col == 'Oil_Level': # Special handling for Oil_Level (lower is worse)
            if val <= thresholds["critical"]:
                alerts.append({"level": "crit", "msg": f"{col}={val:.1f}"})
            elif val <= thresholds["warning"]:
                alerts.append({"level": "warn", "msg": f"{col}={val:.1f}"})
        else: # For other metrics (higher is worse)
            if val >= thresholds["critical"]:
                alerts.append({"level": "crit", "msg": f"{col}={val:.1f}"})
            elif val >= thresholds["warning"]:
                alerts.append({"level": "warn", "msg": f"{col}={val:.1f}"})
    return alerts


def predict_failure(df: pd.DataFrame, model, le, feature_names) -> pd.DataFrame:
    """
    For new data, return original df with added columns:
      - failure_probability  (0.0 – 1.0)
      - risk_level           (Low / Medium / High / Critical)
      - alerts               (list of triggered threshold alerts)
    """
    X, _, _, _ = preprocess(df, le=le, fit=False)
    X = X[feature_names]

    probs = model.predict_proba(X)[:, 1]

    def risk_level(p):
        if p < 0.25: return "Low"
        if p < 0.50: return "Medium"
        if p < 0.75: return "High"
        return "Critical"

    out = df.copy()
    out["_prob"] = np.round(probs, 4)
    out["_risk"] = [risk_level(p) for p in probs]
    # Serialize alerts to JSON string for easier handling in CSV/JS
    out["_alerts"] = [json.dumps(generate_alerts(row)) for _, row in df.iterrows()]
    return out.sort_values("_prob", ascending=False)


# ─────────────────────────────────────────────
# Pipelines
# ─────────────────────────────────────────────
def train_pipeline(data_path: str):
    df = load_data(data_path)

    print("\n[2/6] Preprocessing & feature engineering ...")
    X, y, le, feature_names = preprocess(df, fit=True)
    print(f"      Feature matrix: {X.shape}  |  Features: {len(feature_names)}")
    print(f"      Class balance  — No Failure: {(y==0).sum():,}  "
          f"| Failure: {(y==1).sum():,}  ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    feature_importance_report(model, X_test, y_test, feature_names)

    path = save_artifacts(model, le, feature_names)
    return path, model, le, feature_names


def predict_pipeline(data_path: str, model_path: str):
    model, le, feature_names = load_artifacts(model_path)
    df = load_data(data_path)
    results = predict_failure(df, model, le, feature_names)

    out_path = "predictions_output.xlsx"
    results.to_excel(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}")

    print("\n── Top 10 highest risk equipment ──")
    cols = ["Equipment_ID", "Equipment_Type", "_prob", "_risk", "_alerts"]
    available = [c for c in cols if c in results.columns]
    print(results[available].head(10).to_string(index=False))
    return results


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running in demo mode (train + predict on sample file) ...")
    data_file = "equipment_maintenance_data.xlsx"

    model_path, model, le, feature_names = train_pipeline(data_file)
    df = pd.read_excel(data_file)
    sample = df.sample(200, random_state=42)
    results = predict_failure(sample, model, le, feature_names)

    print("\n── Sample predictions ──")
    cols = ["Equipment_ID", "Equipment_Type", "_prob", "_risk", "_alerts"]
    available = [c for c in cols if c in results.columns]
    print(results[available].head(10).to_string(index=False))