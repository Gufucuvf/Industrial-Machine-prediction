"""
Predictive Maintenance — Flask Backend
=======================================
Fixed version: feature engineering exactly matches the trained RF model.

Endpoints:
    GET  /                → serves index.html dashboard
    GET  /health          → health check
    POST /predict-file    → upload Excel/CSV, get predictions back
    POST /predict         → send JSON rows, get predictions back

Run locally:
    pip install flask flask-cors joblib pandas scikit-learn imbalanced-learn openpyxl
    python app.py

Run in Colab:
    !pip install flask flask-cors pyngrok -q
    from pyngrok import ngrok
    import threading, subprocess
    threading.Thread(target=lambda: subprocess.run(["python","app.py"])).start()
    print(ngrok.connect(5000))
"""

import io
import os
import re

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

MODEL_DIR = "saved_models"

# ══════════════════════════════════════════════════════════
# CONSTANTS  — must match training exactly
# ══════════════════════════════════════════════════════════

# Columns that exist in the raw file but are NOT model features
DROP_COLS = ["Record_ID", "Date", "Equipment_ID", "Failure"]

# Raw sensor columns the uploaded file must contain
REQUIRED_COLS = [
    "Equipment_Type",
    "Temperature",
    "Pressure",
    "Vibration",
    "Running_Hours",
    "Current",
    "Power_Consumption",
    "Oil_Level",
]

ALERT_THRESHOLDS = {
    "Temperature":       {"warning": 90,   "critical": 105},
    "Vibration":         {"warning": 4.5,  "critical": 5.5},
    "Running_Hours":     {"warning": 4000, "critical": 4800},
    "Oil_Level":         {"warning": 30,   "critical": 20},
    "Pressure":          {"warning": 8.5,  "critical": 9.5},
    "Current":           {"warning": 35,   "critical": 39},
    "Power_Consumption": {"warning": 160,  "critical": 175},
}


# ══════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════

def pick_latest_model(model_dir: str) -> str:
    """Pick the most recently saved .pkl from saved_models/"""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pkls = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".pkl")
    ]
    if not pkls:
        raise FileNotFoundError(f"No .pkl files found in {model_dir}")

    # Sort by filename timestamp — newest last
    pkls.sort()
    return pkls[-1]


MODEL_PATH    = pick_latest_model(MODEL_DIR)
bundle        = joblib.load(MODEL_PATH)
model         = bundle["model"]
le            = bundle["label_encoder"]
feature_names = bundle["feature_names"]   # exact list saved during training
THRESHOLD     = bundle.get("threshold", 0.50)

print(f"✅ Loaded model  : {MODEL_PATH}")
print(f"   Model type    : {bundle.get('model_type','Unknown')}")
print(f"   Features ({len(feature_names)}): {feature_names}")
print(f"   Threshold     : {THRESHOLD}")


# ══════════════════════════════════════════════════════════
# FEATURE ENGINEERING  — identical to training pipeline
# ══════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce every feature created during model training.
    Any mismatch here = wrong predictions.
    """
    df = df.copy()

    # ── Continuous interaction features ──────────────────
    df["Thermal_Stress"]     = df["Temperature"] * df["Running_Hours"] / 10_000
    df["Mech_Load"]          = df["Vibration"] * df["Current"]
    df["Power_Efficiency"]   = df["Power_Consumption"] / (df["Current"] + 1e-6)
    df["Temp_Vib_Product"]   = df["Temperature"] * df["Vibration"]        # ← was missing
    df["Pressure_Oil_Ratio"] = df["Pressure"] / (df["Oil_Level"] + 1e-6)  # ← was missing

    # ── Binary risk flags ─────────────────────────────────
    df["Low_Oil_Flag"]    = (df["Oil_Level"]     < 30  ).astype(int)
    df["High_Vib_Flag"]   = (df["Vibration"]     > 4.5 ).astype(int)
    df["High_Temp_Flag"]  = (df["Temperature"]   > 90  ).astype(int)
    df["High_Hours_Flag"] = (df["Running_Hours"] > 4000).astype(int)      # ← was missing

    # ── Combined stress score ─────────────────────────────
    df["Stress_Score"] = (
        df["Low_Oil_Flag"] + df["High_Vib_Flag"] +
        df["High_Temp_Flag"] + df["High_Hours_Flag"]                       # ← was 3 flags, now 4
    )

    # ── Running hours wear bucket ─────────────────────────
    df["Hours_Bin"] = pd.cut(
        df["Running_Hours"],
        bins=[0, 1000, 3000, 5000],
        labels=[0, 1, 2]
    ).astype(int)

    return df


def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Validate required columns exist
      2. Coerce numeric columns
      3. Engineer features
      4. Encode Equipment_Type
      5. Drop admin columns
      6. Reindex to exact feature_names order
    """
    # 1. Check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded file is missing columns: {missing}\n"
            f"File has: {list(df.columns)}"
        )

    df = df.copy()

    # 2. Coerce numeric sensor columns
    numeric_cols = [
        "Temperature", "Pressure", "Vibration",
        "Running_Hours", "Current", "Power_Consumption", "Oil_Level",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill any NaN with column median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Encode Equipment_Type safely
    known_classes = set(map(str, le.classes_))
    df["Equipment_Type"] = df["Equipment_Type"].astype(str).map(
        lambda x: x if x in known_classes else str(le.classes_[0])
    )
    df["Equipment_Type"] = le.transform(df["Equipment_Type"])

    # 5. Drop admin/target columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # 6. Reindex to exact training feature order (fills 0 for any unseen column)
    X = df.reindex(columns=feature_names, fill_value=0)
    return X


# ══════════════════════════════════════════════════════════
# RISK & ALERT HELPERS
# ══════════════════════════════════════════════════════════

def risk_level(prob: float) -> str:
    if prob < 0.25: return "Low"
    if prob < 0.50: return "Medium"
    if prob < 0.75: return "High"
    return "Critical"


def generate_alerts(row: pd.Series) -> list:
    alerts = []
    for col, limits in ALERT_THRESHOLDS.items():
        if col not in row:
            continue
        try:
            value = float(row[col])
        except (TypeError, ValueError):
            continue

        # Oil_Level: low is bad
        if col == "Oil_Level":
            if value <= limits["critical"]:
                alerts.append({"level": "critical", "msg": f"{col} = {value:.1f} (critical ≤ {limits['critical']})"})
            elif value <= limits["warning"]:
                alerts.append({"level": "warning",  "msg": f"{col} = {value:.1f} (warning ≤ {limits['warning']})"})
        else:
            if value >= limits["critical"]:
                alerts.append({"level": "critical", "msg": f"{col} = {value:.1f} (critical ≥ {limits['critical']})"})
            elif value >= limits["warning"]:
                alerts.append({"level": "warning",  "msg": f"{col} = {value:.1f} (warning ≥ {limits['warning']})"})
    return alerts


# ══════════════════════════════════════════════════════════
# CORE PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════

def run_prediction(raw_df: pd.DataFrame) -> list:
    """
    Takes raw uploaded DataFrame → returns list of result dicts.
    Each dict has the original row data + ML predictions + alerts.
    """
    X     = preprocess_for_prediction(raw_df)
    probs = model.predict_proba(X)[:, 1]

    results = []
    for i, prob in enumerate(probs):
        row  = raw_df.iloc[i]
        prob = float(prob)
        results.append({
            # identifiers (present if columns exist)
            "Equipment_ID":          str(row.get("Equipment_ID",   f"EQ-{i+1}")),
            "Equipment_Type":        str(row.get("Equipment_Type", "Unknown")),
            # raw sensor readings
            "Temperature":           round(float(row.get("Temperature",       0)), 2),
            "Pressure":              round(float(row.get("Pressure",           0)), 2),
            "Vibration":             round(float(row.get("Vibration",          0)), 2),
            "Running_Hours":         round(float(row.get("Running_Hours",      0)), 1),
            "Current":               round(float(row.get("Current",            0)), 2),
            "Power_Consumption":     round(float(row.get("Power_Consumption",  0)), 2),
            "Oil_Level":             round(float(row.get("Oil_Level",          0)), 2),
            # ML output
            "failure_probability":   round(prob, 4),
            "failure_probability_pct": f"{prob*100:.1f}%",
            "predicted_failure":     "Yes" if prob >= THRESHOLD else "No",
            "risk_level":            risk_level(prob),
            # rule-based alerts
            "alerts":                generate_alerts(row),
            "alert_count":           len(generate_alerts(row)),
        })

    # Sort: Critical first, then by probability descending
    risk_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    results.sort(key=lambda r: (risk_order[r["risk_level"]], -r["failure_probability"]))
    return results


# ══════════════════════════════════════════════════════════
# SUMMARY HELPER
# ══════════════════════════════════════════════════════════

def build_summary(results: list) -> dict:
    total     = len(results)
    n_crit    = sum(1 for r in results if r["risk_level"] == "Critical")
    n_high    = sum(1 for r in results if r["risk_level"] == "High")
    n_medium  = sum(1 for r in results if r["risk_level"] == "Medium")
    n_low     = sum(1 for r in results if r["risk_level"] == "Low")
    n_fail    = sum(1 for r in results if r["predicted_failure"] == "Yes")
    avg_prob  = np.mean([r["failure_probability"] for r in results]) if results else 0
    return {
        "total":          total,
        "predicted_fail": n_fail,
        "critical":       n_crit,
        "high":           n_high,
        "medium":         n_medium,
        "low":            n_low,
        "avg_probability": round(float(avg_prob), 4),
        "fleet_health_pct": round((1 - n_fail / total) * 100, 1) if total else 100,
    }


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

@app.get("/")
def home():
    # Serves templates/index.html if it exists, otherwise returns JSON
    if os.path.exists("templates/index.html"):
        return render_template("index.html")
    return jsonify({
        "service": "Predictive Maintenance API",
        "model":   bundle.get("model_type", "Unknown"),
        "endpoints": {
            "POST /predict-file": "Upload Excel/CSV file",
            "POST /predict":      "Send JSON rows",
            "GET  /health":       "Health check",
        }
    })


@app.get("/health")
def health():
    return jsonify({
        "ok":         True,
        "model_path": MODEL_PATH,
        "model_type": bundle.get("model_type", "Unknown"),
        "features":   len(feature_names),
        "threshold":  THRESHOLD,
    })


# ──────────────────────────────────────────────
# POST /predict-file   (Excel or CSV upload)
# ──────────────────────────────────────────────
@app.post("/predict-file")
def predict_file():
    try:
        # Validate file was sent
        if "file" not in request.files:
            return jsonify({
                "error": (
                    "No file received. "
                    "In Postman: Body → form-data → key='file' (type=File) → select your .xlsx or .csv"
                )
            }), 400

        file     = request.files["file"]
        filename = file.filename.lower()
        raw      = file.read()

        # Parse file
        if filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
        else:
            return jsonify({"error": "Only .xlsx, .xls, or .csv files are supported."}), 400

        if df.empty:
            return jsonify({"error": "The uploaded file is empty."}), 400

        print(f"predict-file: {file.filename} — {len(df)} rows, columns: {list(df.columns)}")

        # Run prediction
        results = run_prediction(df)
        summary = build_summary(results)

        return jsonify({
            "filename": file.filename,
            "summary":  summary,
            "rows":     results,
        })

    except ValueError as ve:
        # Missing columns, bad data, etc.
        return jsonify({"error": str(ve)}), 422

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# POST /predict   (JSON body)
# ──────────────────────────────────────────────
@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        rows    = payload.get("rows")

        if not isinstance(rows, list) or not rows:
            return jsonify({
                "error": "Body must be { 'rows': [ { ...equipment record... }, ... ] }"
            }), 400

        df      = pd.DataFrame(rows)
        results = run_prediction(df)
        summary = build_summary(results)

        return jsonify({"summary": summary, "rows": results})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n🚀 Starting server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)