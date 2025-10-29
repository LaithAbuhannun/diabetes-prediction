from pathlib import Path
from datetime import datetime, timezone
import csv
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from joblib import load  # loads the StandardScaler from Colab


# ── Paths / constants ─────────────────────────────────
# HERE = /.../diabetes/server
HERE = Path(__file__).resolve().parent

ARTIFACTS = HERE / "artifacts"
MODEL_PATH = ARTIFACTS / "my_model.keras"
SCALER_PATH = ARTIFACTS / "scaler.joblib"

LOG_PATH = HERE / "prediction.csv"   # stays inside /server

# this is where interface.html lives: one directory up from server/
PROJECT_ROOT = HERE.parent
FRONTEND_HTML = PROJECT_ROOT / "interface.html"

# MUST match training order exactly
FEATURES = ["Pregnancies", "Glucose", "BMI", "Age"]

THRESHOLD = 0.5  # classify = 1 if prob >= 0.5


# ── App setup ─────────────────────────────────────────
app = Flask(__name__)
CORS(app)

MODEL = load_model(MODEL_PATH)
SCALER = load(SCALER_PATH)  # StandardScaler() you fitted in Colab


# ── Helpers ───────────────────────────────────────────
def preprocess_payload(payload_dict):
    """
    Take raw JSON:
    {
        "Pregnancies": ...,
        "Glucose": ...,
        "BMI": ...,
        "Age": ...
    }

    1. Put in the correct feature order
    2. Convert to float32
    3. Apply SAME scaler used in training
    4. Return shape (1,4) for model.predict()
    """

    row = [[
        float(payload_dict["Pregnancies"]),
        float(payload_dict["Glucose"]),
        float(payload_dict["BMI"]),
        float(payload_dict["Age"]),
    ]]

    row = np.array(row, dtype="float32")        # shape (1,4)
    row_scaled = SCALER.transform(row).astype("float32")
    return row_scaled


def _log_prediction(payload, prob, predicted_class):
    """
    Append the prediction into prediction.csv in /server.

    Columns:
    timestamp_utc, Pregnancies, Glucose, BMI, Age, probability, predicted_class
    """
    newfile = not LOG_PATH.exists()

    with LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "Pregnancies",
                "Glucose",
                "BMI",
                "Age",
                "probability",
                "predicted_class"
            ],
        )
        if newfile:
            writer.writeheader()

        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "Pregnancies": payload["Pregnancies"],
            "Glucose": payload["Glucose"],
            "BMI": payload["BMI"],
            "Age": payload["Age"],
            "probability": round(prob, 6),
            "predicted_class": predicted_class
        })


def _read_log_rows():
    """
    Read all rows from prediction.csv (or [] if it doesn't exist yet).
    Each row is a dict with keys matching the header above.
    """
    if not LOG_PATH.exists():
        return []

    rows = []
    with LOG_PATH.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


# ── Routes ────────────────────────────────────────────

# 1. Serve the frontend HTML at "/"
@app.get("/")
def root_page():
    # We can't just return send_from_directory(HERE, "interface.html")
    # because interface.html is in PROJECT_ROOT, not HERE.
    #
    # send_from_directory needs a directory path, not the full file path.
    return send_from_directory(PROJECT_ROOT, "interface.html")


# 2. Prediction endpoint hit by your "Predict" button in JS
@app.post("/predict")
def predict():
    try:
        payload = request.get_json()

        # Scale input
        x_scaled = preprocess_payload(payload)

        # Model inference
        yhat = MODEL.predict(x_scaled)
        prob = float(yhat[0][0])  # sigmoid output
        predicted_class = 1 if prob >= THRESHOLD else 0

        # Log for metrics
        _log_prediction(payload, prob, predicted_class)

        return jsonify({
            "probability": prob,
            "predicted_class": predicted_class
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# 3. Summary metrics endpoint → frontend calls fetch("/metrics")
@app.get("/metrics")
def metrics_summary():
    """
    Returns totals and averages for the summary cards + bar chart.
    {
      "total": ...,
      "diabetics": ...,
      "non_diabetics": ...,
      "avg_probability": ...
    }
    """
    rows = _read_log_rows()
    if not rows:
        return jsonify({
            "total": 0,
            "diabetics": 0,
            "non_diabetics": 0,
            "avg_probability": 0.0
        })

    total = 0
    diabetics = 0
    non_diabetics = 0
    prob_sum = 0.0

    for r in rows:
        total += 1
        cls = int(r["predicted_class"])
        p = float(r["probability"])
        prob_sum += p
        if cls == 1:
            diabetics += 1
        else:
            non_diabetics += 1

    avg_probability = prob_sum / total if total > 0 else 0.0

    return jsonify({
        "total": total,
        "diabetics": diabetics,
        "non_diabetics": non_diabetics,
        "avg_probability": avg_probability
    })


# 4. Time-series metrics endpoint → frontend calls fetch("/metrics/timeseries")
@app.get("/metrics/timeseries")
def metrics_timeseries():
    """
    Returns a list like:
    [
      { "date": "2025-10-25", "diabetics": 3, "non_diabetics": 5 },
      ...
    ]

    We group by calendar day (UTC) using the timestamps we stored.
    """
    rows = _read_log_rows()
    if not rows:
        return jsonify([])

    daily = {}  # day -> { "diabetics": x, "non_diabetics": y }

    for r in rows:
        # parse timestamp_utc
        ts_raw = r["timestamp_utc"]
        # handle "Z" vs +00:00 just in case
        if ts_raw.endswith("Z"):
            ts_raw = ts_raw.replace("Z", "+00:00")
        try:
            ts = datetime.fromisoformat(ts_raw)
        except Exception:
            # skip bad rows quietly
            continue

        day = ts.date().isoformat()  # 'YYYY-MM-DD'

        if day not in daily:
            daily[day] = {"diabetics": 0, "non_diabetics": 0}

        cls = int(r["predicted_class"])
        if cls == 1:
            daily[day]["diabetics"] += 1
        else:
            daily[day]["non_diabetics"] += 1

    # turn dict into sorted list
    out = []
    for day in sorted(daily.keys()):
        out.append({
            "date": day,
            "diabetics": daily[day]["diabetics"],
            "non_diabetics": daily[day]["non_diabetics"],
        })

    return jsonify(out)


# ── Run dev server ────────────────────────────────────
if __name__ == "__main__":
    # Run from inside /server directory:
    # (venv) PS .../diabetes/server> python server.py
    app.run(host="127.0.0.1", port=5000, debug=True)
