"""
app.py
------
Flask REST API + dashboard for the Network Intrusion Detection System.

Endpoints
---------
GET  /                        – Dashboard (HTML)
GET  /api/health              – Health check
POST /api/predict             – Predict from JSON payload
POST /api/predict/batch       – Batch prediction from JSON array
GET  /api/metrics             – Saved model metrics
GET  /api/alerts              – Recent alerts (Cloudant or in-memory)
GET  /api/attack_distribution – Attack type distribution (for chart)
"""

import os
import json
import logging
import traceback
from datetime import datetime, timezone

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.FileHandler(os.getenv("LOG_FILE", "logs/nids.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Import NIDS modules ───────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessor import NIDSPreprocessor, ATTACK_MAP
from trainer import NIDSTrainer
from ibm_watson import WatsonMLService, CloudantLogger

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nids_secret_key")
CORS(app)

# ── Attack family colour map ──────────────────────────────────────────────────
FAMILY_COLORS = {
    "normal":  "#22c55e",
    "DoS":     "#ef4444",
    "Probe":   "#f59e0b",
    "R2L":     "#8b5cf6",
    "U2R":     "#ec4899",
    "Other":   "#6b7280",
}

# ── In-memory alert buffer (max 200) ─────────────────────────────────────────
ALERT_BUFFER: list[dict] = []

# ── Load model & preprocessor ─────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_PATH", "models/nids_model.pkl")
MODEL_DIR = os.path.dirname(MODEL_DIR) if MODEL_DIR.endswith(".pkl") else MODEL_DIR

preprocessor: NIDSPreprocessor | None = None
model = None
wml_service = WatsonMLService()
cloudant    = CloudantLogger()


def _load_artefacts():
    """Try to load saved preprocessor + model from disk."""
    global preprocessor, model
    model_dir = "models"
    pkl_path  = os.path.join(model_dir, "nids_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if os.path.exists(pkl_path) and os.path.exists(scaler_path):
        try:
            preprocessor = NIDSPreprocessor.load(model_dir)
            model = NIDSTrainer.load_model(model_dir)
            logger.info("Model & preprocessor loaded from disk.")
        except Exception as exc:
            logger.warning("Could not load artefacts: %s", exc)
    else:
        logger.warning(
            "No trained model found at %s. "
            "Run the Jupyter notebook or train.py first.", pkl_path
        )


_load_artefacts()


# =============================================================================
# Routes
# =============================================================================

@app.route("/")
def index():
    """Render the main dashboard."""
    model_ready = model is not None
    return render_template("index.html", model_ready=model_ready)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── Single prediction ─────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict the attack type for a single network connection record.

    Request body (JSON):
    {
      "duration": 0, "protocol_type": "tcp", "service": "http",
      "flag": "SF", "src_bytes": 181, "dst_bytes": 5450, ...
    }
    """
    if model is None or preprocessor is None:
        return jsonify({
            "error": "Model not loaded. Please train the model first."
        }), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty request body."}), 400

    try:
        X = preprocessor.transform_single(data)
        pred, proba = NIDSTrainer.predict(model, X)

        label      = preprocessor.decode_prediction(int(pred[0]))
        family     = ATTACK_MAP.get(label, "Other")
        confidence = float(np.max(proba[0])) if proba is not None else None
        is_attack  = family != "normal"

        result = {
            "prediction": label,
            "attack_family": family,
            "is_attack": is_attack,
            "confidence": round(confidence, 4) if confidence else None,
            "color": FAMILY_COLORS.get(family, "#6b7280"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add to in-memory alert buffer
        alert = {**result, "input": data}
        ALERT_BUFFER.append(alert)
        if len(ALERT_BUFFER) > 200:
            ALERT_BUFFER.pop(0)

        # Log to Cloudant (best-effort)
        cloudant.log_event(result)

        return jsonify(result)

    except Exception as exc:
        logger.error("Prediction error: %s\n%s", exc, traceback.format_exc())
        return jsonify({"error": str(exc)}), 500


# ── Batch prediction ──────────────────────────────────────────────────────────

@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch predict. Request body: JSON array of connection records.
    """
    if model is None or preprocessor is None:
        return jsonify({
            "error": "Model not loaded."
        }), 503

    records = request.get_json(force=True)
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "Expected a non-empty JSON array."}), 400

    results = []
    try:
        for rec in records:
            X = preprocessor.transform_single(rec)
            pred, proba = NIDSTrainer.predict(model, X)
            label  = preprocessor.decode_prediction(int(pred[0]))
            family = ATTACK_MAP.get(label, "Other")
            conf   = float(np.max(proba[0])) if proba is not None else None
            results.append({
                "prediction":    label,
                "attack_family": family,
                "is_attack":     family != "normal",
                "confidence":    round(conf, 4) if conf else None,
                "color":         FAMILY_COLORS.get(family, "#6b7280"),
            })
        return jsonify({"count": len(results), "results": results})
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ── IBM WML prediction (cloud endpoint) ──────────────────────────────────────

@app.route("/api/predict/wml", methods=["POST"])
def predict_wml():
    """
    Score a record via IBM Watson Machine Learning deployed endpoint.
    Falls back to local model if WML is unavailable.
    """
    data = request.get_json(force=True)
    if preprocessor is None:
        return jsonify({"error": "Preprocessor not loaded."}), 503

    X = preprocessor.transform_single(data).tolist()
    wml_result = wml_service.predict([X])

    if wml_result:
        return jsonify({"source": "ibm_wml", "result": wml_result})

    # Fallback to local
    if model:
        pred, proba = NIDSTrainer.predict(model, np.array(X))
        label  = preprocessor.decode_prediction(int(pred[0]))
        family = ATTACK_MAP.get(label, "Other")
        return jsonify({
            "source": "local_fallback",
            "prediction": label,
            "attack_family": family,
            "is_attack": family != "normal",
            "confidence": round(float(np.max(proba[0])), 4) if proba is not None else None,
        })

    return jsonify({"error": "No scoring endpoint available."}), 503


# ── Metrics ───────────────────────────────────────────────────────────────────

@app.route("/api/metrics", methods=["GET"])
def metrics():
    path = os.path.join("models", "metrics.json")
    if not os.path.exists(path):
        return jsonify({"error": "metrics.json not found. Train the model first."}), 404
    with open(path) as fh:
        return jsonify(json.load(fh))


# ── Alerts ────────────────────────────────────────────────────────────────────

@app.route("/api/alerts", methods=["GET"])
def alerts():
    limit = int(request.args.get("limit", 50))
    # Try Cloudant first, fall back to in-memory buffer
    cloud_alerts = cloudant.fetch_recent(limit)
    if cloud_alerts:
        return jsonify({"source": "cloudant", "alerts": cloud_alerts})
    return jsonify({
        "source": "memory",
        "alerts": list(reversed(ALERT_BUFFER))[:limit]
    })


# ── Attack distribution summary ───────────────────────────────────────────────

@app.route("/api/attack_distribution", methods=["GET"])
def attack_distribution():
    """Return counts of each attack family in recent alerts."""
    counts = {"normal": 0, "DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0, "Other": 0}
    for a in ALERT_BUFFER:
        fam = a.get("attack_family", "Other")
        counts[fam] = counts.get(fam, 0) + 1
    return jsonify(counts)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    logger.info("Starting NIDS Flask server on %s:%s", host, port)
    app.run(host=host, port=port, debug=debug, use_reloader=False)
