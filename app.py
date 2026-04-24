import math
import sys
from pathlib import Path

import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.preprocess import clean_text
from src.features import compose_features
from src.explain import explain

app = Flask(__name__, static_folder=".")
CORS(app)

MODELS_DIR = ROOT / "models"
clf_path        = MODELS_DIR / "best_classifier.pkl"
tfidf_path      = MODELS_DIR / "tfidf.pkl"
tfidf_char_path = MODELS_DIR / "tfidf_char.pkl"

if not clf_path.exists() or not tfidf_path.exists():
    print("[app] Model files not found. Please run training first.")
    sys.exit(1)

clf        = joblib.load(clf_path)
tfidf      = joblib.load(tfidf_path)
char_tfidf = joblib.load(tfidf_char_path) if tfidf_char_path.exists() else None
if char_tfidf is None:
    print("[app] tfidf_char.pkl missing — running word-only TF-IDF. Re-run training to fix.")


def _confidence(model, X, pred: int):
    """Return P(predicted_class). Falls back to sigmoid(decision_function) for LinearSVC."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        return float(probs[pred])
    if hasattr(model, "decision_function"):
        margin = float(model.decision_function(X)[0])
        signed = margin if pred == 1 else -margin
        return 1.0 / (1.0 + math.exp(-signed))
    return None


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(".", path)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        msg  = (data.get("text") or "").strip()
        if not msg:
            return jsonify({"error": "No text provided"}), 400

        clean_msg  = clean_text(msg)
        X          = compose_features([msg], [clean_msg], tfidf, char_tfidf)
        pred       = int(clf.predict(X)[0])
        confidence = _confidence(clf, X, pred)
        explanation = explain(msg)

        return jsonify({
            "prediction":      "SMISHING" if pred == 1 else "LEGIT",
            "is_smishing":     bool(pred == 1),
            "confidence":      confidence,
            "reasons":         explanation["reasons"],
            "highlights":      explanation["highlights"],
            "rule_flags":      explanation["rule_flags"],
            "heuristic_score": explanation["heuristic_score"],
            "text":            msg,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "ready",
        "model":      type(clf).__name__,
        "char_tfidf": char_tfidf is not None,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
