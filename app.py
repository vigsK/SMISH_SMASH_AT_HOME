import sys
from pathlib import Path
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.preprocess import clean_text
from src.features import compose_features

app = Flask(__name__, static_folder=".")
CORS(app)

MODELS_DIR = ROOT / "models"
clf_path = MODELS_DIR / "best_classifier.pkl"
tfidf_path = MODELS_DIR / "tfidf.pkl"

if not clf_path.exists() or not tfidf_path.exists():
    print("[app] Model files not found. Please run training first.")
    sys.exit(1)

clf = joblib.load(clf_path)
tfidf = joblib.load(tfidf_path)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(".", path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
        
        msg = data["text"]
        clean_msg = clean_text(msg)
        
        # compose_features expects lists
        X = compose_features([msg], [clean_msg], tfidf)
        pred = clf.predict(X)[0]
        
        # Get probability if available
        confidence = None
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[0]
            confidence = float(probs[pred])
        elif hasattr(clf, "decision_function"):
            # LinearSVC doesn't have predict_proba by default
            # We can use decision_function as a proxy or just leave it
            pass

        return jsonify({
            "prediction": "SMISHING" if pred == 1 else "LEGIT",
            "is_smishing": bool(pred == 1),
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
