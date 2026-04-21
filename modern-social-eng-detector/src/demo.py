"""
demo.py — Live inference on adversarial-style SMS samples.

Loads the persisted best classifier and TF-IDF vectorizer, runs prediction
on three hand-crafted test messages, and prints labelled results.

Run: python -m src.demo  (from project root)
"""

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import clean_text
from src.features import compose_features

MODELS_DIR = ROOT / "models"

# ---------------------------------------------------------------------------
# Adversarial test samples
# ---------------------------------------------------------------------------
SAMPLES = [
    # 1. Obvious smishing: URL + prize + urgency + currency
    (
        "URGENT: You've WON a £500 Amazon gift card! "
        "Claim NOW before it EXPIRES: http://amaz0n-reward.tk/claim?id=9921 "
        "Reply STOP to opt out."
    ),
    # 2. Benign: plain conversational message, no indicators
    (
        "Hey, are you coming to the study session tomorrow at 3pm? "
        "Let me know if you need the lecture notes."
    ),
    # 3. Urgent-tone smishing: OTP + bank account suspension angle
    (
        "Your SBI bank account has been SUSPENDED due to suspicious activity. "
        "Verify your KYC immediately or your account will be blocked. "
        "Call 9876543210 or visit http://sbi-kyc-verify.net"
    ),
]


def run_demo():
    """Load saved model artefacts and classify the three sample messages.

    For each sample, prints  ``[SMISHING]`` or ``[HAM]`` followed by a
    truncated preview of the message.  Mirrors the inference pipeline used
    during training: raw text → compose_features → predict.
    """
    clf_path   = MODELS_DIR / "best_classifier.pkl"
    tfidf_path = MODELS_DIR / "tfidf.pkl"

    if not clf_path.exists() or not tfidf_path.exists():
        print("[demo] Model files not found. Run  python -m src.train  first.")
        sys.exit(1)

    clf   = joblib.load(clf_path)
    tfidf = joblib.load(tfidf_path)

    clean_msgs = [clean_text(m) for m in SAMPLES]

    X = compose_features(raw_msgs, clean_msgs, tfidf)
    preds = clf.predict(X)

    print("\n" + "=" * 70)
    print("DEMO INFERENCE RESULTS")
    print("=" * 70)
    for msg, pred in zip(raw_msgs, preds):
        label   = "SMISHING" if pred == 1 else "HAM"
        preview = msg[:80] + ("…" if len(msg) > 80 else "")
        print(f"[{label:<8}]  {preview}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
