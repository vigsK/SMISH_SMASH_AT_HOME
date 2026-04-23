"""
train.py — End-to-end training pipeline for the smishing classifier.

Trains four lightweight classifiers (Naïve Bayes, Logistic Regression,
Random Forest, Linear SVM) on hybrid TF-IDF + rule + heuristic features,
prints a metrics table, saves comparison figures, and persists the best model.

Run: python -m src.train  (from project root)

Echoes Seo et al. (2024): classifier comparison methodology.
"""

import sys
import time
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — safe on Colab and servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# Project-local imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import load_sms, clean_text
from src.features import build_tfidf, compose_features, RuleBasedClassifier

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Classifier definitions
# ---------------------------------------------------------------------------

CLASSIFIERS = {
    # Oldest baseline: pure rule threshold, no learned text features (Jain & Gupta 2018)
    "Rule-Based (2018)":  RuleBasedClassifier(threshold=0.25),
    "Naïve Bayes":        MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest":      RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
    ),
    "Linear SVM":         LinearSVC(class_weight="balanced", random_state=42),
}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _score(clf, X, y):
    """Return accuracy, precision, recall, F1 for the spam class (label=1)."""
    preds = clf.predict(X)
    return {
        "accuracy":  round(accuracy_score(y, preds),  4),
        "precision": round(precision_score(y, preds, pos_label=1, zero_division=0), 4),
        "recall":    round(recall_score(y, preds, pos_label=1, zero_division=0),    4),
        "f1":        round(f1_score(y, preds, pos_label=1, zero_division=0),        4),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_bars(results_df: pd.DataFrame, out_path: Path) -> None:
    """Save grouped bar chart comparing all classifiers across four metrics."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    n_clf   = len(results_df)
    x       = np.arange(len(metrics))
    width   = 0.15  # fits 5 classifiers without overlap

    fig, ax = plt.subplots(figsize=(10, 5))
    # Distinct grey for the rule-based baseline; ML models use Set2 palette.
    colors  = ["#b0b0b0"] + list(sns.color_palette("Set2", max(n_clf - 1, 1)))

    for i, (clf_name, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=clf_name, color=colors[i])

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Classifier Comparison — Smishing Detection")
    ax.set_xticks(x + width * (n_clf - 1) / 2)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0.60, 1.02)  # show the rule-based gap clearly
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[train] Saved {out_path}")


def _plot_confusion(clf, X_test, y_test, clf_name: str, out_path: Path) -> None:
    """Save annotated confusion matrix for the best classifier."""
    preds = clf.predict(X_test)
    cm    = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {clf_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[train] Saved {out_path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train():
    """Full training pipeline: load → clean → featurize → train → evaluate → save.

    Steps
    -----
    1. Load SMS dataset (auto-downloads if missing).
    2. Clean text and build TF-IDF vocabulary on the training split only.
    3. Compose hybrid feature matrix [TF-IDF | rule flags | heuristic score].
    4. Train four classifiers with fixed random_state=42.
    5. Print results table; save fig3_bars.png and fig4_cm.png.
    6. Persist best-F1 classifier and fitted TF-IDF vectorizer via joblib.
    """
    t0 = time.time()

    # 1. Load data
    print("[train] Loading dataset …")
    df = load_sms()
    print(f"[train] {len(df)} messages  |  spam: {df['y'].sum()}  ham: {(df['y']==0).sum()}")

    # 2. Clean
    print("[train] Cleaning text …")
    df["clean"] = df["text"].apply(clean_text)

    # 3. Stratified split
    X_raw_tr, X_raw_te, X_cl_tr, X_cl_te, y_tr, y_te = train_test_split(
        df["text"].tolist(),
        df["clean"].tolist(),
        df["y"].tolist(),
        test_size=0.20,
        random_state=42,
        stratify=df["y"],
    )

    # 4. Fit TF-IDF on train clean text
    print("[train] Fitting TF-IDF …")
    tfidf = build_tfidf()
    tfidf.fit(X_cl_tr)

    # 5. Compose feature matrices
    print("[train] Composing features …")
    X_train = compose_features(X_raw_tr, X_cl_tr, tfidf)
    X_test  = compose_features(X_raw_te, X_cl_te, tfidf)
    print(f"[train] Feature matrix shape: {X_train.shape}")

    # 6. Train & evaluate
    results  = {}
    trained  = {}
    for name, clf in CLASSIFIERS.items():
        print(f"[train] Training {name} …", end=" ", flush=True)
        t1 = time.time()
        clf.fit(X_train, y_tr)
        trained[name] = clf
        metrics = _score(clf, X_test, y_te)
        results[name] = metrics
        print(f"done ({time.time()-t1:.1f}s)  F1={metrics['f1']}")

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Classifier"

    print("\n" + "=" * 60)
    print("RESULTS TABLE (spam class metrics)")
    print("=" * 60)
    print(results_df.to_string())
    print("=" * 60 + "\n")

    # 7. Plots
    _plot_bars(results_df, ROOT / "models" / "fig3_bars.png")

    best_name = results_df["f1"].idxmax()
    best_clf  = trained[best_name]
    print(f"[train] Best classifier by F1: {best_name} (F1={results_df.loc[best_name,'f1']})")
    _plot_confusion(best_clf, X_test, y_te, best_name, ROOT / "models" / "fig4_cm.png")

    # 8. Persist
    joblib.dump(best_clf, MODELS_DIR / "best_classifier.pkl")
    joblib.dump(tfidf,    MODELS_DIR / "tfidf.pkl")
    print(f"[train] Saved models to {MODELS_DIR}")
    print(f"[train] Total time: {time.time()-t0:.1f}s")

    return results_df, best_name


if __name__ == "__main__":
    train()
