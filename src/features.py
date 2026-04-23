"""
features.py — Three-branch feature engineering for smishing detection.

Branch 1 (rule flags)    — Jain & Gupta (2018): 9 binary IF-THEN rules.
Branch 2 (heuristic score) — Jayaprakash et al. (2024): weighted dot product.
Branch 3 (TF-IDF)        — Seo et al. (2024): n-gram bag-of-words baseline.
"""

import re
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from .preprocess import URL_RE, PHONE_RE, MONEY_RE

# ---------------------------------------------------------------------------
# Curated suspicious keyword set (Jain & Gupta Table 2 + extensions)
# ---------------------------------------------------------------------------
SUSPICIOUS = {
    "win", "winner", "prize", "free", "claim", "urgent", "verify",
    "bank", "account", "otp", "suspended", "click", "offer", "cash",
    "loan", "gift", "congrats", "selected", "limited", "expire",
    "password", "confirm", "update", "bonus", "reward", "lucky",
    "credit", "debit", "alert", "important", "immediately", "kyc",
    "blocked", "reactivate", "transaction", "approve", "activate",
}

# Heuristic weights aligned to feature order (Jayaprakash et al. §4.2)
WEIGHTS = np.array([0.18, 0.05, 0.15, 0.10, 0.20, 0.07, 0.08, 0.07, 0.10])

# Regex helpers for rule features
_MATH_RE   = re.compile(r"[%^*/=]")
_REPLY_RE  = re.compile(r"\b(?:reply|text\s|call\s)", re.IGNORECASE)
_LEET_RE   = re.compile(r"[0-9@#$!]{2,}")
_EMAIL_RE  = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


# ---------------------------------------------------------------------------
# Branch 1: Rule features
# ---------------------------------------------------------------------------

def rule_features(msg: str) -> np.ndarray:
    """Extract 9 binary rule flags from a raw SMS message.

    Implements the IF-THEN detection rules from Jain & Gupta (2018), Table 1.
    Rules operate on the *raw* (not cleaned) text so that URLs, phone numbers,
    and currency symbols are still present.

    Parameters
    ----------
    msg : str
        Original (uncleaned) SMS text.

    Returns
    -------
    np.ndarray
        Shape (9,), dtype float32.  Each element is 0.0 or 1.0.

        f1 — URL present
        f2 — Math/operator symbols present
        f3 — Currency sign present
        f4 — Phone number present
        f5 — Suspicious keyword present
        f6 — Message length > 150 characters
        f7 — Self-answering call-to-action ("reply", "text ", "call ")
        f8 — Leet-speak / visual morphemes (2+ consecutive [0-9@#$!])
        f9 — Embedded email address
    """
    lower = msg.lower()
    words = set(re.findall(r"\b\w+\b", lower))

    f1 = float(bool(URL_RE.search(msg)))
    f2 = float(bool(_MATH_RE.search(msg)))
    f3 = float(bool(MONEY_RE.search(msg)))
    f4 = float(bool(PHONE_RE.search(msg)))
    f5 = float(bool(words & SUSPICIOUS))
    f6 = float(len(msg) > 150)
    f7 = float(bool(_REPLY_RE.search(msg)))
    f8 = float(bool(_LEET_RE.search(msg)))
    f9 = float(bool(_EMAIL_RE.search(msg)))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9], dtype=np.float32)


def rule_features_batch(msgs) -> np.ndarray:
    """Vectorized wrapper: apply rule_features to a list/Series of messages.

    Parameters
    ----------
    msgs : list or pd.Series
        Raw SMS texts.

    Returns
    -------
    np.ndarray
        Shape (n_samples, 9).
    """
    return np.vstack([rule_features(m) for m in msgs])


# ---------------------------------------------------------------------------
# Branch 2: Heuristic weighted score
# ---------------------------------------------------------------------------

def heuristic_score(rule_vec: np.ndarray) -> float:
    """Compute the Jayaprakash et al. (2024) weighted heuristic score.

    Score = Σ wᵢ · fᵢ(xᵢ)

    A higher score indicates a higher probability of smishing.

    Parameters
    ----------
    rule_vec : np.ndarray
        Shape (9,) binary rule feature vector from :func:`rule_features`.

    Returns
    -------
    float
        Scalar score in [0, 1] (since all weights sum to 1 and features are binary).
    """
    return float(np.dot(WEIGHTS, rule_vec))


def heuristic_score_batch(rule_matrix: np.ndarray) -> np.ndarray:
    """Vectorized wrapper over :func:`heuristic_score`.

    Parameters
    ----------
    rule_matrix : np.ndarray
        Shape (n_samples, 9).

    Returns
    -------
    np.ndarray
        Shape (n_samples, 1) — column vector for hstack compatibility.
    """
    scores = rule_matrix @ WEIGHTS
    return scores.reshape(-1, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Branch 3: TF-IDF vectorizer factory
# ---------------------------------------------------------------------------

def build_tfidf() -> TfidfVectorizer:
    """Create the TF-IDF vectorizer with project-standard hyperparameters.

    Settings follow Seo et al. (2024): unigram + bigram, sublinear TF,
    vocabulary pruned to top-5000 features.

    Returns
    -------
    sklearn.feature_extraction.text.TfidfVectorizer
        Unfitted vectorizer instance.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=5000,
        sublinear_tf=True,
    )


# ---------------------------------------------------------------------------
# Feature composition
# ---------------------------------------------------------------------------

def compose_features(
    raw_msgs,
    clean_msgs,
    tfidf: TfidfVectorizer,
) -> sp.csr_matrix:
    """Horizontally stack TF-IDF, rule flags, and heuristic score.

    Column layout: [tfidf (5000) | rule_flags (9) | heuristic_score (1)]

    Parameters
    ----------
    raw_msgs : list or pd.Series
        Original unprocessed SMS texts (used for rule/score branches).
    clean_msgs : list or pd.Series
        Cleaned/lemmatized texts (used for TF-IDF branch).
    tfidf : TfidfVectorizer
        A *fitted* TfidfVectorizer instance.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape (n_samples, n_tfidf + 10).
    """
    tfidf_mat  = tfidf.transform(clean_msgs)                  # sparse (n, 5000)
    rule_mat   = rule_features_batch(raw_msgs)                 # dense  (n, 9)
    score_mat  = heuristic_score_batch(rule_mat)               # dense  (n, 1)

    return sp.hstack(
        [tfidf_mat, sp.csr_matrix(rule_mat), sp.csr_matrix(score_mat)],
        format="csr",
    )


# ---------------------------------------------------------------------------
# Rule-based heuristic classifier (Jain & Gupta 2018 baseline)
# ---------------------------------------------------------------------------

class RuleBasedClassifier(BaseEstimator, ClassifierMixin):
    """Pure rule-based smishing detector — no learned parameters from text.

    Mirrors the Jain & Gupta (2018) approach: classify a message as smishing
    if its weighted heuristic score (the last column of the composed feature
    matrix) meets or exceeds a fixed threshold. The threshold is the only
    parameter — and it is **not** tuned from data. This keeps the baseline
    purely heuristic (no learning).

    This makes it a fair "old-school" baseline: deterministic, fully
    interpretable, and representative of pre-ML detection systems.

    Parameters
    ----------
    threshold : float
        Decision boundary on the heuristic score in [0, 1]. Default is 0.25.

    Paper echo: Jain & Gupta (2018) Section 3 — IF-THEN rule evaluation.
    """

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def fit(self, X, y):
        """No-op fit to satisfy scikit-learn's estimator API.

        Parameters
        ----------
        X : sparse matrix, shape (n_samples, n_features)
            Full composed feature matrix; heuristic score is the last column.
        y : array-like of int
            Binary labels (1 = spam).
        """
        self.threshold_ = float(self.threshold)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        """Classify messages using the heuristic score threshold.

        Parameters
        ----------
        X : sparse matrix, shape (n_samples, n_features)
            Full composed feature matrix; only the last column (score) is used.

        Returns
        -------
        np.ndarray of int
            Predicted labels: 1 = smishing, 0 = ham.
        """
        check_is_fitted(self, "threshold_")
        scores = np.asarray(X[:, -1].toarray()).ravel()
        return (scores >= self.threshold_).astype(int)
