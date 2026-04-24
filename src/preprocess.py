"""
preprocess.py — Data loading and text cleaning pipeline.

Echoes Jain & Gupta (2018): placeholder substitution ensures URL/phone/money
signals survive into the rule-based feature layer even after text normalization.
"""

import re
import string
import zipfile
import io
from pathlib import Path

import requests
import pandas as pd
import nltk

# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = [
    ("tokenizers/punkt",       "punkt"),
    ("tokenizers/punkt_tab",   "punkt_tab"),
    ("corpora/stopwords",      "stopwords"),
    ("corpora/wordnet",        "wordnet"),
    ("corpora/omw-1.4",        "omw-1.4"),
]

for _path, _pkg in _NLTK_RESOURCES:
    try:
        nltk.data.find(_path)
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Compiled regex constants (exposed for features.py)
# ---------------------------------------------------------------------------

# Matches http(s)://, www., or bare domains like example.com / bit.ly/abc
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|"          # explicit scheme or www
    r"\b[a-z0-9-]{2,}\.[a-z]{2,6}(?:/\S*)?",  # bare domain
    re.IGNORECASE,
)

# Matches international phone numbers: +1-800-..., (0xx), standalone digit runs
PHONE_RE = re.compile(
    r"(?:\+?\d[\d\s\-().]{7,}\d)|"      # international / formatted
    r"\b\d{10,}\b",                      # long digit run (10+ digits)
    re.IGNORECASE,
)

# Matches currency symbols (£ $ € ¥ ₹) or ISO codes followed by an amount,
# or standalone ISO currency codes (USD, GBP, EUR, INR, JPY)
MONEY_RE = re.compile(
    r"[£$€¥₹]\s*\d[\d,.]*|"            # symbol + number
    r"\b\d[\d,.]*\s*[£$€¥₹]|"          # number + symbol
    r"\b(?:USD|GBP|EUR|INR|JPY|CAD|AUD)\s*\d[\d,.]*",  # ISO code + number
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_DATA_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)
_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "SMSSpamCollection"


def load_sms(path: Path | str | None = None) -> pd.DataFrame:
    """Load the SMS spam dataset, combining UCI with a HuggingFace supplement.

    Primary source: UCI SMS Spam Collection (5 574 messages, auto-downloaded).
    Supplementary : ``Deysi/spam-detection-dataset`` via the ``datasets``
                    library (~5 000 additional rows).  Falls back silently if
                    the package is missing or the dataset is unavailable.

    Returns
    -------
    pd.DataFrame
        Columns: ``label`` (str, 'ham'/'spam'), ``text`` (str), ``y`` (int, 1=spam).
    """
    path = Path(path) if path else _DEFAULT_PATH

    # --- Primary: UCI SMS Spam Collection ---
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[preprocess] Downloading SMS Spam Collection -> {path}")
        resp = requests.get(_DATA_URL, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = next(n for n in zf.namelist() if "SMSSpamCollection" in n and not n.endswith("/"))
            with zf.open(name) as src, open(path, "wb") as dst:
                dst.write(src.read())
        print(f"[preprocess] Saved {path.stat().st_size // 1024} KB")

    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="latin-1")
    df["y"] = (df["label"] == "spam").astype(int)
    print(f"[preprocess] UCI: {len(df)} messages (spam: {df['y'].sum()}, ham: {(df['y']==0).sum()})")

    # --- Supplementary: HuggingFace SMS spam dataset ---
    df2 = _load_supplementary()
    if df2 is not None:
        df = pd.concat([df, df2], ignore_index=True).drop_duplicates(subset="text").reset_index(drop=True)
        print(f"[preprocess] Combined: {len(df)} messages (spam: {df['y'].sum()}, ham: {(df['y']==0).sum()})")

    return df


def _load_supplementary() -> "pd.DataFrame | None":
    """Try to load ``Deysi/spam-detection-dataset`` from HuggingFace.

    Returns a DataFrame with columns [label, text, y] on success, else None.
    Column names are detected dynamically so minor schema changes are tolerated.
    """
    try:
        from datasets import load_dataset  # optional dependency
    except ImportError:
        print("[preprocess] 'datasets' package not installed — using UCI only (pip install datasets)")
        return None

    try:
        ds = load_dataset("Deysi/spam-detection-dataset", split="train", trust_remote_code=True)
        cols = ds.column_names

        # Detect text column (prefer 'sms', fall back to 'text' or 'message')
        text_col = next((c for c in ("sms", "text", "message") if c in cols), None)
        # Detect label column (prefer 'label', fall back to 'spam')
        label_col = next((c for c in ("label", "spam", "y") if c in cols), None)

        if text_col is None or label_col is None:
            print(f"[preprocess] Unexpected schema in supplementary dataset: {cols} — skipping")
            return None

        raw_labels = ds[label_col]
        # Normalise to int: accept 0/1 ints or 'ham'/'spam' strings
        if isinstance(raw_labels[0], str):
            y_vals = [1 if v.strip().lower() == "spam" else 0 for v in raw_labels]
        else:
            y_vals = [int(v) for v in raw_labels]

        df2 = pd.DataFrame({"text": ds[text_col], "y": y_vals})
        df2["label"] = df2["y"].map({1: "spam", 0: "ham"})
        df2 = df2[["label", "text", "y"]].dropna(subset=["text"])
        print(f"[preprocess] Supplementary: {len(df2)} messages loaded")
        return df2

    except Exception as exc:
        print(f"[preprocess] Supplementary dataset unavailable ({exc}) — using UCI only")
        return None


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_STOP = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(msg: str) -> str:
    """Normalize a raw SMS message for TF-IDF consumption.

    Pipeline (Jain & Gupta §3.1 inspired):
      1. Lowercase
      2. Replace URLs → ``<URL>``, phones → ``<PHONE>``, money → ``<MONEY>``
         (placeholders, not deletion — rule layer detects the originals separately)
      3. Strip punctuation
      4. Word-tokenize (NLTK punkt)
      5. Remove English stopwords
      6. Lemmatize with WordNetLemmatizer

    Parameters
    ----------
    msg : str
        Raw SMS text.

    Returns
    -------
    str
        Space-joined cleaned token string.
    """
    text = msg.lower()
    text = URL_RE.sub(" <url> ", text)
    text = PHONE_RE.sub(" <phone> ", text)
    text = MONEY_RE.sub(" <money> ", text)
    text = text.translate(_PUNCT_TABLE)
    tokens = word_tokenize(text)
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens if t not in _STOP and t.strip()]
    return " ".join(tokens)
