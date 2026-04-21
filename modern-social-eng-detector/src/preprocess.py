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
    """Load the UCI SMS Spam Collection into a DataFrame.

    If the file at *path* (default: ``data/SMSSpamCollection``) does not exist,
    it is downloaded and extracted automatically from the UCI ML Repository.

    Returns
    -------
    pd.DataFrame
        Columns: ``label`` (str, 'ham'/'spam'), ``text`` (str), ``y`` (int, 1=spam).

    Paper echo: dataset used across all three reference papers.
    """
    path = Path(path) if path else _DEFAULT_PATH

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[preprocess] Downloading SMS Spam Collection → {path}")
        resp = requests.get(_DATA_URL, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # The zip contains 'SMSSpamCollection' (no extension) among others
            name = next(n for n in zf.namelist() if "SMSSpamCollection" in n and not n.endswith("/"))
            with zf.open(name) as src, open(path, "wb") as dst:
                dst.write(src.read())
        print(f"[preprocess] Saved {path.stat().st_size // 1024} KB")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="latin-1",
    )
    df["y"] = (df["label"] == "spam").astype(int)
    return df


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
