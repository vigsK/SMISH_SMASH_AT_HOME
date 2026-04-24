# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**Smish-Smash** is a smishing (SMS phishing) detector that traces the evolution of detection methods across three papers: Jain & Gupta (2018) rule-based heuristics → Jayaprakash et al. (2024) weighted scoring → Seo et al. (2024) ML classifiers trained on hybrid TF-IDF + rule features. Built for a 3rd-year undergraduate Modern Social Engineering course.

**Two deliverables:**
- `notebook/smishing_detector.ipynb` — self-contained Colab-ready notebook (primary educational artifact)
- `app.py` + frontend (`index.html`, `style.css`, `script.js`) — Flask web app for live inference

## Commands

**Install:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install flask flask-cors   # not in requirements.txt yet
```

**Train all classifiers** (downloads dataset automatically, saves to `models/`):
```bash
python -m src.train
```

**Run inference demo** (requires trained models in `models/`):
```bash
python -m src.demo
```

**Start Flask web app** (requires trained models in `models/`):
```bash
python app.py   # http://localhost:5000
```

## Architecture

### Feature Engineering (3-branch pipeline)

The core is a hybrid 5010-dimensional feature vector built in `src/features.py`:

| Branch | Dims | Source | What it captures |
|---|---|---|---|
| TF-IDF unigrams+bigrams | 5000 | `clean_text()` output | Statistical bag-of-words |
| Rule flags | 9 | Raw (uncleaned) text | Binary heuristics from Jain & Gupta 2018 |
| Heuristic score | 1 | Rule flags × fixed weights | Jayaprakash 2024 weighted sum |

`compose_features()` in `features.py` stacks all three into a sparse matrix for sklearn classifiers.

**Critical:** Rule features (`rule_features()`) must receive **raw, uncleaned** text. TF-IDF must be fit on **training split only** then applied to test.

### Text Preprocessing (`src/preprocess.py`)

`clean_text()` pipeline: lowercase → substitute URLs/phones/money with `<url>/<phone>/<money>` placeholders (not delete) → strip punctuation → tokenize → remove stopwords → lemmatize.

Placeholder substitution is intentional: both the TF-IDF branch and rule branch can independently detect these patterns.

### Classifiers (`src/train.py`)

Five classifiers are trained and compared:
1. **Rule-Based** — `RuleBasedClassifier` in `features.py`; fixed threshold 0.25 on heuristic score, no learned parameters (intentionally untuned to represent 2018-era detection)
2. **Naïve Bayes** — MultinomialNB
3. **Logistic Regression** — `class_weight="balanced"`, `max_iter=1000`
4. **Random Forest** — 300 trees, `class_weight="balanced"`
5. **Linear SVM** — LinearSVC, `class_weight="balanced"`

All use `random_state=42`. All ML classifiers use `class_weight="balanced"` for the 87/13 ham/spam imbalance.

Best-F1 classifier and the fitted TF-IDF vectorizer are saved to `models/` via joblib.

### Web App (`app.py`)

Flask serves `index.html` as root and static files by path. The single `/predict` endpoint cleans text, composes features, calls the loaded classifier, and returns `{"prediction": "SMISHING"|"LEGIT", "is_smishing": bool, "confidence": float|null}`. The app exits at startup if `models/best_classifier.pkl` or `models/tfidf.pkl` are missing — training must run first.

### Notebook (`notebook/smishing_detector.ipynb`)

8 cells, top-to-bottom dependency. Cell 8 includes a live side-by-side comparison of Rule-Based vs best ML classifier on adversarial samples. Runtime ~45s on CPU. All pipeline code is duplicated inline (does not import from `src/`).

## Dataset

UCI SMS Spam Collection (5,572 messages, 13.4% spam). Auto-downloaded by `load_sms()` in `preprocess.py` to `data/sms.csv` if absent. Labels: 0 = ham, 1 = spam/smishing.

## Key Files

| File | Role |
|---|---|
| `src/preprocess.py` | `load_sms()`, `clean_text()`, regex patterns (`URL_RE`, `PHONE_RE`, `MONEY_RE`) |
| `src/features.py` | `rule_features()`, `heuristic_score_batch()`, `build_tfidf()`, `compose_features()`, `RuleBasedClassifier` |
| `src/train.py` | Full training pipeline; saves `models/best_classifier.pkl`, `models/tfidf.pkl`, `models/fig3_bars.png`, `models/fig4_cm.png` |
| `src/demo.py` | Loads saved models, predicts on 3 hardcoded adversarial samples |
| `app.py` | Flask API; loads models once at startup |
