# Modern Social Engineering Detector — Smishing Classifier

A hybrid SMS-phishing (smishing) detector that blends rule-based features,
a weighted heuristic score, and TF-IDF vectors, evaluated across four
classical ML classifiers.  No GPU required; end-to-end runtime ≈ 45 s on
Colab/laptop CPU.

---

## Research inspiration

| Paper | One-line contribution |
|---|---|
| **Jain & Gupta (2018)** | 9 IF-THEN rules that flag URLs, currency signs, suspicious keywords, etc. |
| **Jayaprakash et al. (2024)** | Weighted heuristic score: `Score = Σ wᵢ · fᵢ(xᵢ)` aggregates rule signals |
| **Seo et al. (2024)** | Lightweight classifier comparison (NB, LR, RF, SVM) as reproducible baselines |

---

## Folder layout

```
modern-social-eng-detector/
├── requirements.txt
├── README.md
├── .gitignore
├── data/                    # created at runtime; .gitkeep committed
├── models/                  # best_classifier.pkl, tfidf.pkl, fig3_bars.png, fig4_cm.png
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # regex constants, load_sms(), clean_text()
│   ├── features.py          # rule_features(), heuristic_score(), compose_features()
│   ├── train.py             # full training pipeline, saves model artefacts
│   └── demo.py              # loads saved model and classifies 3 sample messages
└── notebook/
    └── smishing_detector.ipynb   # Colab-ready, fully self-contained
```

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

---

## How to run

### 1. Train

```bash
python -m src.train
```

Outputs:
- Console metrics table
- `models/fig3_bars.png` — grouped bar chart (4 classifiers × 4 metrics)
- `models/fig4_cm.png`   — confusion matrix of the best-F1 model
- `models/best_classifier.pkl` + `models/tfidf.pkl`

### 2. Demo

```bash
python -m src.demo
```

Classifies three hard-coded adversarial messages and prints `[SMISHING]` / `[HAM]`.

### 3. Colab notebook

Open `notebook/smishing_detector.ipynb` in Google Colab and click **Runtime → Run all**.
No local installation needed.

---

## Expected metrics table

| Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Rule-Based (2018) | ~0.95 | ~0.82 | ~0.83 | ~0.83 |
| Naïve Bayes | ~0.985 | ~0.965 | ~0.920 | ~0.942 |
| Logistic Regression | ~0.980 | ~0.932 | ~0.920 | ~0.926 |
| Random Forest | ~0.979 | ~0.977 | ~0.859 | ~0.914 |
| **Linear SVM** | **~0.985** | **~0.958** | **~0.926** | **~0.942** |

> **Note:** If your Linear SVM F1 differs from ≈ 0.942 by more than 1 %, check that
> `random_state=42` is set everywhere and that you are using the full 5 572-row
> dataset (not a truncated download).
>
> **Rule-based baseline note:** `Rule-Based (2018)` uses only the heuristic score
> computed from the 9 Jain & Gupta-style rule flags and a **fixed threshold of 0.25**
> (no training / no tuning). Your exact numbers will vary slightly with dataset version,
> but it should underperform the ML models, especially on Recall and F1.

---

## Presentation flow (6 slides)

1. **The Threat** — What is smishing? Real-world statistics and financial impact.
2. **Dataset** — UCI SMS Spam Collection: 5 574 messages, 13 % spam.
3. **Feature Engineering** — Rule flags (Jain & Gupta), heuristic score (Jayaprakash), TF-IDF (Seo).
4. **Classifier Comparison** — Show `fig3_bars.png`; discuss trade-offs.
5. **Error Analysis** — Show `fig4_cm.png`; false negatives are the costly error.
6. **Live Demo** — Run `python -m src.demo` or Cell 8 of the notebook on stage.
