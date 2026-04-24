# Smishing Detection: Complete Technical Reference

**Project:** Smish-Smash — hybrid smishing detector tracing three paper lineages  
**Dataset:** UCI SMS Spam Collection (5,572 messages, 13.4% spam)  
**Papers implemented:** Jain & Gupta (2018) · Jayaprakash et al. (2024) · Seo et al. (2024)

---

## 1. Paper-to-Implementation Audit

### 1.1 Jain & Gupta (2018) — Rule-Based Framework

**Paper proposes:** 9 IF-THEN binary rules extracted from SMS content, evaluated against Decision Tree (C4.5), RIPPER, and PRISM classifiers. SMS dataset: 5,169 messages (362 smishing filtered from 5,574 spam, 4,807 ham). Best result: RIPPER achieves 92.82% TPR, 99.01% TNR.

**Implementation audit per rule:**

| Rule | Paper Specification | Implementation | Status |
|---|---|---|---|
| R1 | URL present in message | `URL_RE.search(msg)` — matches http(s)://, www., bare domains | ✓ Correct |
| R2 | Math symbols: `+, -, <, >, /` | `_MATH_RE = re.compile(r"[%^*/=]")` — missing `+`, `-`, `<`, `>`; added `%`, `^`, `=` | ⚠ Deviated |
| R3 | Currency signs `$`, `£` | `MONEY_RE` — adds `€ ¥ ₹` and ISO codes (USD, GBP, EUR, INR, JPY) | ✓ Extended |
| R4 | Mobile number present | `PHONE_RE` — international and 10+ digit formats | ✓ Correct |
| R5 | Suspicious keywords: free, accident, awards, dating, won, service, lottery, mins, visit, delivery, cash, claim, Prize | `SUSPICIOUS` set — 37 keywords including all paper keywords + OTP, KYC, blocked, activate, etc. | ✓ Extended |
| R6 | Message length > 150 chars | `float(len(msg) > 150)` | ✓ Exact |
| R7 | Self-answering SMS (subscribe/unsubscribe) | `_REPLY_RE = re.compile(r"\b(?:reply\|text\s\|call\s)")` — captures call-to-action pattern | ✓ Semantically equivalent |
| R8 | Visual morphemes (numerals and signs) | `_LEET_RE = re.compile(r"[0-9@#$!]{2,}")` — requires 2+ consecutive chars (stricter than paper) | ⚠ Stricter |
| R9 | Email address present | `_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")` | ✓ Correct |

**What is missing from Jain & Gupta (2018):**
- Decision Tree (C4.5), RIPPER, and PRISM classifiers — paper's actual classifiers; replaced here with threshold on heuristic score
- Lingo language normalization — paper preprocesses SMS abbreviations (u→you, wanna→want to) using the Lingo framework; implementation uses NLTK lemmatization instead
- Histogram-based threshold derivation — paper plots rule count distributions (Figure 3) to empirically select the smishing threshold; implementation uses fixed threshold 0.25
- Explicit smishing/spam distinction — paper filters 362 true smishing out of 5,574 spam messages; implementation treats all 747 spam messages as smishing proxies
- Zero-hour attack evaluation — paper claims effectiveness on zero-day attacks; this project does not evaluate this property

### 1.2 Jayaprakash et al. (2024) — Heuristic ML for Phishing Detection

**Paper proposes:** Multi-modality heuristic detection for URL, email, and website phishing. Core contribution is a weighted linear scoring function of domain-specific features for each modality:

- URL: `Phishing_Score = w1·f1(Length) + w2·f2(HTTPS) + w3·f3(DomainAge) + … + wn·fn(other)`  
- Email: `S = W_header·F_header + W_keyword·F_keyword + W_link·F_link + W_attachment·F_attachment + W_behavior·F_behavior + W_language·F_language`  
- Website: `P = W_url·F_url + W_https·F_https + W_domain·F_domain + … + W_brand·F_brand`

Best accuracies: URL 97.2%, Email 97.4%, Website 98.1% (vs. Random Forest, SVM, Naïve Bayes).

**Implementation audit:**

| Paper Element | Implementation | Status |
|---|---|---|
| Weighted linear score formula (Eq. 1) | `heuristic_score = np.dot(WEIGHTS, rule_vec)` — `WEIGHTS = [0.18, 0.05, 0.15, 0.10, 0.20, 0.07, 0.08, 0.07, 0.10]` | ✓ Formula applied |
| Feature weights | Paper does not publish specific values for SMS; weights are independently assigned to 9 Jain & Gupta rules | ⚠ No paper source for values |
| URL syntactic features (56 features: length, @symbol, HTTPS, IP, subdomain depth) | Not implemented | ✗ Missing |
| URL external service features (Alexa rank, WHOIS domain age) | Not implemented | ✗ Missing |
| Email SPF verification (Algorithm 1) | Not implemented | ✗ Missing |
| Email weighted score (Algorithm 2) | Not implemented (different modality) | ✗ Out of scope |
| Website weighted score (Algorithm 3) | Not implemented (different modality) | ✗ Out of scope |
| Classifier comparison (RF, SVM, NB) | ✓ Implemented identically | ✓ Correct |
| Evaluation metrics (Accuracy, Precision, Recall, F1) | ✓ All four reported | ✓ Correct |
| Multi-dataset evaluation (5 datasets per modality) | Single dataset (UCI SMS) used | ⚠ Reduced |
| ANOVA significance testing | Not implemented | ✗ Missing |

**Scope mismatch note:** Jayaprakash et al. (2024) targets URL/email/website phishing. This project applies only the scoring formula concept to SMS, mapping the 9 SMS-domain Jain & Gupta rules to the weighted-feature-vector paradigm. The feature weights are not from the paper.

### 1.3 What Is Implemented Beyond Any Paper (Extensions)

1. **Word-level TF-IDF** (5,000 features, unigram+bigram, sublinear TF, min_df=2, max_df=0.95) — not from any cited paper; standard NLP baseline
2. **Character-level TF-IDF** (3,000 features, char_wb analyzer, 3–5 grams) — not from any cited paper; captures obfuscated tokens and leet-speak fragments
3. **Hybrid feature stacking** — [word_tfidf | char_tfidf | rule_flags(9) | heuristic_score(1)] horizontally concatenated into a single sparse matrix
4. **Placeholder substitution** in preprocessing — URLs/phones/money replaced with `<url>/<phone>/<money>` tokens rather than deleted, so both the TF-IDF branch and rule branch can fire independently
5. **Class imbalance handling** — `class_weight="balanced"` on all sklearn classifiers to compensate for 87%/13% ham/spam skew
6. **Stratified 80/20 split** — ensures spam proportion is preserved in both train and test splits
7. **Span-level explanation layer** (`explain.py`) — character-span highlighting with type taxonomy (url, short_link, phone, money, urgency, action, keyword, leet, email), priority-based overlap deduplication, and human-readable reason synthesis
8. **Short-link domain detection** — 23 known URL shortener domains (bit.ly, tinyurl.com, t.co, etc.) flagged as a sub-type of the URL highlight
9. **Urgency/action word sub-lexicons** — SUSPICIOUS set from Jain & Gupta is further split into URGENCY_WORDS and ACTION_WORDS for frontend color-coding granularity
10. **Flask REST API** — `/predict` endpoint returns `{prediction, is_smishing, confidence}` with joblib-persisted model artefacts
11. **Supplementary dataset** — optional augmentation from `Deysi/spam-detection-dataset` via HuggingFace `datasets`

---

## 2. Problem Formulation

**Task:** Binary SMS classification.  
- Class 0 (ham): legitimate message  
- Class 1 (spam/smishing): malicious message attempting credential theft or financial fraud via SMS  

**Definition (Jain & Gupta 2018):** Smishing is a subset of SMS spam where the intent is specifically to steal personal credentials (bank details, OTP, contact information) through malicious links, fake prize announcements, or urgent account alerts — distinguished from benign promotional spam.

**Formal decision function:**

```
ŷ = argmax_{c ∈ {0,1}} P(Y=c | φ(x))
```

where `φ(x)` is the hybrid feature map (Section 4) and the decision boundary is either learned (ML classifiers) or fixed (rule-based threshold).

---

## 3. Dataset

**Source:** UCI SMS Spam Collection (Almeida, Hidalgo & Yamakami, 2011)  
**Size:** 5,572 messages — 4,825 ham (86.6%), 747 spam (13.4%)  
**Language:** English  
**Format:** Tab-separated, columns `[label ∈ {ham, spam}, text]`  
**Split:** 80% train (4,457 messages), 20% test (1,115 messages), stratified by label, `random_state=42`

**Class imbalance ratio:** 4825:747 ≈ 6.5:1 (ham:spam)  
This imbalance is addressed via `class_weight="balanced"` in all sklearn classifiers, which internally re-weights samples as:

```
w_c = n_samples / (n_classes × n_samples_in_class_c)
w_spam = 5572 / (2 × 747) ≈ 3.73
w_ham  = 5572 / (2 × 4825) ≈ 0.58
```

---

## 4. Text Preprocessing Pipeline

Defined in `src/preprocess.py`. Applied to the **cleaned text branch only** — the rule branch always receives raw (uncleaned) text.

### 4.1 Compiled Regex Patterns

**URL_RE:**
```
(?:https?://|www\.)\S+               — explicit scheme or www prefix
| \b[a-z0-9-]{2,}\.[a-z]{2,6}(?:/\S*)?  — bare domain (e.g. amaz0n.tk/claim)
```
Flags: `re.IGNORECASE`

**PHONE_RE:**
```
(?:\+?\d[\d\s\-().]{7,}\d)    — international/formatted numbers
| \b\d{10,}\b                 — standalone 10+ digit runs
```

**MONEY_RE:**
```
[£$€¥₹]\s*\d[\d,.]*           — symbol followed by amount
| \b\d[\d,.]*\s*[£$€¥₹]       — amount followed by symbol
| \b(?:USD|GBP|EUR|INR|JPY|CAD|AUD)\s*\d[\d,.]*  — ISO currency code + amount
```

### 4.2 clean_text() Pipeline

```
raw_msg
  │
  ▼
1. Lowercase
  │
  ▼
2. URL_RE.sub(" <url> ")      ← placeholder, not deletion
   PHONE_RE.sub(" <phone> ")
   MONEY_RE.sub(" <money> ")
  │
  ▼
3. str.maketrans("","",punctuation)  ← strip all punctuation
  │
  ▼
4. NLTK word_tokenize()
  │
  ▼
5. Remove NLTK English stopwords (179 tokens)
  │
  ▼
6. WordNetLemmatizer.lemmatize()     ← inflection reduction
  │
  ▼
" ".join(tokens)
```

**Rationale for placeholder substitution vs. deletion:** The TF-IDF vocabulary learns `<url>`, `<phone>`, `<money>` as highly discriminative features (they appear almost exclusively in spam). Deletion would destroy this signal. Meanwhile, the rule branch fires on the raw text patterns independently — so both branches detect the signal, avoiding loss.

---

## 5. Feature Engineering

Defined in `src/features.py`. All three branches are horizontally stacked into a single sparse CSR matrix:

```
X = [word_tfidf (5000) | char_tfidf (≤3000) | rule_flags (9) | heuristic_score (1)]
Total: up to 8010 dimensions
```

Training shape (5010 dims without char_tfidf in notebook version): `(4457, 5010)`

### 5.1 Branch A — TF-IDF (Seo et al. 2024 influence)

**Vectorizer parameters:**
```python
TfidfVectorizer(
    ngram_range=(1, 2),    # unigrams and bigrams
    min_df=2,              # term must appear in ≥2 documents
    max_df=0.95,           # term must not appear in >95% of documents
    max_features=5000,     # vocabulary pruned to top 5000 by TF-IDF score
    sublinear_tf=True,     # replace TF with 1 + log(TF) to compress heavy hitters
)
```

**TF-IDF weight computation per term t in document d:**

```
TF(t, d) = 1 + log(count(t, d))   [sublinear]
IDF(t)   = log((1 + N) / (1 + df(t))) + 1   [sklearn smoothed]
TFIDF(t, d) = TF(t, d) × IDF(t)
```

Each document vector is L2-normalized post-computation.

**Critical:** TF-IDF is fit **only on the training split**, then `.transform()` applied to test — preventing vocabulary leakage.

### 5.2 Branch B — Character-level TF-IDF (Extension)

```python
TfidfVectorizer(
    analyzer="char_wb",    # n-grams within word boundaries only
    ngram_range=(3, 5),    # 3-, 4-, 5-character n-grams
    min_df=2,
    max_df=0.95,
    max_features=3000,
    sublinear_tf=True,
    lowercase=True,
)
```

Applied to **raw** (uncleaned) text. Captures:
- Obfuscated tokens: `amaz0n`, `fr33`, `l0ttery`
- Short-link fragments: `.tk/`, `.ly/ab`
- Morphological sub-word patterns independent of word boundaries

### 5.3 Branch C — Rule Feature Vector (Jain & Gupta 2018)

`rule_features(msg)` → `np.ndarray, shape (9,), dtype float32`

Each element is a binary indicator (0.0 or 1.0):

```
f1 = bool(URL_RE.search(msg))               # URL present
f2 = bool(_MATH_RE.search(msg))             # Math/operator symbols [%^*/=]
f3 = bool(MONEY_RE.search(msg))             # Currency sign
f4 = bool(PHONE_RE.search(msg))             # Phone number
f5 = bool(set(words) & SUSPICIOUS)          # Suspicious keyword match
f6 = bool(len(msg) > 150)                   # Message length threshold
f7 = bool(_REPLY_RE.search(msg))            # Call-to-action: reply/text/call
f8 = bool(_LEET_RE.search(msg))             # Visual morphemes (2+ leet chars)
f9 = bool(_EMAIL_RE.search(msg))            # Embedded email address
```

**SUSPICIOUS keyword set (37 tokens):**
`win, winner, prize, free, claim, urgent, verify, bank, account, otp, suspended, click, offer, cash, loan, gift, congrats, selected, limited, expire, password, confirm, update, bonus, reward, lucky, credit, debit, alert, important, immediately, kyc, blocked, reactivate, transaction, approve, activate`

### 5.4 Branch D — Weighted Heuristic Score (Jayaprakash et al. 2024)

```
Score = W · f = Σᵢ wᵢ · fᵢ
```

where `W = [0.18, 0.05, 0.15, 0.10, 0.20, 0.07, 0.08, 0.07, 0.10]` and `f` is the rule feature vector.

Weight assignment rationale (not from paper; assigned by signal strength):

| Feature | Weight | Rationale |
|---|---|---|
| f1 (URL) | 0.18 | Strong smishing indicator — phishing requires a link |
| f2 (Math) | 0.05 | Weak signal — common in legitimate messages too |
| f3 (Currency) | 0.15 | Strong — prize/payment lures always use currency symbols |
| f4 (Phone) | 0.10 | Moderate — attackers embed callback numbers |
| f5 (Keyword) | 0.20 | Strongest — vocabulary gap between ham and spam is maximal |
| f6 (Length) | 0.07 | Moderate — smishing tends to be verbose |
| f7 (CTA) | 0.08 | Moderate — call-to-action is characteristic |
| f8 (Leet) | 0.07 | Moderate — obfuscation signal |
| f9 (Email) | 0.10 | Moderate — credential collection via email redirect |

Score range: [0, 1] since all weights sum to 1.0 and features are binary.

### 5.5 Feature Composition

```python
def compose_features(raw_msgs, clean_msgs, tfidf, char_tfidf=None):
    word_mat  = tfidf.transform(clean_msgs)          # sparse, (n, 5000)
    rule_mat  = rule_features_batch(raw_msgs)        # dense,  (n, 9)
    score_mat = heuristic_score_batch(rule_mat)      # dense,  (n, 1)
    
    blocks = [word_mat]
    if char_tfidf is not None:
        char_mat = char_tfidf.transform(raw_msgs)    # sparse, (n, ≤3000)
        blocks.append(char_mat)
    blocks.append(sp.csr_matrix(rule_mat))
    blocks.append(sp.csr_matrix(score_mat))
    
    return sp.hstack(blocks, format="csr")
```

The last column of the composed matrix is always the heuristic score — `RuleBasedClassifier` depends on this positional convention.

---

## 6. Classification

### 6.1 Rule-Based Classifier (Jain & Gupta 2018 era baseline)

```python
class RuleBasedClassifier:
    threshold: float = 0.25   # fixed, not learned

    def predict(self, X):
        scores = np.asarray(X[:, -1].toarray()).ravel()  # last column = heuristic score
        return (scores >= self.threshold).astype(int)
```

**Characteristics:**
- Zero learned parameters from text
- Threshold 0.25 means: any message scoring ≥0.25 on the weighted rule sum is classified smishing
- A message satisfying f5 (keyword, w=0.20) + f1 (URL, w=0.18) alone scores 0.38 → classified smishing
- Purely deterministic; identical prediction for identical input regardless of training data
- Intentionally untuned to represent 2018-era detection capability

**Equivalent threshold interpretation:** score ≥ 0.25 is triggered by any two rules of moderate weight, or any single high-weight rule (f5=0.20 alone does not reach 0.25; needs one more signal).

### 6.2 Naïve Bayes — MultinomialNB

Generative model assuming conditional independence of features given class label:

```
P(Y=c | x) ∝ P(Y=c) · ∏ᵢ P(xᵢ | Y=c)
```

For TF-IDF features, MultinomialNB is not technically correct (TF-IDF values are real-valued and can be negative with sublinear_tf if rare terms are absent), but in practice it remains effective. Applied to the full feature matrix including rule flags.

**Prior:** `P(spam) = 747/5572 ≈ 0.134`  
**No class reweighting** (class_weight not applicable to MultinomialNB; dataset imbalance partially handled by the prior).

### 6.3 Logistic Regression

Linear discriminant trained with L2 regularization (default `C=1.0`):

```
ŷ = σ(wᵀφ(x) + b)   where σ is the sigmoid function
```

Loss: Cross-entropy with L2 penalty `λ||w||²`

`class_weight="balanced"` reweights the per-sample loss:

```
loss(i) = w_{y_i} · CE(ŷᵢ, yᵢ)
```

`max_iter=1000` ensures convergence on the 5010-dimensional sparse feature space.

### 6.4 Random Forest — RandomForestClassifier

Ensemble of 300 decision trees (`n_estimators=300`). Each tree is fit on a bootstrap sample of the training data (bagging), and each node split considers a random subset of √(5010) ≈ 71 features.

Prediction: majority vote across 300 trees.

`class_weight="balanced"` adjusts sample weights at each bootstrap draw.

**Characteristics:** High precision (0.9776) due to voting — false positives require majority agreement. Lower recall (0.8792) than linear models because sparse high-dimensional features with many zero entries are harder for tree-based splitting.

### 6.5 Linear SVM — LinearSVC

Maximizes the margin between the two class hyperplanes in the high-dimensional TF-IDF feature space:

```
minimize  (1/2)||w||² + C · Σᵢ ξᵢ
subject to  yᵢ(wᵀφ(xᵢ) + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0
```

Default `C=1.0`. Uses the dual coordinate descent solver (liblinear).

`class_weight="balanced"` scales the penalty `C` per class:  
`C_spam = C × w_spam ≈ 3.73`, `C_ham = C × w_ham ≈ 0.58` — the model is penalized ~6.5× more for misclassifying a spam message than a ham message.

LinearSVC achieves the highest F1 because large-margin optimization on sparse TF-IDF features is exactly the problem SVMs were designed for.

---

## 7. Evaluation

### 7.1 Metrics

All metrics computed on the test split (1,115 messages, 150 spam, 965 ham at the 13.4% ratio).

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)      [spam class; zero_division=0]
Recall    = TP / (TP + FN)      [spam class; zero_division=0]
F1        = 2 · (Precision · Recall) / (Precision + Recall)
```

**Primary metric:** F1, because the dataset is imbalanced — accuracy is misleading (a classifier predicting all-ham achieves 86.6% accuracy but 0% F1 on spam).

### 7.2 Results Table

Test set: 1,115 messages | Feature matrix: (4457, 5010) training, (1115, 5010) test

| Classifier | Accuracy | Precision | Recall | F1 | Era |
|---|---|---|---|---|---|
| Rule-Based (2018) | 0.9534 | 0.8212 | 0.8322 | 0.8267 | 2018 heuristic |
| Naïve Bayes | 0.9848 | 0.9648 | 0.9195 | 0.9416 | Classic ML |
| Logistic Regression | 0.9803 | 0.9320 | 0.9195 | 0.9257 | Classic ML |
| Random Forest | 0.9812 | 0.9776 | 0.8792 | 0.9258 | Ensemble |
| **Linear SVM** | **0.9848** | **0.9583** | **0.9262** | **0.9420** | **Best** |

### 7.3 Result Analysis

**Rule-Based vs. ML gap:**
- F1 gap: 0.8267 → 0.9416–0.9420 (+0.115 absolute)
- The rule-based system misses smishing messages that contain none of the hard-coded linguistic markers — any creative phrasing without URLs, currency, or keywords passes through undetected
- Rule-based has surprisingly high accuracy (95.3%) because most ham messages satisfy zero rules — the TNR is high; the gap is entirely in TPR (recall)

**Precision vs. Recall trade-off:**
- Random Forest: highest precision (0.9776) but lowest recall (0.8792) — conservative, rarely triggers on ham
- Naïve Bayes and LinearSVC: balanced precision/recall trade-off — preferred in deployment where missing a smishing message (FN) is more costly than a false alarm (FP)

**Feature importance intuition:**
- TF-IDF contributes most of the discriminative power — spam messages have a distinct vocabulary (`win`, `prize`, `free`, `claim`, `http://`, `<url>`)
- Rule flags add 9 dimensions of interpretable signal; heuristic score provides a single summarizer for the rule branch
- Character TF-IDF (in the `src/` module) captures obfuscation that word-level tokenization misses

### 7.4 Comparison to Paper Benchmarks

Jayaprakash et al. (2024) reports on email (97.4% accuracy on UCI ML dataset) and website (98.1% on Mendeley) tasks — not SMS. The implementation achieves 98.48% accuracy for LinearSVM on SMS, which is broadly comparable in magnitude but not directly comparable due to different tasks and datasets.

Jain & Gupta (2018) reports RIPPER: 92.82% TPR, 99.01% TNR on their filtered 362-smishing dataset. The Rule-Based baseline here achieves 83.22% recall / 97.9% specificity (estimated from precision/accuracy) on the full 747-spam corpus — a performance difference attributable to: (1) different dataset composition (general spam vs. filtered smishing), (2) using threshold-on-score vs. RIPPER's learned rule induction, and (3) no Lingo normalization of SMS abbreviations.

---

## 8. Full Pipeline Architecture

```
Raw SMS message
      │
      ├──────────────────────────────────────── Branch C/D (rules)
      │   rule_features(raw_msg)
      │   → f = [f1, f2, ..., f9]  ∈ {0,1}⁹
      │   → score = W·f  ∈ [0,1]
      │
      ├──────── clean_text(raw_msg)
      │             │
      │             ├──────── Branch A (word TF-IDF)
      │             │   tfidf.transform([cleaned])
      │             │   → sparse vector ∈ ℝ⁵⁰⁰⁰
      │             │
      │             └──────── Branch B (char TF-IDF, optional)
      │                 char_tfidf.transform([raw])
      │                 → sparse vector ∈ ℝ≤³⁰⁰⁰
      │
      └──── compose_features()
                │
                ▼
           X = [word_tfidf | char_tfidf | rule_flags | score]
                        ↓
                  clf.predict(X)
                        ↓
                  ŷ ∈ {0, 1}
                  "HAM" | "SMISHING"
```

**Training-time invariants:**
1. TF-IDF vectorizers fit on training split only
2. Rule features require no fitting — purely deterministic
3. Heuristic score weights are constants — no learning
4. `RuleBasedClassifier.fit()` is a no-op (records `threshold_` only)
5. All sklearn classifiers use `random_state=42` for reproducibility

**Inference-time (Flask API):**
```
POST /predict {"text": "..."}
      │
      ├── clean_text(msg)
      ├── compose_features([msg], [clean_msg], tfidf)   ← loaded at startup
      ├── clf.predict(X)                                ← loaded at startup
      └── jsonify({prediction, is_smishing, confidence})
```

---

## 9. Explanation Layer (explain.py)

A deterministic, model-agnostic explanation layer over the rule branch, designed for frontend highlighting. Not present in any of the three papers — fully custom extension.

### 9.1 Highlight Type Taxonomy

| Type | Detection Method | Visual Purpose |
|---|---|---|
| `url` | URL_RE match on raw msg | Flag phishing links |
| `short_link` | URL_RE match + domain in SHORT_LINK_DOMAINS (23 entries) | Distinguish obfuscated links |
| `phone` | PHONE_RE match | Callback number detection |
| `money` | MONEY_RE match | Prize/payment lure highlight |
| `email` | _EMAIL_RE match | Embedded address for credential harvest |
| `urgency` | Word boundary match against URGENCY_WORDS set | Pressure tactics |
| `action` | Word boundary match against ACTION_WORDS set | Call-to-action phrasing |
| `keyword` | Match against SUSPICIOUS - URGENCY_WORDS - ACTION_WORDS | General spam vocabulary |
| `leet` | _LEET_RE match (2+ consecutive [0-9@#$!]) | Obfuscation detection |

### 9.2 Overlap Resolution (Priority Deduplication)

Priority order (descending): `short_link(9) > url(8) > money(7) = phone(7) > email(6) > urgency(5) > action(4) > keyword(3) > leet(2)`

Algorithm:
```
for each span s (sorted by start, then by descending length):
    if s is fully contained in any kept span k
    AND priority(s) <= priority(k):
        drop s
    else:
        keep s
```

This ensures a URL like `http://bit.ly/claim` is tagged `short_link` (priority 9), not also `url` (8) or `keyword` ("claim", priority 3).

### 9.3 explain() Return Schema

```python
{
    "highlights": [
        {"type": "url", "start": 42, "end": 71, "text": "http://amaz0n-reward.tk/..."},
        ...
    ],
    "reasons": [
        "suspicious URL detected",
        "urgency keyword detected",
        "currency symbol detected"
    ],
    "rule_flags": {
        "url_present": True,
        "math_operators": False,
        "currency_symbol": True,
        ...
    },
    "heuristic_score": 0.48
}
```

---

## 10. Key Deviations and Design Decisions

### 10.1 Rule 2 (Math Symbols) Deviation

**Paper:** `+, -, <, >, /`  
**Implementation:** `[%^*/=]`

The paper's intent was to detect mathematical operators as an indicator of encoded content or equations in smishing messages. However, `-` and `+` are extremely common in legitimate messages (phone numbers: `+44-1234-567890`, date ranges: `3-4pm`), making them poor discriminators. The implementation replaces them with `%^*=` — operators that are rare in legitimate SMS but present in some encoded/formatted smishing content. The `/` in the paper is subsumed by URL detection (f1). This deviation likely improves precision without reducing recall significantly.

### 10.2 No Lingo Normalization

Jain & Gupta (2018) uses Lingo — an SMS-specific normalization dictionary (u→you, 2mrw→tomorrow, lol→laugh out loud). The implementation uses NLTK WordNetLemmatizer, which handles morphological inflections (running→run, prizes→prize) but not SMS abbreviations.

**Impact:** Suspicious keywords expressed as abbreviations (e.g., `fr33`, `wnr`, `acc`) pass through the keyword filter. The char TF-IDF branch partially compensates by capturing sub-character patterns.

### 10.3 RuleBasedClassifier ≠ RIPPER/DT/PRISM

Jain & Gupta (2018) uses data-driven rule classifiers that **learn which combination of rules predicts smishing** from labeled training data. The `RuleBasedClassifier` here uses a **fixed linear combination (heuristic score)** with a fixed threshold — no learning occurs. This is a deliberate simplification to represent the "old-school" baseline era; a proper replication would train RIPPER on the rule feature matrix.

### 10.4 Weight Values Have No Paper Source

`WEIGHTS = [0.18, 0.05, 0.15, 0.10, 0.20, 0.07, 0.08, 0.07, 0.10]` are assigned by reasoning about relative signal strength per rule, not extracted from Jayaprakash et al. (2024). Jayaprakash's paper describes the formula structure but provides no specific weight values for SMS features — its weights are for URL syntactic features, email header analysis, and website DOM properties.

### 10.5 Spam ≠ Smishing

The UCI SMS Spam Collection labels messages as `ham` or `spam`. Not all spam is smishing (some are promotional advertisements). Jain & Gupta (2018) manually filtered 362 true smishing messages from 5,574 spam. This project uses all 747 spam as a proxy — introducing label noise: promotional spam messages may lack phishing intent but are trained as smishing.

---

## 11. Summary: What Was Implemented, What Was Missed, What Was Added

### Implemented from Jain & Gupta (2018)
- All 9 binary rules (with R2 and R8 deviation noted)
- Binary feature vector format
- Threshold-based classification concept
- SMS Spam Collection dataset

### Missed from Jain & Gupta (2018)
- DT/RIPPER/PRISM classifiers (data-driven rule induction, not threshold)
- Lingo SMS normalization
- Histogram-based threshold selection
- Smishing-vs-spam distinction in dataset filtering
- Zero-day attack evaluation

### Implemented from Jayaprakash et al. (2024)
- Weighted linear heuristic scoring formula (Score = Σ wᵢ · fᵢ)
- Classifier comparison methodology
- Accuracy/Precision/Recall/F1 evaluation framework

### Missed from Jayaprakash et al. (2024)
- URL syntactic features (56 features)
- Email SPF verification and weighted email classifier
- Website DOM-based phishing detection
- External service feature extraction (Alexa rank, WHOIS)
- ANOVA significance testing
- Multi-dataset evaluation

### Added Beyond Papers
- Word TF-IDF (5000 features, unigram+bigram, sublinear TF)
- Character TF-IDF (3000 features, char_wb, 3–5 grams)
- Hybrid feature stacking (sparse matrix concatenation)
- Placeholder substitution in preprocessing
- Class imbalance handling (class_weight="balanced")
- Stratified train/test split
- Span-level explanation with priority-based deduplication
- Short-link domain taxonomy (23 domains)
- Flask REST API with live inference
- Supplementary dataset augmentation (HuggingFace)
- Confusion matrix and grouped bar chart visualization
