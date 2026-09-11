# Evaluation Methodology

## 1. Dataset and Uber_Support Selection

- **Source**: Kaggle "Customer Support on Twitter" (twcs.csv, 2.8M+ tweets)
- **Brand**: Uber_Support — filtered from multi-brand dataset
- **Total Uber_Support rows**: ~40,000 conversation turns
- **Train/Dev/Golden split**: 35,473 training / 200 human-labelled golden set
- **Preprocessing**: Text cleaning, URL/mention removal, lowercasing

## 2. Temporal / Conversation-Aware Splitting

1. **Conversation grouping**: All turns kept together — no conversation split across train/eval.
2. **Temporal ordering**: Sorted by first-tweet timestamp. Earliest 85% = training.
3. **Golden-set pool**: Last 200 of held-out pool, reserved exclusively for human annotation.

Reproducibility: random seed 42 (see `evaluation/rebuild_golden_set.py`).

## 3. Golden Set Methodology

- **Size**: 200 examples (verified — 200/200 annotated)
- **Human annotation**: Intent label (1 of 8) + escalation binary + reason + annotator notes
- **Annotator notes**: 200/200 present
- **Escalated examples**: 51/200 (25.5%) — 100% have explicit escalation reasons
- **File**: `evaluation/human_annotation/annotation.csv`

### Auto vs Human Agreement on Golden Set

| Metric | Agreement |
|--------|-----------|
| Intent | 130/200 (65.0%) |
| Escalation | 145/200 (72.5%) |
| Intent disagreements | 70 cases |

The 35% disagreement rate confirms pseudo-labels are not reliable ground truth.

## 4. Intent Taxonomy (8 Classes)

| Intent | Description | n=200 | % |
|--------|-------------|-------|---|
| `ride_status` | Driver location, wait times, lost items | 61 | 30.5 |
| `other` | General inquiries, off-topic | 44 | 22.0 |
| `trip_fare_dispute` | Overcharge, surge, receipt mismatch | 23 | 11.5 |
| `account_access` | Login failure, verification, banned account | 20 | 10.0 |
| `complaint` | Service dissatisfaction, rude driver | 19 | 9.5 |
| `app_functionality` | App crash, GPS bug, feature broken | 11 | 5.5 |
| `refund_request` | Explicit refund / money back ask | 11 | 5.5 |
| `cancellation` | Customer-initiated cancel, cancel fees | 11 | 5.5 |

Distribution reflects real traffic — NOT artificially balanced.

## 5. Architecture

```
Customer message
      |
[Intent Classifier] -- TF-IDF + Logistic Regression, trained on 35,473 turns
      |
[Historical Retrieval] -- FAISS semantic search (all-MiniLM-L6-v2)
      |
[Resolution Extraction] -- Extract brand_response from top-k retrieved cases
      |
[Grounded Generation] -- Template + retrieved brand response prefix
      |
[Escalation Engine] -- Rule-based v2: intent + confidence + trigger keywords
      |
Final reply + escalation flag + reason
```

## 6. Baselines

| System | Intent Acc | Macro-F1 | Weighted-F1 | Escalation F1 |
|--------|-----------|----------|-------------|---------------|
| Trivial (majority class) | ~0.22 | ~0.09 | ~0.32 | 0.000 |
| TF-IDF + LogReg | 0.560 | 0.419 | 0.513 | 0.087 |
| TF-IDF + LogReg + Escalation v2 | 0.560 | 0.419 | 0.513 | 0.291 |

## 7. Classification Results (Human Golden Set — OFFICIAL)

| Metric | Score |
|--------|-------|
| Accuracy | **0.560** |
| Macro-F1 | **0.419** |
| Weighted-F1 | **0.513** |

### Per-Intent Breakdown

| Intent | Prec | Rec | F1 | Support |
|--------|------|-----|----|---------|
| account_access | 0.667 | 0.500 | 0.571 | 20 |
| app_functionality | 0.667 | 0.545 | 0.600 | 11 |
| cancellation | 0.000 | 0.000 | 0.000 | 11 |
| complaint | 0.800 | 0.211 | 0.333 | 19 |
| other | 0.607 | 0.841 | 0.705 | 44 |
| refund_request | 1.000 | 0.091 | 0.167 | 11 |
| ride_status | 0.500 | 0.770 | 0.606 | 61 |
| trip_fare_dispute | 0.467 | 0.304 | 0.368 | 23 |

### Confusion Matrix (rows=true, cols=predicted)

```
             acct  app   canc  compl other refnd ride  fare
account_      10    2     0     0     2     0     6     0
app_func       0    6     0     0     4     0     1     0
cancella       0    0     0     0     0     0    10     1
complaint      0    0     0     4     4     0    11     0
other          0    1     0     0    37     0     5     1
refund_r       0    0     0     0     1     1     4     5
ride_sta       3    0     0     0    10     0    47     1
trip_fare      2    0     0     1     3     0    10     7
```

`ride_status` absorbs: cancellation (10/11), complaint (11/19), fare_dispute (10/23).

## 8. Escalation Results

| Metric | Baseline | Improved v2 | Delta |
|--------|----------|-------------|-------|
| Accuracy | 0.555 | **0.610** | +0.055 |
| Precision | 0.161 | **0.271** | +0.110 |
| Recall | 0.176 | **0.314** | +0.138 |
| F1 | 0.168 | **0.291** | +0.123 |

Human escalation rate: 25.5% (51/200). Missed: 48. False escalations: 15.

## 9. Grounding Audit (10-Example Trace)

| Metric | Value |
|--------|-------|
| Examples traced | 10 |
| Template-based replies | 0/10 |
| Uses brand_response from evidence | 9/10 (90%) |
| Average grounding confidence | **0.612** |

Example:
- Message: "my uber didn't turn up and I got charged" -> Intent: trip_fare_dispute (correct)
- Evidence confidence: 0.448
- Reply: "Regarding your fare concern, Happy to help, Jill..."

## 10. Reply Quality (Human Scores — 30 Examples)

| Dimension | Human Mean | Range |
|-----------|-----------|-------|
| Relevance | 2.4 | 1-5 |
| Groundedness | 1.8 | 1-2 |
| Completeness | 1.6 | 1-3 |
| Correctness | 2.5 | 1-4 |
| Tone | 4.2 | 3-5 |
| Hallucination Risk | 4.3 | 3-5 |

Strengths: tone, hallucination avoidance.
Weaknesses: relevance, groundedness, completeness (generic templates, unfilled placeholders).

## 11. LLM Judge Results (WARNING: MOCK / DEVELOPMENT ONLY)

`OPENAI_API_KEY` was not available. `run_llm_judge.py --mode openai` fell back to mock mode.
Scores below are deterministic heuristics — NOT real LLM evaluation. Do NOT cite as final results.

| Dimension | Mock Mean |
|-----------|-----------|
| Relevance | 2.28 |
| Groundedness | 2.00 |
| Correctness | 4.00 |
| Completeness | 3.27 |
| Tone | 4.38 |
| Hallucination Risk | 4.00 |
| **Overall** | **3.32** |

To run real evaluation: `set OPENAI_API_KEY=<key> && python evaluation/run_llm_judge.py`

## 12. Judge-Human Agreement (Mock LLM vs Human — 30 examples)

| Metric | Value |
|--------|-------|
| Exact agreement rate | 43.3% |
| Weighted agreement (+-1 point) | **80.6%** |

| Dimension | Spearman r | Exact | Weighted |
|-----------|-----------|-------|----------|
| Relevance | 0.033 | 33.3% | 86.7% |
| Groundedness | 0.688 | 80.0% | 100.0% |
| Correctness | 0.369 | 3.3% | 66.7% |
| Completeness | -0.112 | 10.0% | 33.3% |
| Tone | 0.601 | 76.7% | 96.7% |
| Hallucination Risk | 0.661 | 56.7% | 100.0% |

## 13. What Is Misleading About the Headline Number?

> **The real headline: 56.0% accuracy / 41.9% macro-F1** — against 200 independently human-labelled examples. The earlier 90.5% / 81.1% numbers were misleading. Here is why.

| Metric | Pseudo-label eval (misleading) | Human-label eval (official) |
|--------|:-:|:-:|
| Intent Accuracy | 0.905 | **0.560** |
| Intent Macro-F1 | 0.811 | **0.419** |
| Escalation F1 | 0.000 | **0.291** (V2) |

### Why the drop happened

The model was evaluated against labels it co-created. Intent labels were generated by keyword matching (e.g., "cancel" → cancellation, "refund" → refund_request). The classifier was trained on these pseudo-labels and tested against them — measuring agreement with its own training heuristics, not real intent understanding.

Human annotation of 200 held-out examples revealed **35% disagreement** between pseudo-labels and human labels (70/200 cases). The model had learned to mimic keyword rules, not to understand customer intent.

### The key failure pattern: ride_status absorbs everything

`ride_status` acts as a black hole, absorbing messages that should be classified as `complaint`, `cancellation`, `trip_fare_dispute`, `account_access`, and `refund_request`. This is invisible in pseudo-label evaluation because the keyword heuristics don't distinguish these either.

Confusion matrix (rows=true, cols=predicted):

```
             acct  app  canc  compl  other  refnd  ride  fare
ride_status   3    0     0     0     10      0    47     1
complaint     0    0     0     4      4      0    11     0
cancellation  0    0     0     0      0      0    10     1
trip_fare     2    0     0     1      3      0    10     7
account_acc  10    2     0     0      2      0     6     0
refund_req    0    0     0     0      1      1     4     5
```

`ride_status` absorbs: **11/19 complaints**, **10/11 cancellations**, **10/23 fare disputes**, **6/20 account access issues**.

### Concrete examples

| True intent | Model predicted | Message | Why it's misleading |
|-------------|----------------|---------|-------------------|
| complaint | ride_status | "Your driver just literally pulled into the parking lot I'm waiting at, drove out, called me after he left and didn't say anything on phone! How do I file Official complaint" | Contains "ride" context but primary intent is filing a complaint about driver behavior |
| trip_fare_dispute | ride_status | "I was charged $75 for a $10 ride how do I dispute or find out why I was overcharged" | Mentions "ride" but the core issue is a fare dispute |
| cancellation | ride_status | "I just waited over 10 mins for a car that was supposed to be 3 mins away. It drove in circles. Then cancelled on me. And somehow I am charged 6 euro???? Please fix this." | Customer-initiated cancellation with fee complaint, not a ride status inquiry |
| account_access | ride_status | "My account has been disabled and I haven't even used it for the first time!!! I can not even sign in!! Please heeeeeelp" | Account access issue with zero ride context, but classifier defaults to ride_status |
| refund_request | ride_status | "You charged my credit card 3 times for 1 ride. Can I get refunded for my other 2 charges" | Explicit refund request, but "ride" in text triggers ride_status |

### Bottom line

The earlier numbers (90.5% accuracy, 0.811 macro-F1) measured how well the classifier agrees with keyword heuristics. The real numbers (56.0% accuracy, 0.419 macro-F1) measure how well it classifies actual human intent.

## 14. Top 5 Failure Modes

### Failure 1: complaint → ride_status (11 cases, 5.5%)

**Concrete example:**
> "@Uber_Support your driver just literally pulled into the parking lot I'm waiting at, drove out, called me after he left and didn't say anything on phone! How do I file Official complaint"

Human label: **complaint**. Model prediction: **ride_status**.

**Why it failed:** The message contains "driver", "waiting", "ride" vocabulary that activates ride_status features. The classifier weights ride-related terms heavily because ride_status is 49.2% of training data. The word "driver" appears frequently in both ride_status and complaint contexts.

**Root cause:** TF-IDF features cannot disambiguate when the same vocabulary ("driver", "ride", "waiting") serves different intents. The model's prior probability for ride_status (49.2%) overwhelms weaker complaint signals like "file Official complaint" or profanity indicating dissatisfaction.

**Proposed fix:** Add a dedicated complaint feature set (sentiment lexicons, "file complaint" n-grams, profanity markers) as separate TF-IDF dimensions, or switch to a model with attention. Multi-label classification could also help.

---

### Failure 2: trip_fare_dispute → ride_status (10 cases, 5.0%)

**Concrete example:**
> "@Uber_Support I was charged $75 for a $10 ride how do I dispute or find out why I was overcharged? Why is there no way to speak directly with someone very frustrating"

Human label: **trip_fare_dispute**. Model prediction: **ride_status**.

**Why it failed:** The word "ride" appears in the text, and the message discusses a trip. The classifier detects ride_status vocabulary and cannot distinguish that the primary intent is disputing a fare — not checking ride status. The $75 amount and "overcharged" signal are present but not weighted strongly enough.

**Root cause:** Fare disputes inherently reference rides, creating vocabulary overlap. The TF-IDF model has no mechanism to understand that "charged $75 for a $10 ride" is fundamentally about money, not about ride logistics.

**Proposed fix:** Add price-related features ("charged", dollar amounts, "overcharged", "dispute", "refund") as explicit indicators for trip_fare_dispute. Consider a two-stage classifier: first detect if money is involved, then classify the specific money-related intent.

---

### Failure 3: ride_status → other (10 cases, 5.0%)

**Concrete example:**
> "@Uber_Support is there an easy way to know whether a trip has GST on it or not?"

Human label: **ride_status**. Model prediction: **other**.

**Why it failed:** The message is a general inquiry about trip taxation. It contains no strong ride_status keywords ("where is my driver", "cancelled", "ETA") and no strong other keywords either. The classifier defaults to other (24% prior) when ride_status signals are weak.

**Root cause:** Some ride_status messages are general inquiries about rides (pricing, policies, tax) that don't fit the "where is my driver / ride status check" archetype the classifier learned. These edge cases fall through to other.

**Proposed fix:** Expand ride_status training examples to include general ride inquiries. Alternatively, add a "ride_inquiry" sub-intent or use conversation context — if preceding messages are about a specific ride, this is more likely ride_status.

---

### Failure 4: cancellation → ride_status (10 cases, 10/11 cancellations)

**Concrete example:**
> "@115873 I just waited over 10 mins for a car that was supposed to be 3 mins away. It drove in circles. Then cancelled on me. And somehow I am charged 6 euro???? Please fix this."

Human label: **cancellation**. Model prediction: **ride_status**.

**Why it failed:** The message describes a ride scenario (waiting, driver location, car) with "cancelled" as one of many events. The classifier treats "cancelled" as a ride_status event (driver cancelled) rather than a customer-initiated cancellation request. The "charged 6 euro" signal is drowned out by ride vocabulary.

**Root cause:** Cancellation is only 0.2% of training data (85 examples) vs. ride_status at 49.2%. The classifier has almost no cancellation examples to learn from. Cancellation messages in Uber's domain often describe ride events ("my driver cancelled"), making the boundary genuinely ambiguous.

**Proposed fix:** Oversample cancellation examples during training, or use class weights. Add cancellation-specific features ("cancel fee", "had to cancel", "charged for cancel"). Consider whether cancellation should be a sub-intent of ride_status.

---

### Failure 5: account_access → ride_status (6 cases, 3.0%)

**Concrete example:**
> "@Uber_Support My account has been disabled and I haven't even used it for the first time!!! I am suposed to have a Free Uber ride from a friend. Who do I contact?? I can not even sign in!! Please heeeeeelp"

Human label: **account_access**. Model prediction: **ride_status**.

**Why it failed:** The message mentions "Uber ride" and is addressed to @Uber_Support, triggering ride_status features. The classifier cannot distinguish that "Free Uber ride from a friend" is context for why the account issue matters — not the primary intent. "Account has been disabled" and "can not even sign in" are strong account_access signals but are outweighed by the ride_status prior.

**Root cause:** Account access messages often mention rides as context ("I can't use my account to book a ride"), and the TF-IDF model cannot parse that ride is secondary to account. Additionally, account_access is 9.8% of training data — 5x smaller than ride_status.

**Proposed fix:** Add account-specific features ("account disabled", "can't sign in", "login", "verification code", "banned") as strong negative indicators for ride_status. A rule-based pre-filter that detects account keywords before classification could catch these cases.

## 15. Tests

32/32 pass in 32.08s. Coverage: preprocessing, taxonomy, classifier, retrieval, escalation,
pipeline, data leakage, golden set validation, baselines, edge cases.

## 16. Limitations

1. Template-based generation — replies partially grounded but use resolution prefixes
2. Low escalation recall (31.4% after v2) — 48 missed escalations
3. Classifier dominated by ride_status — absorbs cancellation/complaint/fare_dispute
4. No real LLM judge — OPENAI_API_KEY unavailable
5. Small golden set (200 examples) — tail intents have <15 examples
6. No confidence calibration

## 17. Reproducibility

```bash
git clone <repo>
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python scripts/download_data.py  # requires Kaggle API key
python evaluation/eval_human_labels.py --with-eval
python evaluation/failure_analysis_human.py
python evaluation/grounding_audit.py
python evaluation/escalation_audit.py
python -m pytest tests/ -v
```

Golden set and experiment artifacts committed. Raw 500MB Kaggle dataset NOT committed.
