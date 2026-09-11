# Uber Customer Support Agent

## Live Demo

> **API is live on Render (free tier):** [https://hiver-support-agent.onrender.com](https://hiver-support-agent.onrender.com)
>
> - Interactive docs: [https://hiver-support-agent.onrender.com/docs](https://hiver-support-agent.onrender.com/docs)
> - Health check: [https://hiver-support-agent.onrender.com/health](https://hiver-support-agent.onrender.com/health)
> - First request after idle takes ~30-60s (cold start). Subsequent requests: ~2-3s.
> - **Note:** Uses template-based replies (no OpenAI API key). Responses are grounded in historical Uber brand responses.

### Quick test

```bash
curl -X POST https://hiver-support-agent.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my driver? I have been waiting for 20 minutes."}'
```

---

## 1. Results

### Human-Grounded Evaluation (Verified)

> **These results are computed against 200 independently human-labelled examples. This is the authoritative evaluation.**

| Metric | Baseline | Escalation v2 |
|--------|----------|---------------|
| Golden set size | 200 examples | 200 examples |
| Intent Accuracy | 0.560 | 0.560 |
| Intent Macro-F1 | **0.419** | **0.419** |
| Intent Weighted-F1 | 0.513 | 0.513 |
| Escalation Accuracy | 0.555 | **0.610** |
| Escalation Precision | 0.161 | **0.271** |
| Escalation Recall | 0.176 | **0.314** |
| Escalation F1 | 0.168 | **0.291** |
| Tests | — | 32/32 ✓ |

### Per-Intent Classification (TF-IDF Baseline, Human Labels)

| Intent | Precision | Recall | F1 | Support |
|--------|:-:|:-:|:-:|:-:|
| account_access | 0.667 | 0.500 | 0.571 | 20 |
| app_functionality | 0.667 | 0.545 | 0.600 | 11 |
| cancellation | 0.000 | 0.000 | 0.000 | 11 |
| complaint | 0.800 | 0.211 | 0.333 | 19 |
| other | 0.607 | 0.841 | 0.705 | 44 |
| refund_request | 1.000 | 0.091 | 0.167 | 11 |
| ride_status | 0.500 | 0.770 | 0.606 | 61 |
| trip_fare_dispute | 0.467 | 0.304 | 0.368 | 23 |

### Earlier Pseudo-Label Evaluation (Misleading — Do Not Cite)

> **The following numbers were computed against automatically generated keyword pseudo-labels, NOT human labels. They are inflated and misleading. Included only for historical traceability.**

| System | Intent Accuracy | Macro-F1 | Weighted-F1 |
|--------|:-:|:-:|:-:|
| TF-IDF + LogReg (pseudo-labels) | 0.905 | 0.811 | 0.903 |

**Why these numbers are misleading:** The earlier headline Macro-F1 of 0.811 measures agreement with keyword heuristics, not real intent understanding. Human annotation revealed 35% disagreement with auto-labels. The real Macro-F1 against human labels is 0.419.

### Escalation: Baseline vs V2 (Same Golden Set)

| Config | Accuracy | Precision | Recall | F1 |
|--------|:-:|:-:|:-:|:-:|
| Pipeline baseline (multi-factor) | 0.555 | 0.161 | 0.176 | 0.168 |
| Escalation V2 (rule-enhanced) | 0.610 | 0.271 | 0.314 | 0.291 |

Both evaluated against the same 200 human-labelled examples. V2 improves recall from 0.176 to 0.314 (+79%) and F1 from 0.168 to 0.291 (+73%).

## 2. Problem & Approach

Build a brand-specific AI customer-support agent that:
1. Classifies customer messages into an Uber-specific intent taxonomy
2. Drafts replies grounded in how Uber historically resolved similar issues
3. Decides whether to auto-handle or escalate, with an explicit reason

**Selected brand:** `Uber_Support` from the Kaggle [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset (2.8M tweets, 108 brands).

## 3. Dataset & Brand Selection

### Dataset

**Source:** Kaggle "Customer Support on Twitter" (twcs.csv, 516 MB, 2,811,774 rows).

### Brand Comparison

| Brand | Brand Msgs | Cust Msgs | Unique Cust | 3+Turn Threads | Intents (>=10) | Score |
|-------|-----------|-----------|-------------|---------------|----------------|-------|
| AmazonHelp | 169,840 | 100,503 | 40,671 | 70,956 | 8 | 220,092 |
| AppleSupport | 106,860 | 36,658 | 23,668 | 23,388 | 5 | 125,189 |
| **Uber_Support** | **56,270** | **71,349** | **~12,780** | **41,932** | **8** | **67,350** |
| SpotifyCares | 43,265 | 15,096 | 8,723 | 11,938 | 8 | 50,813 |
| AmericanAir | 36,764 | 18,045 | 9,699 | 9,209 | 8 | 45,786 |

### Why Uber_Support?

1. **Sufficient volume:** 56K brand messages + 71K customer messages = ample training data
2. **Multi-turn conversations:** 41,932 threads with brand responses
3. **Intent diversity:** 8 intents with good coverage of ride-hailing support themes
4. **Reasonable cost:** ~50% smaller than AmazonHelp, making evaluation reproducible on a laptop
5. **Realistic domain:** Rideshare support covers payments, accounts, complaints, cancellations — diverse and practical

### Data Splits (Temporal, Conversation-Aware)

| Split | Conversations | Purpose |
|-------|--------------|---------|
| Train | 35,473 | Classifier training + retrieval index |
| Dev | 6,259 | Hyperparameter tuning |
| Golden | 200 | Evaluation (held out, most recent) |

**Split methodology:** Conversations sorted chronologically. Golden = most recent 200 conversations. Train = earliest 70% of remaining. Dev = latest 15% of remaining. No conversation appears in multiple splits. Zero leakage verified (0 overlapping messages between golden and train/dev).

## 4. Uber-Specific Intent Taxonomy

Derived from real Uber_Support customer messages (keyword frequency analysis of 41,932 conversations):

| Intent | Count | % | Description |
|--------|-------|---|-------------|
| ride_status | 20,620 | 49.2% | Ride/trip status, driver location, wait times, cancellations by drivers, lost items |
| other | 10,068 | 24.0% | General inquiries, positive feedback, promotional content, ambiguous messages |
| account_access | 4,107 | 9.8% | Login problems, account deactivation/banning, verification, phone/email changes |
| app_functionality | 3,023 | 7.2% | App crashes, GPS issues, map errors, feature malfunctions |
| trip_fare_dispute | 2,504 | 6.0% | Surge pricing, fare disputes, overcharges, receipt issues |
| complaint | 1,172 | 2.8% | Service dissatisfaction, driver behavior complaints, negative feedback |
| refund_request | 353 | 0.8% | Explicit refund/money-back requests |
| cancellation | 85 | 0.2% | Cancel rides, accounts, subscriptions |

### Intent Definitions

**ride_status** — Any message about the current or past ride status: where is my driver, how long until pickup, driver cancelled, ride ended prematurely, wrong destination, ride not showing in app, ETA questions. Includes lost items in vehicles and driver behavior during rides.

**other** — Messages that cannot be clearly assigned to another intent: general inquiries about Uber services, positive feedback, promotional/spam content, off-topic messages, ambiguous requests.

**account_access** — Account-related issues: login problems, account deactivation/banning, OTP/verification failures, phone number or email changes, document submission for driver accounts, account security concerns.

**app_functionality** — Technical issues with the Uber or Uber EATS app: crashes, bugs, GPS problems, map errors, feature malfunctions, update issues, notification failures.

**trip_fare_dispute** — Disputes about charges, surge pricing, fare amounts, double charges, billing errors, receipt issues. Messages where the customer believes they were charged incorrectly.

**complaint** — Expressions of dissatisfaction without a specific actionable request: rude drivers, bad experiences, frustration, threatening to switch competitors.

**refund_request** — Explicit requests for refunds, money back, or charge reversals. Primary intent is getting money returned.

**cancellation** — Requests to cancel rides, accounts, subscriptions, or Uber EATS orders. Includes cancellation fee complaints.

### Representative Examples

| Intent | Example |
|--------|---------|
| ride_status | "@Uber_Support I've been waiting 30 mins for my driver. The app says 5 minutes but he hasn't moved." |
| ride_status | "My driver cancelled on me for the third time tonight. I'm stranded downtown." |
| ride_status | "@Uber_Support please answer your DMs, I have lost my phone I believe in one of your drivers cars" |
| trip_fare_dispute | "@Uber_Support I was charged $36 for a trip that normally costs $11-$15." |
| trip_fare_dispute | "Charged twice for the same trip. My bank shows two identical charges." |
| trip_fare_dispute | "@Uber_Support pls help i've been charged 16.66 for an uber i never took" |
| account_access | "@Uber_Support I been trying to access my account but Uber keeps telling me to update my number" |
| account_access | "@Uber_Support, I'm facing issue in login, not getting OTP on my cell phone." |
| account_access | "Um why is my uber account banned..?" |
| app_functionality | "@Uber_Support The app does not show a picture of my car for passengers to see! Please fix!" |
| app_functionality | "@Uber_Support your apps are absolutely uslesss" |
| app_functionality | "@115873 I'm unable to book cab by paytm or visa. Its showing not applicable to use in web version." |
| complaint | "@115873 is this a scam? It doesn't work." |
| complaint | "Had a terrible experience with you. Food was unacceptable and freezing cold." |
| complaint | "@115873 ordered an uber to take my tv home, a zafira turned up, waved his hand, shook his head & drove off" |
| refund_request | "@115873 give me my damn money back" |
| refund_request | "Thanks for the refund, but I'm still missing half a meal." |
| cancellation | "Why did UberEATS cancel my order" |
| cancellation | "@Uber_Support Not being able to cancel my order a second after accidentally ordering is absolutely ridiculous" |

## 5. System Architecture

```
Customer Message
      |
      v
Preprocessing (lowercase, remove URLs/mentions, strip)
      |
      v
Intent Classification (TF-IDF + Logistic Regression, 5000 features, bigrams)
      |
      v
Historical Case Retrieval (all-MiniLM-L6-v2 + FAISS, top-5 + intent-specific)
      |
      v
Resolution Action Extraction (parse brand responses for resolution patterns)
      |
      v
Response Generation (grounded in historical brand responses; template fallback)
      |
      v
Escalation Policy (6-factor weighted scoring, threshold=0.3)
      |
      v
Structured Output {intent, confidence, reply, should_escalate, escalation_reason, evidence}
```

## 6. Baselines

### Baseline 1: Trivial (Majority Class)
- Always predicts `ride_status` (49.2% of training data)
- Never escalates
- **Intent Accuracy:** 0.485 | **Macro-F1:** 0.093 | **Weighted-F1:** 0.317 | **Escalation F1:** 0.000
- *(pseudo-label baseline — included for historical context only)*

### Baseline 2: TF-IDF + Logistic Regression (Human-Labelled)
- TF-IDF vectorizer (5000 features, bigrams) + Logistic Regression
- Simple confidence-based escalation (threshold: 0.5)
- **Intent Accuracy:** 0.560 | **Macro-F1:** 0.419 | **Weighted-F1:** 0.513
- **Escalation:** Accuracy 0.685, Precision 0.167, Recall 0.059, F1 0.087
- *(evaluated against 200 human-labelled examples)*

### Full Pipeline
- Same classifier + multi-factor escalation + retrieval-grounded generation
- Intent metrics identical to Baseline 2 (same underlying classifier)
- **Escalation (pipeline baseline):** Accuracy 0.555, Precision 0.161, Recall 0.176, F1 0.168
- **Escalation (V2 rule-enhanced):** Accuracy 0.610, Precision 0.271, Recall 0.314, F1 0.291

## 7. Evaluation Methodology

### Metrics
- **Intent classification:** Accuracy, Macro-F1, Weighted-F1, per-class precision/recall/F1
- **Escalation:** Precision, Recall, F1
- **Reply quality:** LLM-as-judge on 6 dimensions (relevance, groundedness, correctness, completeness, tone, hallucination risk) — **Mock mode only** (no API key available)

### Golden Set

- **Size:** 200 examples (verified — `EXPECTED_ROWS = 200`)
- **Sampling:** Stratified temporal holdout from held-out conversations
- **Labels:** Independently human-annotated with intent, escalation decision, and escalation reasons
- **Escalation rate:** 25.5% (51/200) — human-labelled
- **Status:** Fully annotated, validated, no blank labels
- **Leakage check:** 0 messages overlap between golden and train/dev splits
- **No regeneration:** annotation.csv was not regenerated or modified after human labelling

## 8. What is misleading about my headline number?

> **The real headline: 56.0% accuracy / 41.9% macro-F1** — against 200 independently human-labelled examples. The earlier 90.5% / 81.1% numbers were misleading. Here is why.

### The two headlines

| Metric | Pseudo-label eval (misleading) | Human-label eval (official) |
|--------|:-:|:-:|
| Intent Accuracy | 0.905 | **0.560** |
| Intent Macro-F1 | 0.811 | **0.419** |
| Escalation F1 | 0.000 | **0.291** (V2) |

### Why the drop happened

The model was evaluated against labels it co-created. Intent labels were generated by keyword matching (e.g., "cancel" → cancellation, "refund" → refund_request). The classifier was then trained on these pseudo-labels and tested against them — measuring agreement with its own training heuristics, not real intent understanding.

Human annotation of 200 held-out examples revealed **35% disagreement** between pseudo-labels and human labels (70/200 cases). The model had learned to mimic keyword rules, not to understand customer intent.

### The key failure pattern: ride_status absorbs everything

The single dominant failure mode is that `ride_status` acts as a black hole, absorbing messages that should be classified as `complaint`, `cancellation`, `trip_fare_dispute`, `account_access`, and `refund_request`. This is invisible in pseudo-label evaluation because the keyword heuristics don't distinguish these either.

**Confusion matrix (rows=true, cols=predicted):**

```
             acct  app  canc  compl  other  refnd  ride  fare
ride_status   3    0     0     0     10      0    47     1
complaint     0    0     0     4      4      0    11     0
cancellation  0    0     0     0      0      0    10     1
trip_fare     2    0     0     1      3      0    10     7
account_acc   10   2     0     0      2      0     6     0
refund_req    0    0     0     0      1      1     4     5
```

`ride_status` absorbs: **11/19 complaints**, **10/11 cancellations**, **10/23 fare disputes**, **6/20 account access issues**.

### Concrete examples of the mismatch

| True intent | Model predicted | Message | Why it's misleading |
|-------------|----------------|---------|-------------------|
| complaint | ride_status | "Your driver just literally pulled into the parking lot I'm waiting at, drove out, called me after he left and didn't say anything on phone! How do I file Official complaint" | Contains "ride" context but primary intent is filing a complaint about driver behavior |
| trip_fare_dispute | ride_status | "I was charged $75 for a $10 ride how do I dispute or find out why I was overcharged" | Mentions "ride" but the core issue is a fare dispute |
| cancellation | ride_status | "I just waited over 10 mins for a car that was supposed to be 3 mins away. It drove in circles. Then cancelled on me. And somehow I am charged 6 euro???? Please fix this." | Customer-initiated cancellation with fee complaint, not a ride status inquiry |
| account_access | ride_status | "My account has been disabled and I haven't even used it for the first time!!! I can not even sign in!! Please heeeeeelp" | Account access issue with zero ride context, but classifier defaults to ride_status |
| refund_request | ride_status | "You charged my credit card 3 times for 1 ride. Can I get refunded for my other 2 charges" | Explicit refund request, but "ride" in text triggers ride_status |

### Bottom line

The earlier numbers (90.5% accuracy, 81.1% macro-F1) measured how well the classifier agrees with keyword heuristics. The real numbers (56.0% accuracy, 0.419 macro-F1) measure how well it classifies actual human intent. The gap is not a bug in evaluation — it is the real performance picture that pseudo-labels concealed.

## 9. Failure Analysis (Human-Labelled Golden Set)

### Summary

| Metric | Count | % of 200 |
|--------|-------|----------|
| Intent failures | 88 | 44.0% |
| Escalation failures | 63 | 31.5% |
| Missed escalations | 48 | 24.0% |
| False escalations | 15 | 7.5% |

### Top 5 Failure Modes

#### Failure 1: complaint → ride_status (11 cases, 5.5%)

**Concrete example:**
> "@Uber_Support your driver just literally pulled into the parking lot I'm waiting at, drove out, called me after he left and didn't say anything on phone! How do I file Official complaint as I stand here waiting for another driver."

Human label: **complaint**. Model prediction: **ride_status**.

**Why it failed:** The message contains "driver", "waiting", "ride" vocabulary that activates ride_status features. The classifier weights ride-related terms heavily because ride_status is 49.2% of training data, and the word "driver" appears frequently in both ride_status and complaint contexts. The classifier never learned to distinguish "driver behavior complaint" from "ride status inquiry."

**Root cause:** TF-IDF features cannot disambiguate when the same vocabulary ("driver", "ride", "waiting") serves different intents. The model's prior probability for ride_status (49.2%) overwhelms weaker complaint signals like "file Official complaint" or profanity indicating dissatisfaction.

**Proposed fix:** Add a dedicated complaint feature set (sentiment lexicons, "file complaint" n-grams, profanity markers) as separate TF-IDF dimensions, or switch to a model with attention that can weigh "file complaint" more heavily than "driver" context. Multi-label classification could also help — this message is legitimately both ride_status-adjacent and a complaint.

---

#### Failure 2: trip_fare_dispute → ride_status (10 cases, 5.0%)

**Concrete example:**
> "@Uber_Support I was charged $75 for a $10 ride how do I dispute or find out why I was overcharged? Why is there no way to speak directly with someone very frustrating"

Human label: **trip_fare_dispute**. Model prediction: **ride_status**.

**Why it failed:** The word "ride" appears in the text, and the message discusses a trip. The classifier detects ride_status vocabulary ("charged", "ride") and cannot distinguish that the primary intent is disputing a fare — not checking ride status. The $75 amount and "overcharged" signal are present but not weighted strongly enough against the dominant ride_status prior.

**Root cause:** Fare disputes inherently reference rides, creating vocabulary overlap. The TF-IDF model has no mechanism to understand that "charged $75 for a $10 ride" is fundamentally about money, not about ride logistics. The keyword pseudo-labels likely classified this as ride_status too, so the model never learned the distinction.

**Proposed fix:** Add price-related features ("charged", dollar amounts, "overcharged", "dispute", "refund") as explicit indicators for trip_fare_dispute. Consider a two-stage classifier: first detect if money is involved, then classify the specific money-related intent.

---

#### Failure 3: ride_status → other (10 cases, 5.0%)

**Concrete example:**
> "@Uber_Support is there an easy way to know whether a trip has GST on it or not?"

Human label: **ride_status**. Model prediction: **other**.

**Why it failed:** The message is a general inquiry about trip taxation. It contains no strong ride_status keywords ("where is my driver", "cancelled", "ETA") and no strong other keywords either. The classifier defaults to other (24% prior) when ride_status signals are weak. The message is genuinely ambiguous — it references a trip but asks a general policy question.

**Root cause:** Some ride_status messages are genuine general inquiries about rides (pricing, policies, tax) that don't fit the "where is my driver / ride status check" archetype the classifier learned. These edge cases fall through to other because they lack the high-frequency ride_status trigger words.

**Proposed fix:** Expand ride_status training examples to include general ride inquiries (pricing, policies, receipts). Alternatively, add a "ride_inquiry" sub-intent or use conversation context — if the preceding messages are about a specific ride, this is more likely ride_status.

---

#### Failure 4: cancellation → ride_status (10 cases, 10/11 cancellations)

**Concrete example:**
> "@115873 I just waited over 10 mins for a car that was supposed to be 3 mins away. It drove in circles. Then cancelled on me. And somehow I am charged 6 euro???? Please fix this."

Human label: **cancellation**. Model prediction: **ride_status**.

**Why it failed:** The message describes a ride scenario (waiting, driver location, car) with "cancelled" as one of many events. The classifier treats "cancelled" as a ride_status event (driver cancelled) rather than a customer-initiated cancellation request. The "charged 6 euro" and "Please fix this" signals are present but drowned out by ride vocabulary.

**Root cause:** Cancellation is only 0.2% of training data (85 examples) vs. ride_status at 49.2%. The classifier has almost no cancellation examples to learn from. Furthermore, cancellation messages in Uber's domain often describe ride events ("my driver cancelled", "I had to cancel"), making the boundary genuinely ambiguous. The human label distinguishes "customer requesting cancellation/fee dispute" from "ride_status: driver cancelled event" — a distinction the model cannot make.

**Proposed fix:** Oversample cancellation examples during training, or use class weights. Add cancellation-specific features ("cancel fee", "had to cancel", "charged for cancel"). Consider whether cancellation should be a sub-intent of ride_status rather than a separate class — the human annotators clearly distinguish these, but the vocabulary overlap makes it very hard for a bag-of-words model.

---

#### Failure 5: account_access → ride_status (6 cases, 3.0%)

**Concrete example:**
> "@Uber_Support My account has been disabled and I haven't even used it for the first time!!! I am suposed to have a Free Uber ride from a friend. Who do I contact?? I can not even sign in!! Please heeeeeelp"

Human label: **account_access**. Model prediction: **ride_status**.

**Why it failed:** The message mentions "Uber ride" and is addressed to @Uber_Support, triggering ride_status features. The classifier cannot distinguish that "Free Uber ride from a friend" is context for why the account issue matters — not the primary intent. "Account has been disabled" and "can not even sign in" are strong account_access signals but are outweighed by the ride_status prior.

**Root cause:** Account access messages often mention rides as context ("I can't use my account to book a ride"), and the TF-IDF model cannot parse that ride is secondary to account. Additionally, account_access is 9.8% of training data — 5x smaller than ride_status — giving it less weight in the feature space.

**Proposed fix:** Add account-specific features ("account disabled", "can't sign in", "login", "verification code", "banned") as strong negative indicators for ride_status. A rule-based pre-filter that detects account keywords before classification could catch these cases.

## 10. LLM-as-Judge Methodology

**Status: MOCK MODE ONLY**

No `OPENAI_API_KEY` is available in the environment. The existing `experiments/llm_judge_results.json` contains **mock** (deterministic, non-LLM) scores. These must NOT be presented as real LLM evaluation.

The LLM judge harness exists (`evaluation/llm_judge.py`) and supports two modes:

- **mock mode (current):** Deterministic scoring based on text analysis (keyword overlap, evidence matching, tone indicators). NOT a real LLM evaluation.
- **openai mode:** Real LLM judging via GPT-3.5-turbo with structured prompting. Requires `OPENAI_API_KEY`.

**To enable real LLM judging:**
```bash
set OPENAI_API_KEY=your-key-here
python evaluation/run_llm_judge.py --mode openai
```

**Rubric (1-5 scale):**
1. Relevance — Does the reply address the customer issue?
2. Groundedness — Is the reply supported by historical evidence?
3. Correctness — Is the reply factually consistent?
4. Completeness — Does the reply adequately address the issue?
5. Tone — Is the reply professional and empathetic?
6. Hallucination risk — Does the reply invent unsupported claims?

### Human/Judge Agreement (⚠️ Against MOCK Judge Only)

**Human reply-quality ratings COMPLETED** (30/30 examples, 180 dimension comparisons).
Agreement metrics compare human scores against **mock** deterministic judge (not real OpenAI GPT).

| Metric | Value |
|--------|-------|
| Examples | 30 |
| Exact agreement rate | 43.3% |
| Weighted agreement (±1 point) | **80.6%** |

| Dimension | Spearman r | Exact | Weighted |
|-----------|-----------|-------|----------|
| Relevance | 0.033 | 33.3% | 86.7% |
| Groundedness | **0.688** | 80.0% | 100.0% |
| Correctness | 0.369 | 3.3% | 66.7% |
| Completeness | -0.112 | 10.0% | 33.3% |
| Tone | **0.601** | 76.7% | 96.7% |
| Hallucination Risk | **0.661** | 56.7% | 100.0% |

Note: Completeness Spearman = -0.112 indicates mock judge overestimates completeness where human
annotators penalise unfilled `{resolution_details}` placeholders. These numbers characterize mock
judge calibration only — they are NOT real LLM-as-judge agreement statistics.

## 11. Reproduction (<15 minutes)

### Without the raw dataset (headline metrics only)

The 200-example golden set (`evaluation/human_annotation/annotation.csv`) and training data
(`data/processed/train.csv`) are committed. You can reproduce all headline metrics without
the 500 MB Kaggle download:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run headline evaluation (accuracy, F1, per-class breakdown)
#    ~2-3 minutes on a laptop
python evaluation/eval_human_labels.py --with-eval

# 3. Run failure analysis
python evaluation/failure_analysis_human.py

# 4. Run grounding audit
python evaluation/grounding_audit.py

# 5. Run escalation audit (baseline vs V2)
python evaluation/escalation_audit.py

# 6. Run tests
python -m pytest tests/ -v
```

### With the raw dataset (full pipeline rebuild)

If you want to rebuild everything from the raw Kaggle data:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset (requires Kaggle API key)
pip install kaggle
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw/
unzip data/raw/*.zip -d data/raw/

# 3. Build real data pipeline (extracts Uber_Support, builds conversations,
#    derives taxonomy, creates splits, prepares golden set)
#    ~3-5 minutes on a laptop
python scripts/build_real_pipeline.py

# 4. Run full evaluation
python evaluation/run_real_eval.py
```

### CLI demo and API

```bash
# Run CLI demo
python -m src --message "Where is my driver?"

# Start API server locally
python -m uvicorn api.main:app --reload
# Visit http://localhost:8000/docs

# Run 3-scenario structured demo (local or live API)
python demo.py --local    # Direct pipeline, no server
python demo.py --live     # Requires server running on localhost:8000
```

**Live API:** [https://hiver-support-agent.onrender.com/docs](https://hiver-support-agent.onrender.com/docs)

**Note:** The sentence-transformer model (all-MiniLM-L6-v2, ~80MB) is downloaded on first run.

## 12. Golden-Set Sampling & Labeling Note

### Sampling Strategy
- **Method:** Stratified sampling from the 200 held-out conversations (temporal holdout)
- **Reclassification:** Messages with strong cancellation/refund signals were relabelled from original pseudo-labels before sampling
- **Randomization:** Final ordering is shuffled (seed=42), not sorted by auto_intent
- **Conversation-aware:** Entire conversations stay in one split (no leakage)
- **Separation:** Golden set is completely separate from training and retrieval corpus

### Final auto_intent Distribution

| Intent | Count |
|--------|-------|
| ride_status | 92 |
| other | 51 |
| account_access | 18 |
| trip_fare_dispute | 11 |
| app_functionality | 10 |
| refund_request | 9 |
| complaint | 6 |
| cancellation | 3 |

**Limitations:** `cancellation` (3) and `app_functionality` (10) are underrepresented due to low availability in the held-out pool. The `auto_intent` column is a sampling aid, not ground truth — human annotation provides the real labels.

### Labeling Status — COMPLETE ✓
- **200/200 examples human-reviewed and annotated** — annotation finalized and locked
- **human_intent**: 200/200 filled, using all 8 intents
- **human_should_escalate**: 200/200 filled, 51 escalated (25.5%)
- **human_escalation_reason**: 51/51 escalated items have reasons
- **annotator_notes**: 200/200 filled with detailed rationale
- **All final metrics** are computed against human labels in `annotation.csv` — not against auto-labels

### Annotation Template Fields

| Field | Description |
|-------|-------------|
| example_id | Unique identifier for each example |
| customer_message | The actual customer message text |
| auto_intent | Auto-generated intent label (keyword matching) |
| human_intent | Human-annotated intent label |
| auto_should_escalate | Auto-generated escalation label (heuristic) |
| human_should_escalate | Human-annotated escalation decision (True/False) |
| human_escalation_reason | Human-annotated escalation reason |
| annotator_notes | Free-text notes from annotation |

### Data Leakage Prevention
The golden set must NOT enter:
- Training data (verified: 0 overlapping messages)
- Retrieval corpus
- Hyperparameter tuning

## 13. Decision Log (Non-Obvious Decisions)

| # | Decision | Alternatives considered | Why chosen | Trade-off |
|---|----------|------------------------|------------|-----------|
| 1 | **Uber_Support** as brand | AmazonHelp (170K msgs), AppleSupport (37K), SpotifyCares (15K) | AmazonHelp is 3x larger making evaluation slow on a laptop; AppleSupport is 93% tech support (poor intent diversity); Uber has 8 intents with good volume and realistic rideshare domain | Reproducibility and speed vs. raw data volume |
| 2 | **8 Uber-specific intents** derived from data | Generic e-commerce intents (order, delivery, billing), 15+ fine-grained intents | Derived from keyword frequency analysis of 41K conversations; 8 intents cover the domain without fragmenting small classes below 10 examples each | Domain specificity vs. generic transferability |
| 3 | **Conversation-aware + temporal splitting** | Random row split, random conversation split, time-based row split | Temporal split simulates real deployment (train on past, predict future); conversation-aware prevents data leakage (same conversation in train and test destroys validity) | Realism and leakage prevention vs. simplicity |
| 4 | **200 golden examples** | 100, 150, 250, 500 | 200 is within the 150-250 range for statistical meaningfulness with 8 intents; 11 examples per smallest class (cancellation, refund_request) gives ~±8% margin; 500 would require disproportionate human annotation time | Statistical power vs. human annotation burden |
| 5 | **Not the entire dataset** for golden set | Use all held-out examples (4K+) | Golden set must be human-annotated; 200 examples are feasible for manual review in hours; 4K+ would take days and introduce annotator fatigue errors | Annotation quality vs. coverage |
| 6 | **TF-IDF + Logistic Regression** baseline | BERT, sentence-transformers for classification, zero-shot LLM classification | Fast (trains in seconds), interpretable (feature weights visible), works well with 35K training examples, no GPU needed; establishes a reproducible baseline before trying expensive models | Simplicity and speed vs. accuracy ceiling |
| 7 | **Historical response retrieval** via FAISS + all-MiniLM-L6-v2 | Pure template responses, OpenAI generation without retrieval, keyword-only retrieval | Grounds responses in how Uber actually replied to similar issues; semantic search finds relevant cases even with different wording; top-5 provides enough context without noise | Grounding quality vs. retrieval infrastructure complexity |
| 8 | **Resolution action extraction** from brand responses | Generic template responses, LLM-generated responses from scratch | Extracting resolution patterns from real brand responses produces responses that match Uber's actual tone and procedures; more grounded than pure templates or zero-shot generation | Response quality vs. extraction complexity |
| 9 | **Escalation separate from intent** | Joint intent+escalation model, escalation as intent sub-class | Escalation depends on factors beyond intent (urgency, safety keywords, monetary thresholds, confidence); keeping it separate allows rule-based escalation logic that can be tuned independently of classification | Flexibility vs. end-to-end optimization |
| 10 | **Rules for Escalation V2** | Learned escalation classifier, threshold-only escalation | Rule-based escalation is interpretable and auditable (critical for safety decisions); can encode domain knowledge (safety keywords = always escalate) that a classifier would need大量 data to learn; V2 improved F1 from 0.168 to 0.291 | Interpretability vs. learned flexibility |
| 11 | **Human review instead of trusting pseudo-labels** | Use pseudo-labels as ground truth, LLM-as-judge for labels | Pseudo-labels had 35% disagreement with human annotation; the classifier was being evaluated against its own training heuristics; human review revealed the real 0.419 macro-F1 vs. misleading 0.811 | Honest evaluation vs. flattering metrics |
| 12 | **No synthetic golden examples** | Generate synthetic test examples with LLM, augment golden set with paraphrases | Synthetic examples would not reflect real customer language; the golden set's value is measuring performance on actual tweets; synthetic data would reintroduce the same pseudo-label problem | Data authenticity vs. larger evaluation set |
| 13 | **LLM-as-judge** with mock fallback | Human-only reply quality evaluation, skip reply quality assessment | LLM-as-judge scales to all 200 examples; mock mode (deterministic heuristic scoring) is available when API key is unavailable; human/judge agreement measured on 30 examples to calibrate | Scalability vs. evaluation fidelity |
| 14 | **Report failures instead of optimizing headline metrics** | Only report accuracy/F1, hide per-class breakdown | Reporting the 0.419 macro-F1 and per-class failures (cancellation F1=0.000, complaint F1=0.333) is more useful for improvement than a single headline number; stakeholders need to know where the system fails | Transparency vs. perceived performance |
| 15 | **FAISS with embedding caching** | Brute-force cosine similarity, precomputed distance matrix | FAISS indexes 35K vectors in <1s and queries in <10ms; caching embeddings across runs avoids recomputing 35K sentence-transformer embeddings (~5 min); brute-force works but scales poorly | Setup complexity vs. query speed and reproducibility |

## 14. Next-Week Improvement Plan

If I had another week, these are the concrete next steps in priority order:

1. **Intent boundary detection (cancellation / fare / complaint vs ride_status)** — The #1 failure mode is ride_status absorbing 36 messages across 4 intents. Add intent-specific keyword pre-filters (e.g., "cancel fee" → cancellation, "charged $X" → trip_fare_dispute, "file complaint" → complaint) before the classifier. Target: reduce ride_status absorption from 36 to <15 cases.

2. **Confidence calibration** — Current confidence scores are uncalibrated (a 0.8 confidence does not mean 80% true probability). Apply temperature scaling or Platt scaling on the dev set. This improves escalation decisions (which depend on confidence thresholds) and enables abstention ("I'm not sure, escalating to human").

3. **Stronger resolution extraction** — Current extraction pulls brand_response text verbatim from retrieved cases. Improve with pattern matching: identify resolution actions (refund issued, investigation opened, account reactivated) and parameterize them into response templates. Target: fill the `{resolution_details}` placeholder in >80% of responses (currently <30%).

4. **Learned / calibrated escalation** — Replace the 6-factor rule-based escalation with a lightweight classifier (LogReg on the same 6 features + intent). Train on the 200 human-annotated escalation labels. Target: improve escalation F1 from 0.291 to >0.40 while maintaining recall >0.30.

5. **Expand human golden set to 400 examples** — Current 200 examples leave cancellation (11) and refund_request (11) with wide confidence intervals. Add 200 more examples, oversampling underrepresented intents. Include inter-annotator agreement (2 annotators on 50 overlap examples) to measure label quality.

6. **Multi-brand validation** — Test the pipeline on AmazonHelp and AppleSupport without retraining (zero-shot transfer). Measure: does the intent taxonomy transfer? Does escalation logic generalize? This reveals whether the system is Uber-specific or truly brand-agnostic.

7. **Response safety and PII checks** — Add post-generation checks: detect PII in responses (phone numbers, email addresses, account details), verify no financial commitments are made without authorization, ensure responses don't contain harmful content. Critical before any production deployment.

8. **Retrieval quality metrics** — Currently retrieval is a black box (top-5 cases → generation). Add retrieval evaluation: measure precision@k of retrieved cases against human-judged relevance. If retrieval quality is low, the grounded generation is grounded in irrelevant evidence.

9. **Temporal drift testing** — The training data is from 2017. Test on newer Uber support conversations (if available) to measure how much intent patterns, vocabulary, and escalation needs have changed over time. This reveals whether the model is time-bound to 2017 language.

10. **Class-weighted / focal loss training** — Address the 49.2% ride_status dominance by applying class weights (inverse frequency) or focal loss during LogReg training. This should improve recall on cancellation, complaint, and refund_request without degrading ride_status performance.

## 15. Citations

- **Dataset:** Customer Support on Twitter. https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- **Embedding model:** Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. `all-MiniLM-L6-v2` via `sentence-transformers`
- **FAISS:** Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*
- **scikit-learn:** Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR 12*, pp. 2825-2830
- **FastAPI:** tiangolo (2018). FastAPI framework. https://fastapi.tiangolo.com/

## Project Structure

```
hiver/
├── configs/
│   ├── brand.yaml              # Brand config (Uber_Support)
│   └── intents.yaml            # Uber-specific intent taxonomy (8 intents)
├── src/
│   ├── classifier.py           # TF-IDF + LogReg intent classifier
│   ├── retrieval.py            # Sentence-transformer + FAISS retrieval
│   ├── generator.py            # Template/OpenAI response generation
│   ├── escalation.py           # Multi-factor escalation policy
│   ├── pipeline.py             # Main pipeline orchestrator
│   ├── preprocessing.py        # Text cleaning utilities
│   ├── taxonomy.py             # Intent taxonomy derivation
│   └── __main__.py             # CLI entry point
├── evaluation/
│   ├── baselines.py            # Trivial + TF-IDF baselines
│   ├── run_eval.py             # Synthetic evaluation harness
│   ├── run_real_eval.py        # Real data evaluation harness
│   ├── llm_judge.py            # LLM-as-judge (mock + OpenAI)
│   ├── human_judge_agreement.py # Human/judge agreement framework
│   ├── failure_analysis.py     # Failure mode analysis
│   ├── ANNOTATION_GUIDE.md     # Uber-specific labeling instructions
│   └── golden_set_annotation_template.csv
├── scripts/
│   ├── build_real_pipeline.py  # Full real-data pipeline
│   ├── load_real_data.py       # Real data loader
│   ├── analyze_brands_real.py  # Brand analysis (real data)
│   └── prepare_data.py         # Synthetic data preparation
├── api/
│   └── main.py                 # FastAPI REST endpoint
├── tests/
│   ├── test_classifier.py      # Classifier unit tests
│   └── test_comprehensive.py   # Comprehensive tests
├── data/
│   ├── raw/twcs/twcs.csv       # Kaggle dataset (NOT in git)
│   ├── processed/              # Train/dev/golden splits (NOT in git)
│   └── embeddings_cache/       # FAISS index cache (NOT in git)
├── experiments/
│   ├── real_results.json       # Real evaluation results
│   └── failure_analysis_real.md
├── requirements.txt
├── .gitignore
└── README.md
```
