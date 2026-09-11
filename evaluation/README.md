# Golden Evaluation Set

This directory contains the evaluation dataset for the Uber customer support AI agent.

## Dataset Overview

The golden set contains **200 examples** sampled via **temporal holdout** from the most recent Uber_Support conversations. Each example includes a customer message with auto-generated labels for initial development.

**⚠ Status: Auto-generated labels only — NOT hand-labelled.**

## Annotation Status

| Field | Status | Description |
|-------|--------|-------------|
| `auto_intent` | ✅ Generated | Intent label from keyword matching |
| `human_intent` | ❌ Not filled | **Must be filled by human annotator** |
| `auto_should_escalate` | ✅ Generated | Escalation label from heuristic rules |
| `human_should_escalate` | ❌ Not filled | **Must be filled by human annotator** |
| `human_escalation_reason` | ❌ Not filled | **Must be filled by human annotator** |
| `annotator_notes` | ❌ Not filled | Free-text notes from annotation |

## Uber-Specific Intent Taxonomy (8 Intents)

| Intent | Description |
|--------|-------------|
| ride_status | Ride/trip status, driver location, wait times, cancellations, lost items |
| other | General inquiries, positive feedback, promotional content, ambiguous messages |
| account_access | Login problems, account deactivation/banning, verification issues |
| app_functionality | App crashes, GPS issues, map errors, feature malfunctions |
| trip_fare_dispute | Surge pricing, fare disputes, overcharges, receipt issues |
| complaint | Service dissatisfaction, driver behavior complaints, negative feedback |
| refund_request | Explicit refund/money-back requests |
| cancellation | Cancel rides, accounts, subscriptions |

## Annotation Instructions

1. Read the [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) for full instructions
2. Open `golden_set_annotation_template.csv` in a spreadsheet editor
3. For each example, fill in the `human_intent` and `human_should_escalate` columns
4. Optionally add `human_escalation_reason` and `annotator_notes`
5. Save the completed file as `golden_set_annotated.csv`

## Data Format

The golden set is stored in `golden_set.csv` with the following columns:

- `example_id`: Unique identifier for each example
- `conversation_id`: ID of the conversation thread
- `customer_message`: The actual customer message text
- `brand_response`: The brand response (if available)
- `brand_name`: Always "Uber_Support"
- `auto_intent`: Auto-generated intent label (keyword matching)
- `auto_should_escalate`: Auto-generated escalation label (heuristic)
- `human_intent`: **TO BE FILLED BY HUMAN ANNOTATOR**
- `human_should_escalate`: **TO BE FILLED BY HUMAN ANNOTATOR**
- `human_escalation_reason`: **TO BE FILLED BY HUMAN ANNOTATOR**
- `annotator_notes`: Free-text notes from annotation

## Data Leakage Prevention

This golden set must NOT be used for:
- Model training
- Classifier fitting
- Retrieval indexing
- Prompt examples

Temporal splitting ensures:
- Golden set conversations are the most recent (held out last)
- Zero overlapping messages between golden and train/dev splits (verified)
- No information leakage from future into training data

## Evaluation Metrics

When evaluating against this golden set, report:
- Intent: Accuracy, Macro-F1, Weighted-F1, per-class precision/recall/F1
- Escalation: Precision, Recall, F1
- LLM-as-judge (if available): Relevance, Groundedness, Correctness, Completeness, Tone, Hallucination risk

**Important:** All metrics computed on auto-generated labels are preliminary until human annotation is complete.
