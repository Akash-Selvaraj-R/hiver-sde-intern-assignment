"""
Evaluate golden set after human annotation.

Validates the annotated CSV for completeness, then reports metrics.

Usage:
    python evaluation/eval_human_labels.py              # validate + report stats
    python evaluation/eval_human_labels.py --with-eval   # also run model evaluation
"""
import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

ANNOTATION_PATH = Path(__file__).parent / "human_annotation" / "annotation.csv"
EXPECTED_ROWS = 200
VALID_INTENTS = {
    "ride_status", "other", "account_access", "app_functionality",
    "trip_fare_dispute", "complaint", "refund_request", "cancellation",
}


def load_and_validate():
    """Load annotated CSV and run all validation checks. Exit on any error."""
    errors = []

    # --- File existence ---
    if not ANNOTATION_PATH.exists():
        print(f"ERROR: Annotation file not found at {ANNOTATION_PATH}")
        print("Complete your annotation and save it there.")
        sys.exit(1)

    df = pd.read_csv(ANNOTATION_PATH)

    # --- Required columns ---
    required_cols = [
        "example_id", "customer_message", "auto_intent", "human_intent",
        "auto_should_escalate", "human_should_escalate",
        "human_escalation_reason", "annotator_notes",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)

    # --- Row count ---
    if len(df) != EXPECTED_ROWS:
        errors.append(
            f"Expected exactly {EXPECTED_ROWS} examples, found {len(df)}"
        )

    # --- Unique example_id ---
    dup_ids = df["example_id"].duplicated().sum()
    if dup_ids > 0:
        dups = df[df["example_id"].duplicated(keep=False)]["example_id"].unique()
        errors.append(f"Duplicate example_id values ({dup_ids} duplicates): {sorted(dups)}")

    # --- No missing customer_message ---
    null_msg = df["customer_message"].isna().sum()
    if null_msg > 0:
        errors.append(f"{null_msg} examples have blank customer_message")

    # --- human_intent validation ---
    blank_intent = df["human_intent"].isna() | (df["human_intent"].astype(str).str.strip() == "")
    blank_intent_count = blank_intent.sum()
    if blank_intent_count > 0:
        examples = df[blank_intent]["example_id"].tolist()
        errors.append(
            f"{blank_intent_count} examples have blank human_intent: {examples}"
        )

    invalid_intent = ~df["human_intent"].isin(VALID_INTENTS) & ~blank_intent
    if invalid_intent.any():
        bad = df[invalid_intent]["human_intent"].unique().tolist()
        errors.append(
            f"Invalid human_intent values: {bad}. "
            f"Valid: {sorted(VALID_INTENTS)}"
        )

    # --- human_should_escalate validation ---
    blank_esc = df["human_should_escalate"].isna() | (
        df["human_should_escalate"].astype(str).str.strip() == ""
    )
    blank_esc_count = blank_esc.sum()
    if blank_esc_count > 0:
        examples = df[blank_esc]["example_id"].tolist()
        errors.append(
            f"{blank_esc_count} examples have blank human_should_escalate: {examples}"
        )

    # Normalize escalation to boolean for downstream use
    esc_series = df["human_should_escalate"].astype(str).str.strip().str.lower()
    valid_esc = esc_series.isin({"true", "false"})
    invalid_esc = ~valid_esc & ~blank_esc
    if invalid_esc.any():
        bad_vals = df[invalid_esc]["human_should_escalate"].unique().tolist()
        errors.append(
            f"Invalid human_should_escalate values: {bad_vals}. "
            f"Must be exactly 'True' or 'False'."
        )

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print(f"\nAnnotation is INCOMPLETE. Fix the above issues and re-run.")
        sys.exit(1)

    # All validations passed — normalize
    df["human_should_escalate"] = esc_series.map({"true": True, "false": False})

    # --- Escalation reason check (warning, not fatal) ---
    esc_without_reason = df[
        (df["human_should_escalate"] == True)
        & (df["human_escalation_reason"].isna() | (df["human_escalation_reason"].astype(str).str.strip() == ""))
    ]
    if len(esc_without_reason) > 0:
        ids = esc_without_reason["example_id"].tolist()
        print(f"WARNING: {len(ids)} escalated examples have no human_escalation_reason: {ids}")
        print("         Consider adding a reason before final submission.\n")

    return df


def report_statistics(df):
    """Print annotation statistics and auto-vs-human agreement."""
    n = len(df)
    print("=" * 60)
    print("ANNOTATION STATISTICS")
    print("=" * 60)

    # Intent distribution
    dist = df["human_intent"].value_counts()
    print(f"\nIntent distribution ({n} total):")
    for intent, count in dist.items():
        print(f"  {intent}: {count} ({count / n * 100:.1f}%)")

    # Escalation
    esc_count = df["human_should_escalate"].sum()
    print(f"\nEscalation: {esc_count}/{n} ({esc_count / n * 100:.1f}%)")

    # Auto vs Human agreement
    intent_agree = (df["auto_intent"] == df["human_intent"]).sum()
    esc_agree = (df["auto_should_escalate"] == df["human_should_escalate"]).sum()
    print(f"\nAuto vs Human agreement:")
    print(f"  Intent:     {intent_agree}/{n} ({intent_agree / n * 100:.1f}%)")
    print(f"  Escalation: {esc_agree}/{n} ({esc_agree / n * 100:.1f}%)")

    # Intent disagreements
    disagreements = df[df["auto_intent"] != df["human_intent"]]
    if len(disagreements) > 0:
        print(f"\nIntent disagreements ({len(disagreements)} total):")
        for _, row in disagreements.head(10).iterrows():
            print(
                f"  Ex {row['example_id']}: auto={row['auto_intent']}, "
                f"human={row['human_intent']}"
            )
            msg = row['customer_message'][:80].encode('ascii', 'replace').decode('ascii')
            print(f"    \"{msg}\"")
        if len(disagreements) > 10:
            print(f"  ... and {len(disagreements) - 10} more")

    # Notes
    has_notes = df["annotator_notes"].notna() & (df["annotator_notes"].astype(str).str.strip() != "")
    print(f"\nExamples with annotator notes: {has_notes.sum()}/{n}")
    print()


def run_full_evaluation(df):
    """Run TF-IDF baseline evaluation using human labels as ground truth."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        classification_report,
        confusion_matrix,
    )

    print("=" * 60)
    print("FULL EVALUATION (human labels as ground truth)")
    print("=" * 60)

    # Load training data
    train_df = pd.read_csv("data/processed/train.csv")
    train_texts = train_df["customer_message"].tolist()
    if "intent" in train_df.columns:
        train_labels = train_df["intent"].tolist()
    else:
        from classifier import _pseudo_label_batch
        train_labels = _pseudo_label_batch(train_texts)

    golden_texts = df["customer_message"].tolist()
    true_intents = df["human_intent"].tolist()
    true_escalation = df["human_should_escalate"].tolist()

    print(f"Train: {len(train_texts)} examples")
    print(f"Golden (human-labelled): {len(golden_texts)} examples")

    # TF-IDF baseline
    from baselines import TfIdfBaseline
    baseline = TfIdfBaseline()
    baseline.fit(train_texts, train_labels, train_texts)
    pred_intents = baseline.predict_intent(golden_texts)
    pred_escalation = baseline.predict_escalation(golden_texts)

    # --- Intent metrics ---
    intent_acc = accuracy_score(true_intents, pred_intents)
    intent_f1_macro = f1_score(true_intents, pred_intents, average="macro", zero_division=0)
    intent_f1_weighted = f1_score(true_intents, pred_intents, average="weighted", zero_division=0)

    print(f"\n--- Intent Classification ---")
    print(f"  Accuracy:       {intent_acc:.3f}")
    print(f"  Macro-F1:       {intent_f1_macro:.3f}")
    print(f"  Weighted-F1:    {intent_f1_weighted:.3f}")

    # Per-intent P/R/F1
    intents_present = sorted(set(true_intents) | set(pred_intents))
    prec, rec, f1, sup = precision_recall_fscore_support(
        true_intents, pred_intents, labels=intents_present, zero_division=0
    )
    print(f"\n  Per-intent breakdown:")
    print(f"  {'Intent':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>5}")
    print(f"  {'-' * 50}")
    for i, intent in enumerate(intents_present):
        print(f"  {intent:<25} {prec[i]:>6.3f} {rec[i]:>6.3f} {f1[i]:>6.3f} {int(sup[i]):>5}")

    # Confusion matrix
    cm = confusion_matrix(true_intents, pred_intents, labels=intents_present)
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = "  " + "".join(f"{i[:8]:>9}" for i in intents_present)
    print(header)
    for row_idx, intent in enumerate(intents_present):
        row_str = f"  {intent[:8]:<8}" + "".join(f"{cm[row_idx, col]:>9}" for col in range(len(intents_present)))
        print(row_str)

    # --- Escalation metrics ---
    esc_acc = accuracy_score(true_escalation, pred_escalation)
    esc_prec, esc_rec, esc_f1, _ = precision_recall_fscore_support(
        true_escalation, pred_escalation, average="binary", zero_division=0
    )

    print(f"\n--- Escalation ---")
    print(f"  Accuracy:  {esc_acc:.3f}")
    print(f"  Precision: {esc_prec:.3f}")
    print(f"  Recall:    {esc_rec:.3f}")
    print(f"  F1:        {esc_f1:.3f}")

    # Full classification report
    print(f"\n--- Full Classification Report ---")
    print(classification_report(true_intents, pred_intents, zero_division=0))

    # Save results
    import json
    from datetime import datetime

    results = {
        "label_source": "human_annotation",
        "timestamp": datetime.now().isoformat(),
        "golden_size": len(df),
        "intent": {
            "accuracy": intent_acc,
            "macro_f1": intent_f1_macro,
            "weighted_f1": intent_f1_weighted,
        },
        "escalation": {
            "accuracy": esc_acc,
            "precision": esc_prec,
            "recall": esc_rec,
            "f1": esc_f1,
        },
        "confusion_matrix": cm.tolist(),
        "intent_labels": intents_present,
    }

    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "human_label_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and evaluate human-annotated golden set"
    )
    parser.add_argument(
        "--with-eval",
        action="store_true",
        help="Run full model evaluation using human labels as ground truth",
    )
    args = parser.parse_args()

    df = load_and_validate()
    report_statistics(df)

    if args.with_eval:
        run_full_evaluation(df)
    else:
        print("To run full evaluation with human labels:")
        print("  python evaluation/eval_human_labels.py --with-eval")


if __name__ == "__main__":
    main()
