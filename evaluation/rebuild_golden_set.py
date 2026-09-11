"""
Rebuild the 200-example golden annotation set with better class balance.

Strategy:
1. Keep original pseudo-labels as base
2. Reclassify into cancellation/refund_request where strong keyword evidence exists
3. Stratified sample to achieve target distribution
4. Save as evaluation/human_annotation/annotation.csv

No human labels are generated. No model is modified.
"""
import pandas as pd
import numpy as np
import re
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SEED = 42
np.random.seed(SEED)

POOL_PATH = Path("data/processed/golden_set.csv")
TRAIN_PATH = Path("data/processed/train.csv")
DEV_PATH = Path("data/processed/dev.csv")
OUTPUT_PATH = Path("evaluation/human_annotation/annotation.csv")

# Target distribution (sum = 200)
TARGETS = {
    "ride_status": 55,
    "other": 35,
    "account_access": 22,
    "app_functionality": 22,
    "trip_fare_dispute": 22,
    "complaint": 18,
    "refund_request": 15,
    "cancellation": 11,
}


def reclassify_cancellation(text: str, original_intent: str) -> str:
    """Reclassify to cancellation if strong customer-initiated cancellation signal."""
    t = text.lower()

    # Must NOT be a driver cancellation complaint
    is_driver_cancel = (
        re.search(r"driver.{0,40}cancel", t) or
        re.search(r"cancel.{0,40}driver", t) or
        re.search(r"drivers?.{0,20}(cancel|cancell)", t)
    )
    if is_driver_cancel:
        return original_intent

    # Strong cancellation signals (customer wants to cancel)
    strong_signals = [
        r"\bi (want|need|would like) to cancel\b",
        r"\bhow (do|can) i cancel\b",
        r"\bcancel (my )?(ride|order|account|trip|booking|subscription)\b",
        r"\bdelete my (uber )?account\b",
        r"\bunsubscri(be|ption)\b",
        r"\bstop (my )?(subscription|membership|account)\b",
        r"\bcan'?t cancel\b",
        r"\bwouldn'?t let me cancel\b",
        r"\bnot (being )?able to cancel\b",
        r"\bcancellation fee\b",
        r"\bcharged.*cancel(lation)?\b",
    ]
    for s in strong_signals:
        if re.search(s, t):
            return "cancellation"

    return original_intent


def reclassify_refund(text: str, original_intent: str) -> str:
    """Reclassify to refund_request if explicit refund ask."""
    t = text.lower()

    # Must have an explicit refund/return request
    refund_signals = [
        r"\brefund\b",
        r"\bmoney back\b",
        r"\breturn (my )?money\b",
        r"\bcharge ?back\b",
        r"\bgive me back\b",
        r"\bpay(ment)? (back|refund)\b",
        r"\bwant.*refund\b",
        r"\bneed.*refund\b",
        r"\bget.*refund\b",
        r"\bcan i (get|have|receive).*refund\b",
        r"\bhow (do|can) i (get|request).*refund\b",
        r"\bcompensat(e|ion|ed)\b",
    ]
    for s in refund_signals:
        if re.search(s, t):
            return "refund_request"

    return original_intent


def main():
    print("=" * 60)
    print("REBUILD GOLDEN SET - Stratified Sampling")
    print("=" * 60)
    print(f"Random seed: {SEED}")
    print(f"Target total: {sum(TARGETS.values())}")
    print()

    # Load pool
    pool = pd.read_csv(POOL_PATH)
    print(f"Loaded {len(pool)} held-out conversations from {POOL_PATH}")

    # Check no overlap with train/dev
    train_df = pd.read_csv(TRAIN_PATH)
    dev_df = pd.read_csv(DEV_PATH)
    train_msgs = set(train_df['customer_message'].dropna())
    dev_msgs = set(dev_df['customer_message'].dropna())
    pool_msgs = set(pool['customer_message'].dropna())

    train_overlap = pool_msgs & train_msgs
    dev_overlap = pool_msgs & dev_msgs
    print(f"Leakage check: {len(train_overlap)} overlap with train, {len(dev_overlap)} overlap with dev")
    assert len(train_overlap) == 0, "LEAKAGE: golden-train overlap!"
    assert len(dev_overlap) == 0, "LEAKAGE: golden-dev overlap!"
    print("No leakage detected.")
    print()

    # Reclassify cancellation and refund_request
    original_dist = pool['auto_intent'].value_counts()
    print("Original auto_intent distribution:")
    for intent, count in original_dist.items():
        print(f"  {intent}: {count}")

    pool['auto_intent'] = pool.apply(
        lambda row: reclassify_cancellation(row['customer_message'], row['auto_intent']),
        axis=1
    )
    pool['auto_intent'] = pool.apply(
        lambda row: reclassify_refund(row['customer_message'], row['auto_intent']),
        axis=1
    )

    new_dist = pool['auto_intent'].value_counts()
    print()
    print("After reclassification:")
    for intent in sorted(new_dist.index):
        old = original_dist.get(intent, 0)
        new = new_dist[intent]
        delta = new - old
        marker = f" (+{delta})" if delta > 0 else f" ({delta})" if delta < 0 else ""
        print(f"  {intent}: {new}{marker}")
    print()

    # Stratified sampling
    sampled = []
    for intent, target in TARGETS.items():
        candidates = pool[pool['auto_intent'] == intent]
        n_available = len(candidates)

        if n_available == 0:
            print(f"  WARNING: {intent} has 0 available examples after reclassification.")
            continue

        n_sample = min(target, n_available)
        if n_sample < target:
            print(f"  NOTE: {intent} target {target}, only {n_available} available. Using {n_sample}.")

        # Sample for diversity
        if n_sample < n_available:
            chosen = candidates.sample(n=n_sample, random_state=SEED)
        else:
            chosen = candidates

        sampled.append(chosen)
        print(f"  {intent}: sampled {n_sample}")

    golden = pd.concat(sampled, ignore_index=True)

    # If we're short of 200, fill remaining from largest underfilled groups
    total = len(golden)
    if total < 200:
        shortfall = 200 - total
        print(f"\n  Short by {shortfall}. Filling from available candidates...")
        sampled_ids = set(golden['conversation_id'])
        remaining = pool[~pool['conversation_id'].isin(sampled_ids)]

        # Fill from ride_status (likely largest remaining group), then other, etc.
        fill_order = ['ride_status', 'other', 'account_access', 'app_functionality',
                      'trip_fare_dispute', 'complaint', 'refund_request', 'cancellation']
        for intent in fill_order:
            if shortfall <= 0:
                break
            candidates = remaining[remaining['auto_intent'] == intent]
            n_take = min(shortfall, len(candidates))
            if n_take > 0:
                chosen = candidates.sample(n=n_take, random_state=SEED)
                sampled.append(chosen)
                shortfall -= n_take
                print(f"    Filled {n_take} from {intent}")

    golden = pd.concat(sampled, ignore_index=True)

    # Trim to exactly 200 if over
    if len(golden) > 200:
        golden = golden.sample(n=200, random_state=SEED).reset_index(drop=True)

    # Randomize ordering
    golden = golden.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Re-assign sequential example_ids
    golden['example_id'] = range(1, len(golden) + 1)

    # Build annotation CSV
    annot = pd.DataFrame({
        'example_id': golden['example_id'],
        'customer_message': golden['customer_message'],
        'auto_intent': golden['auto_intent'],
        'human_intent': '',  # BLANK
        'auto_should_escalate': golden['auto_should_escalate'],
        'human_should_escalate': '',  # BLANK
        'human_escalation_reason': '',  # BLANK
        'annotator_notes': '',  # BLANK
    })

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    annot.to_csv(OUTPUT_PATH, index=False)

    # Report
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Saved {len(annot)} examples to {OUTPUT_PATH}")
    print()
    print("Final auto_intent distribution:")
    final_dist = annot['auto_intent'].value_counts()
    for intent in sorted(final_dist.index):
        count = final_dist[intent]
        target = TARGETS.get(intent, 0)
        print(f"  {intent}: {count} (target {target})")
    print()
    print(f"Total: {len(annot)}")
    print(f"Unique example_ids: {annot['example_id'].nunique()}")
    print(f"Blank human_intent: {(annot['human_intent'] == '').sum()}")
    print(f"Blank human_should_escalate: {(annot['human_should_escalate'] == '').sum()}")

    # Verify randomized
    intent_order = annot['auto_intent'].tolist()
    is_sorted = intent_order == sorted(intent_order)
    print(f"Randomized order: {'YES' if not is_sorted else 'NO'}")


if __name__ == "__main__":
    main()
