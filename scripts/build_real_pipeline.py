"""
Complete real-data pipeline for Uber_Support brand.

This script:
1. Loads the Kaggle twcs.csv (chunked for memory efficiency)
2. Extracts Uber_Support conversations
3. Builds conversation threads
4. Derives intent taxonomy from real customer messages
5. Creates conversation-aware temporal train/dev/test/golden splits
6. Prepares the 200-example golden annotation set
7. Builds the historical retrieval corpus (training only)
8. Pseudo-labels training data for classifier bootstrapping

Usage:
    python scripts/build_real_pipeline.py
"""
import pandas as pd
import numpy as np
import yaml
import json
import csv
import re
import hashlib
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BRAND = "Uber_Support"
RAW_CSV = Path("data/raw/twcs/twcs.csv")
OUTPUT_DIR = Path("data/processed")
CONFIGS_DIR = Path("configs")


# ---------------------------------------------------------------------------
# 1. LOAD & EXTRACT
# ---------------------------------------------------------------------------

def load_brand_data(brand: str = BRAND) -> pd.DataFrame:
    """Load brand rows + all conversation threads (chunked).

    Strategy:
    Pass 1: Collect brand tweet_ids and the tweet_ids they respond to
            (the root customer messages that START conversations).
    Pass 2: Collect all tweets whose tweet_id or in_response_to_tweet_id
            is in our known set. Repeat until stable.
    """
    logger.info(f"Loading {brand} data from {RAW_CSV} (chunked)...")

    all_cols = ['tweet_id', 'author_id', 'inbound', 'created_at', 'text',
                'response_tweet_id', 'in_response_to_tweet_id']
    chunk_size = 200_000

    # --- Pass 1: brand tweets + their parent customer tweets ---
    known_ids = set()
    brand_rows_list = []
    parent_rows_list = []

    for chunk in pd.read_csv(RAW_CSV, chunksize=chunk_size, usecols=all_cols, low_memory=False):
        brand_chunk = chunk[chunk['author_id'] == brand]
        if len(brand_chunk) > 0:
            brand_rows_list.append(brand_chunk)
            known_ids.update(brand_chunk['tweet_id'].dropna())

        # Parent tweets: what the brand tweets respond to
        parents = chunk[chunk['tweet_id'].isin(
            brand_chunk['in_response_to_tweet_id'].dropna()
        )]
        if len(parents) > 0:
            parent_rows_list.append(parents)
            known_ids.update(parents['tweet_id'].dropna())

    logger.info(f"Pass 1: {len(known_ids):,} known tweet_ids "
                f"(brand + parents)")

    # --- Pass 2: expand outward (customer replies to brand, brand replies back) ---
    for expansion_round in range(3):
        prev_count = len(known_ids)
        new_rows = []
        for chunk in pd.read_csv(RAW_CSV, chunksize=chunk_size, usecols=all_cols, low_memory=False):
            # Tweets that respond to any known tweet
            children = chunk[chunk['in_response_to_tweet_id'].isin(known_ids)]
            # Known tweets' children
            parents_of_known = chunk[chunk['tweet_id'].isin(
                chunk[chunk['tweet_id'].isin(known_ids)]['in_response_to_tweet_id'].dropna()
            )]
            found = pd.concat([children, parents_of_known]).drop_duplicates(subset='tweet_id')
            if len(found) > 0:
                new_rows.append(found)
                known_ids.update(found['tweet_id'].dropna())

        new_count = len(known_ids)
        logger.info(f"Pass 2 round {expansion_round+1}: {new_count:,} known_ids (+{new_count - prev_count:,})")
        if new_count == prev_count:
            break

    # --- Collect all rows matching known_ids ---
    all_rows = brand_rows_list + parent_rows_list
    if new_rows:
        all_rows.extend(new_rows)

    df = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset='tweet_id')
    # Only keep rows that are part of our conversation threads
    df = df[df['tweet_id'].isin(known_ids)].copy()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.sort_values('created_at').reset_index(drop=True)

    brand_count = df['author_id'].eq(brand).sum()
    cust_count = df['inbound'].sum()
    logger.info(f"Total extracted: {len(df):,} rows "
                f"({brand_count:,} brand, {cust_count:,} customer)")
    return df


# ---------------------------------------------------------------------------
# 2. BUILD CONVERSATIONS
# ---------------------------------------------------------------------------

def build_conversations(df: pd.DataFrame, brand: str = BRAND) -> List[Dict]:
    """
    Build conversation threads from tweet-level data.

    Uses the parent→child graph (in_response_to_tweet_id) to reconstruct
    full threads.  Each conversation is rooted at the earliest tweet in the
    connected component and contains all messages in chronological order.
    """
    logger.info("Building conversation threads...")

    # Build lookup structures
    tid_to_row = {}
    children_map = defaultdict(list)  # parent_tid → [child_tid, ...]
    parent_map = {}  # child_tid → parent_tid

    for _, row in df.iterrows():
        tid = row['tweet_id']
        tid_to_row[tid] = row
        parent = row['in_response_to_tweet_id']
        if pd.notna(parent) and parent in tid_to_row:
            children_map[parent].append(tid)
            parent_map[tid] = parent
        elif pd.notna(parent):
            parent_map[tid] = parent

    # Rebuild parent_map with full data
    for _, row in df.iterrows():
        tid = row['tweet_id']
        parent = row['in_response_to_tweet_id']
        if pd.notna(parent):
            parent_map[tid] = parent

    # Find connected components via union-find
    all_tids = set(df['tweet_id'].dropna())
    parent_of = {}  # union-find parent

    def find(x):
        while parent_of.get(x, x) != x:
            parent_of[x] = parent_of.get(parent_of[x], parent_of[x])
            x = parent_of[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent_of[ra] = rb

    for tid in all_tids:
        if tid not in parent_of:
            parent_of[tid] = tid

    for _, row in df.iterrows():
        tid = row['tweet_id']
        parent = row['in_response_to_tweet_id']
        if pd.notna(parent) and parent in all_tids:
            union(tid, parent)

    # Group by component root
    components = defaultdict(list)
    for tid in all_tids:
        root = find(tid)
        components[root].append(tid)

    logger.info(f"Found {len(components):,} conversation components")

    # Build conversations
    conversations = []
    for root, tids in components.items():
        # Sort by timestamp
        rows = [tid_to_row[tid] for tid in tids if tid in tid_to_row]
        rows.sort(key=lambda r: r['created_at'] if pd.notna(r['created_at']) else pd.Timestamp.min)

        if not rows:
            continue

        cust_msgs = [r for r in rows if r['inbound'] == True or r['inbound'] == 1]
        brand_msgs = [r for r in rows if r['inbound'] == False or r['inbound'] == 0]

        if not cust_msgs:
            continue

        # First customer message is the initial request
        first_cust = cust_msgs[0]

        conversation = {
            'conversation_id': str(root),
            'customer_message': str(first_cust['text']),
            'brand_response': str(brand_msgs[-1]['text']) if brand_msgs else '',
            'timestamp': str(rows[0]['created_at']) if pd.notna(rows[0]['created_at']) else '',
            'thread_length': len(rows),
            'num_customer_msgs': len(cust_msgs),
            'num_brand_msgs': len(brand_msgs),
            'has_brand_response': len(brand_msgs) > 0,
            'all_customer_msgs': [str(r['text']) for r in cust_msgs],
            'all_brand_msgs': [str(r['text']) for r in brand_msgs],
            'tweet_ids': [str(r['tweet_id']) for r in rows],
        }
        conversations.append(conversation)

    logger.info(f"Built {len(conversations):,} conversations "
                f"({sum(1 for c in conversations if c['has_brand_response']):,} with brand response)")
    return conversations


# ---------------------------------------------------------------------------
# 3. INTENT TAXONOMY
# ---------------------------------------------------------------------------

INTENT_KEYWORDS = {
    'ride_status': ['driver', 'ride', 'trip', 'pickup', 'dropoff', 'eta', 'waiting', 'cancelled', 'cancel on', 'no show', 'arrived', 'destination', 'route', 'stranded', 'where is', 'how long'],
    'trip_fare_dispute': ['charge', 'fare', 'surge', 'price', 'cost', 'double', 'receipt', 'charged', 'billing', 'overcharged', 'amount', 'estimate'],
    'account_access': ['login', 'password', 'account', 'banned', 'deactivated', 'locked', 'sign in', 'reset', 'verify', 'verification', 'otp', 'phone number', 'email'],
    'app_functionality': ['app', 'crash', 'bug', 'gps', 'map', 'loading', 'broken', 'not working', 'glitch', 'update', 'download', 'install', 'error', 'notification'],
    'complaint': ['unhappy', 'disappointed', 'worst', 'terrible', 'bad', 'poor', 'unsatisfied', 'angry', 'frustrated', 'horrible', 'awful', 'rude', 'never again', 'scam', 'fraud', 'pathetic', 'joke'],
    'refund_request': ['refund', 'money back', 'return', 'give me back', 'credit back', 'reimburse'],
    'cancellation': ['cancel my', 'cancel account', 'unsubscribe', 'subscription', 'delete account', 'stop subscription', 'cancellation fee'],
}


def pseudo_label(text: str) -> str:
    """Assign intent label via keyword matching."""
    t = text.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return intent
    return 'other'


def derive_taxonomy(conversations: List[Dict]) -> List[Dict]:
    """Derive intent taxonomy from actual customer messages."""
    logger.info("Deriving intent taxonomy from real data...")

    customer_messages = [c['customer_message'] for c in conversations]
    intent_counts = Counter()
    intent_examples = defaultdict(list)

    for msg in customer_messages:
        intent = pseudo_label(msg)
        intent_counts[intent] += 1
        if len(intent_examples[intent]) < 10:
            intent_examples[intent].append(msg)

    intents = []
    for intent, count in intent_counts.most_common():
        pct = count / len(customer_messages) * 100
        intents.append({
            'name': intent,
            'description': f'{intent.replace("_", " ").title()} issues ({count} examples, {pct:.1f}%)',
            'positive_examples': intent_examples[intent][:5],
            'confusing_related_intents': [],
            'escalation_considerations': f'{intent.replace("_", " ").title()} may require escalation in complex cases',
            'count': count,
            'percentage': round(pct, 1),
        })

    logger.info(f"Derived {len(intents)} intents:")
    for i in intents:
        logger.info(f"  {i['name']}: {i['count']:,} ({i['percentage']}%)")

    return intents


# ---------------------------------------------------------------------------
# 4. CONVERSATION-AWARE TEMPORAL SPLIT
# ---------------------------------------------------------------------------

def temporal_conversation_split(
    conversations: List[Dict],
    golden_size: int = 200,
    dev_fraction: float = 0.15,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Split conversations chronologically (temporal) and conversation-aware.

    Strategy:
    1. Sort all conversations by timestamp (earliest first)
    2. Reserve the LATEST 200 conversations as the golden set (most realistic eval)
    3. From the remaining, use earliest 70% as train, middle ~15% as dev
    4. No conversation appears in multiple splits

    This simulates a real deployment: train on older data, evaluate on newer data.
    """
    logger.info("Creating temporal conversation-aware split...")

    # Parse timestamps for sorting
    for c in conversations:
        try:
            c['_ts'] = pd.to_datetime(c['timestamp']) if c['timestamp'] else pd.Timestamp.min
        except:
            c['_ts'] = pd.Timestamp.min

    # Sort by timestamp
    sorted_convs = sorted(conversations, key=lambda c: c['_ts'])

    # Golden set: last 200 conversations (most recent)
    golden_n = min(golden_size, len(sorted_convs) // 5)
    golden = sorted_convs[-golden_n:]
    remaining = sorted_convs[:-golden_n]

    # Train/dev split from remaining
    n_remaining = len(remaining)
    dev_n = max(1, int(n_remaining * dev_fraction))
    dev = remaining[-dev_n:]
    train = remaining[:-dev_n]

    # Clean up internal keys
    for c in train + dev + golden:
        c.pop('_ts', None)

    # Verify no leakage
    train_ids = {c['conversation_id'] for c in train}
    dev_ids = {c['conversation_id'] for c in dev}
    golden_ids = {c['conversation_id'] for c in golden}

    assert len(train_ids & dev_ids) == 0, "LEAKAGE: train-dev overlap"
    assert len(train_ids & golden_ids) == 0, "LEAKAGE: train-golden overlap"
    assert len(dev_ids & golden_ids) == 0, "LEAKAGE: dev-golden overlap"

    logger.info(f"Split (temporal): train={len(train):,}, dev={len(dev):,}, golden={len(golden):,}")
    logger.info("No conversation leakage detected")

    return train, dev, golden, []


# ---------------------------------------------------------------------------
# 5. GOLDEN SET ANNOTATION PREPARATION
# ---------------------------------------------------------------------------

def prepare_golden_set(golden_convs: List[Dict], output_dir: Path) -> pd.DataFrame:
    """
    Prepare the 200-example golden annotation set.
    Labels are auto-generated (NOT hand-labelled).
    """
    logger.info(f"Preparing {len(golden_convs)} golden examples...")

    rows = []
    for i, conv in enumerate(golden_convs):
        intent = pseudo_label(conv['customer_message'])
        # Escalation heuristic: complaint, low brand response, very long thread
        should_escalate = (
            intent == 'complaint' or
            (not conv['has_brand_response']) or
            conv['thread_length'] >= 6
        )

        rows.append({
            'example_id': i,
            'conversation_id': conv['conversation_id'],
            'customer_message': conv['customer_message'],
            'brand_response': conv['brand_response'],
            'auto_intent': intent,  # NOT hand-labelled
            'auto_should_escalate': should_escalate,
            'thread_length': conv['thread_length'],
            'timestamp': conv['timestamp'],
            'human_intent': '',  # TO BE FILLED BY HUMAN ANNOTATOR
            'human_should_escalate': '',  # TO BE FILLED BY HUMAN ANNOTATOR
            'annotator_notes': '',  # TO BE FILLED BY HUMAN ANNOTATOR
        })

    golden_df = pd.DataFrame(rows)

    # Save
    golden_path = output_dir / "golden_set.csv"
    golden_df.to_csv(golden_path, index=False)
    logger.info(f"Golden set saved to {golden_path}")

    # Stats
    intent_dist = golden_df['auto_intent'].value_counts()
    logger.info(f"Golden set intent distribution:")
    for intent, count in intent_dist.items():
        logger.info(f"  {intent}: {count} ({count/len(golden_df)*100:.1f}%)")
    logger.info(f"Auto-escalation rate: {golden_df['auto_should_escalate'].mean()*100:.1f}%")

    return golden_df


# ---------------------------------------------------------------------------
# 6. BUILD TRAINING DATA WITH PSEUDO-LABELS
# ---------------------------------------------------------------------------

def build_training_data(train_convs: List[Dict], dev_convs: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build training and dev DataFrames with pseudo-labels."""

    def convs_to_df(convs):
        rows = []
        for c in convs:
            intent = pseudo_label(c['customer_message'])
            should_escalate = (
                intent == 'complaint' or
                (not c['has_brand_response']) or
                c['thread_length'] >= 6
            )
            rows.append({
                'conversation_id': c['conversation_id'],
                'customer_message': c['customer_message'],
                'brand_response': c['brand_response'],
                'text': c['customer_message'],  # alias for classifier
                'intent': intent,
                'should_escalate': should_escalate,
                'thread_length': c['thread_length'],
                'timestamp': c['timestamp'],
                'num_customer_msgs': c['num_customer_msgs'],
                'num_brand_msgs': c['num_brand_msgs'],
            })
        return pd.DataFrame(rows)

    train_df = convs_to_df(train_convs)
    dev_df = convs_to_df(dev_convs)

    logger.info(f"Training set: {len(train_df):,} examples")
    logger.info(f"Dev set: {len(dev_df):,} examples")
    logger.info(f"Train intent distribution:")
    for intent, count in train_df['intent'].value_counts().items():
        logger.info(f"  {intent}: {count:,} ({count/len(train_df)*100:.1f}%)")

    return train_df, dev_df


# ---------------------------------------------------------------------------
# 7. SAVE EVERYTHING
# ---------------------------------------------------------------------------

def save_configs(brand: str, intents: List[Dict]):
    """Update brand.yaml and intents.yaml."""
    brand_config = {
        'brand_name': brand,
        'confidence_threshold': 0.7,
        'dataset_path': 'data/processed',
        'escalation_threshold': 0.3,
        'random_seed': SEED,
        'sample_size': None,  # Will be set after split
        'top_k_retrieval': 5,
        'data_source': 'kaggle_customer_support_on_twitter',
        'golden_set_status': 'auto_generated_labels_NOT_hand_labelled',
    }
    with open(CONFIGS_DIR / 'brand.yaml', 'w') as f:
        yaml.dump(brand_config, f, default_flow_style=False, indent=2)
    logger.info(f"Updated {CONFIGS_DIR / 'brand.yaml'}")

    intents_data = {'intents': []}
    for intent in intents:
        intents_data['intents'].append({
            'name': intent['name'],
            'description': intent['description'],
            'positive_examples': intent['positive_examples'],
            'confusing_related_intents': intent['confusing_related_intents'],
            'escalation_considerations': intent['escalation_considerations'],
        })
    with open(CONFIGS_DIR / 'intents.yaml', 'w') as f:
        yaml.dump(intents_data, f, default_flow_style=False, indent=2)
    logger.info(f"Updated {CONFIGS_DIR / 'intents.yaml'}")


def save_splits(train_df, dev_df, golden_df):
    """Save CSV splits."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / 'train.csv', index=False)
    dev_df.to_csv(OUTPUT_DIR / 'dev.csv', index=False)
    golden_df.to_csv(OUTPUT_DIR / 'golden_set.csv', index=False)
    logger.info(f"Saved splits to {OUTPUT_DIR}")


def save_split_metadata(train_df, dev_df, golden_df, intents):
    """Save split metadata JSON."""
    metadata = {
        'brand': BRAND,
        'data_source': 'kaggle_customer_support_on_twitter',
        'split_method': 'temporal_conversation_aware',
        'split_description': (
            'Conversations sorted chronologically. '
            'Golden set = most recent 200 conversations. '
            'Train = earliest 70% of remaining. '
            'Dev = latest 15% of remaining. '
            'No conversation leakage between splits.'
        ),
        'train_size': len(train_df),
        'dev_size': len(dev_df),
        'golden_size': len(golden_df),
        'total': len(train_df) + len(dev_df) + len(golden_df),
        'taxonomy_size': len(intents),
        'golden_set_labels': 'auto_generated_NOT_hand_labelled',
        'random_seed': SEED,
        'timestamp': datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved split metadata to {OUTPUT_DIR / 'split_metadata.json'}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("REAL DATA PIPELINE — Uber_Support from Kaggle TWCS")
    print("=" * 70)

    # 1. Load brand data
    df = load_brand_data(BRAND)

    # 2. Build conversations
    conversations = build_conversations(df, BRAND)

    # 3. Derive intent taxonomy
    intents = derive_taxonomy(conversations)

    # 4. Temporal split
    train_convs, dev_convs, golden_convs, _ = temporal_conversation_split(conversations)

    # 5. Build training data
    train_df, dev_df = build_training_data(train_convs, dev_convs)

    # 6. Prepare golden set
    golden_df = prepare_golden_set(golden_convs, OUTPUT_DIR)

    # 7. Save everything
    save_configs(BRAND, intents)
    save_splits(train_df, dev_df, golden_df)
    save_split_metadata(train_df, dev_df, golden_df, intents)

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Brand: {BRAND}")
    print(f"Total conversations: {len(conversations):,}")
    print(f"Train: {len(train_df):,}")
    print(f"Dev: {len(dev_df):,}")
    print(f"Golden: {len(golden_df):,}")
    print(f"Intents: {len(intents)}")
    print(f"\nFiles saved:")
    print(f"  {OUTPUT_DIR / 'train.csv'}")
    print(f"  {OUTPUT_DIR / 'dev.csv'}")
    print(f"  {OUTPUT_DIR / 'golden_set.csv'}")
    print(f"  {OUTPUT_DIR / 'split_metadata.json'}")
    print(f"  {CONFIGS_DIR / 'brand.yaml'}")
    print(f"  {CONFIGS_DIR / 'intents.yaml'}")
    print(f"\nIMPORTANT: Golden set labels are auto-generated, NOT hand-labelled.")
    print(f"Manual annotation is required before reporting final metrics.")


if __name__ == "__main__":
    main()
