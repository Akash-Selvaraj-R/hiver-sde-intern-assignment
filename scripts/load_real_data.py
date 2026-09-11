"""
Load and profile the real Kaggle Customer Support on Twitter dataset.

This script:
1. Loads twcs.csv from data/raw/
2. Profiles the dataset (rows, columns, brands, distribution)
3. Analyzes brand support volume and selects the best brand
4. Creates conversation-aware train/validation/test splits
5. Derives the intent taxonomy from real data
6. Prepares the golden set annotation workflow

Usage:
    python scripts/load_real_data.py
    python scripts/load_real_data.py --brand @AmazonHelp
"""
import pandas as pd
import numpy as np
import yaml
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_raw_dataset() -> pd.DataFrame:
    """Load the raw Kaggle dataset from data/raw/."""
    raw_dir = Path("data/raw")

    # Try different possible filenames
    candidates = ["twcs.csv", "customer_support_on_twitter.csv"]
    for fname in candidates:
        path = raw_dir / fname
        if path.exists():
            logger.info(f"Loading dataset from {path}")
            df = pd.read_csv(path, low_memory=False)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df

    raise FileNotFoundError(
        f"Kaggle dataset not found in {raw_dir}/. "
        f"Expected one of: {candidates}\n"
        f"Run: kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw/ && "
        f"unzip data/raw/*.zip -d data/raw/"
    )


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a comprehensive dataset profile."""
    logger.info("Profiling dataset...")

    profile = {
        'total_rows': len(df),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_counts': df.isnull().sum().to_dict(),
        'unique_values': {col: int(df[col].nunique()) for col in df.columns},
    }

    # Inbound/outbound distribution
    if 'inbound' in df.columns:
        inbound_counts = df['inbound'].value_counts()
        profile['inbound_distribution'] = {
            'customer_messages': int(inbound_counts.get(True, inbound_counts.get(1, 0))),
            'brand_messages': int(inbound_counts.get(False, inbound_counts.get(0, 0)))
        }

    # Brand analysis
    if 'author_id' in df.columns:
        brand_messages = df[df['inbound'] == 0] if 'inbound' in df.columns else df
        brand_counts = brand_messages['author_id'].value_counts()
        profile['top_20_brands'] = [
            {'brand': brand, 'message_count': int(count)}
            for brand, count in brand_counts.head(20).items()
        ]
        profile['unique_brands'] = int(brand_counts.shape[0])

    # Timestamp coverage
    if 'created_at' in df.columns:
        try:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            valid_dates = df['created_at'].dropna()
            profile['date_range'] = {
                'start': str(valid_dates.min()),
                'end': str(valid_dates.max()),
                'span_days': int((valid_dates.max() - valid_dates.min()).days) if len(valid_dates) > 1 else 0
            }
        except Exception as e:
            logger.warning(f"Could not parse timestamps: {e}")

    # Response relationships
    if 'in_response_to_tweet_id' in df.columns:
        response_count = df['in_response_to_tweet_id'].notna().sum()
        profile['response_relationships'] = int(response_count)

    # Text statistics
    if 'text' in df.columns:
        text_lengths = df['text'].astype(str).str.len()
        profile['text_stats'] = {
            'mean_length': float(text_lengths.mean()),
            'median_length': float(text_lengths.median()),
            'min_length': int(text_lengths.min()),
            'max_length': int(text_lengths.max()),
        }

    logger.info(f"Dataset profile complete: {profile['total_rows']} rows, {profile.get('unique_brands', 0)} unique brands")
    return profile


def select_brand(df: pd.DataFrame, min_messages: int = 500) -> str:
    """Select the best brand based on data-driven criteria."""
    logger.info("Selecting brand...")

    if 'author_id' not in df.columns or 'inbound' not in df.columns:
        raise ValueError("Dataset must have 'author_id' and 'inbound' columns")

    brand_messages = df[df['inbound'] == False]
    brand_counts = brand_messages['author_id'].value_counts()

    # Filter by minimum message count
    eligible_brands = brand_counts[brand_counts >= min_messages]

    if len(eligible_brands) == 0:
        # Lower the threshold and take the top brand
        logger.warning(f"No brands with >= {min_messages} messages. Using top brand.")
        top_brand = brand_counts.index[0]
        logger.info(f"Selected brand: {top_brand} ({brand_counts.iloc[0]} brand messages)")
        return top_brand

    # Score brands by: (1) brand message count, (2) customer message count, (3) conversation diversity
    brand_scores = {}
    for brand in eligible_brands.index:
        brand_df = df[df['author_id'] == brand]

        # Count customer messages in response to this brand
        brand_tweet_ids = set(brand_df['tweet_id'].tolist()) if 'tweet_id' in brand_df.columns else set()
        brand_count = len(brand_df)

        # Count customer messages that are responses to this brand
        if 'in_response_to_tweet_id' in df.columns:
            customer_msgs = df[
                (df['inbound'] == True) &
                (df['in_response_to_tweet_id'].isin(brand_tweet_ids))
            ]
            customer_count = len(customer_msgs)
        else:
            customer_count = brand_count  # Approximate

        # Score: brand messages + customer messages * 0.5
        score = brand_count + customer_count * 0.5
        brand_scores[brand] = {
            'brand_messages': brand_count,
            'customer_messages': customer_count,
            'score': score
        }

    # Select brand with highest score
    best_brand = max(brand_scores, key=lambda x: brand_scores[x]['score'])
    best_info = brand_scores[best_brand]

    logger.info(f"Selected brand: {best_brand}")
    logger.info(f"  Brand messages: {best_info['brand_messages']}")
    logger.info(f"  Customer messages: {best_info['customer_messages']}")
    logger.info(f"  Score: {best_info['score']:.0f}")

    return best_brand


def build_conversations(df: pd.DataFrame, brand: str) -> List[Dict]:
    """
    Build conversation-level cases from tweet-level data.

    Uses response_tweet_id / in_response_to_tweet_id to reconstruct threads.
    Each conversation is a thread of messages between customer and brand.
    """
    logger.info(f"Building conversations for brand {brand}...")

    brand_df = df[df['author_id'] == brand].copy()

    if 'tweet_id' not in df.columns:
        logger.warning("No tweet_id column - treating each row as independent")
        return [{'customer_message': row['text'], 'brand_response': '', 'conversation_id': f"conv_{i}"}
                for i, (_, row) in enumerate(brand_df.iterrows())]

    # Build response graph
    tweet_id_to_row = {}
    for _, row in df.iterrows():
        tweet_id_to_row[row['tweet_id']] = row

    conversations = []
    visited = set()

    for _, brand_row in brand_df.iterrows():
        if brand_row['tweet_id'] in visited:
            continue

        # Trace the conversation thread backward
        thread = []
        current_id = brand_row['tweet_id']

        while current_id and current_id in tweet_id_to_row and current_id not in visited:
            visited.add(current_id)
            row = tweet_id_to_row[current_id]
            thread.append(row)
            # Move to the tweet this responds to
            if 'in_response_to_tweet_id' in row.index:
                current_id = row['in_response_to_tweet_id']
            else:
                break

        # Reverse to get chronological order
        thread.reverse()

        # Extract customer message and brand response
        customer_msgs = [r for r in thread if r.get('inbound') == True or r.get('inbound') == 1]
        brand_msgs = [r for r in thread if r.get('inbound') == False or r.get('inbound') == 0]

        if customer_msgs:
            conversation = {
                'conversation_id': str(brand_row.get('tweet_id', f'conv_{len(conversations)}')),
                'customer_message': str(customer_msgs[0]['text']) if customer_msgs else '',
                'brand_response': str(brand_msgs[-1]['text']) if brand_msgs else '',
                'timestamp': str(brand_row.get('created_at', '')),
                'thread_length': len(thread),
                'has_brand_response': len(brand_msgs) > 0,
            }
            conversations.append(conversation)

    logger.info(f"Built {len(conversations)} conversations ({sum(1 for c in conversations if c['has_brand_response'])} with brand response)")
    return conversations


def conversation_aware_split(
    conversations: List[Dict],
    golden_size: int = 200,
    validation_size: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split conversations into train/validation/golden test sets.

    CRITICAL: Entire conversations go into only one split.
    No conversation appears in multiple splits.
    """
    logger.info("Splitting conversations (conversation-aware)...")

    random.seed(seed)
    np.random.seed(seed)

    n = len(conversations)
    golden_n = min(golden_size, n // 3)
    remaining = [c for c in conversations]

    # Stratified golden set sampling
    # Try to maintain intent distribution if available
    golden = random.sample(remaining, golden_n)
    golden_ids = {c['conversation_id'] for c in golden}

    remaining = [c for c in remaining if c['conversation_id'] not in golden_ids]

    # Split remaining into train and validation
    val_n = max(1, int(len(remaining) * validation_size))
    validation = random.sample(remaining, val_n)
    validation_ids = {c['conversation_id'] for c in validation}

    train = [c for c in remaining if c['conversation_id'] not in validation_ids]

    logger.info(f"Split: train={len(train)}, validation={len(validation)}, golden={len(golden)}")

    # Verify no leakage
    train_ids = {c['conversation_id'] for c in train}
    overlap_tg = train_ids & golden_ids
    overlap_tv = train_ids & validation_ids
    overlap_vg = validation_ids & golden_ids

    if overlap_tg or overlap_tv or overlap_vg:
        logger.error(f"DATA LEAKAGE DETECTED: train-golden={len(overlap_tg)}, train-val={len(overlap_tv)}, val-golden={len(overlap_vg)}")
    else:
        logger.info("No conversation leakage detected - all conversations unique across splits")

    return train, validation, golden


def derive_intent_taxonomy(conversations: List[Dict]) -> List[Dict]:
    """Derive intent taxonomy from real customer messages."""
    logger.info("Deriving intent taxonomy from real data...")

    # Simple keyword-based taxonomy derivation
    # In a real scenario, you'd use clustering or manual analysis
    customer_messages = [c['customer_message'] for c in conversations]

    intent_keywords = {
        'order_status': ['order', 'delivery', 'shipping', 'tracking', 'package', 'arrive', 'ship', 'receive'],
        'refund_request': ['refund', 'money back', 'return', 'chargeback', 'back'],
        'payment_issue': ['charge', 'payment', 'bill', 'invoice', 'cost', 'price', 'paid', 'double', 'credit'],
        'account_login': ['login', 'password', 'account', 'access', 'locked', 'sign in', 'reset'],
        'technical_issue': ['website', 'app', 'error', 'crash', 'loading', 'broken', 'not working', 'page', 'bug'],
        'cancellation': ['cancel', 'stop', 'unsubscribe', 'membership', 'subscription', 'recurring'],
        'billing_problem': ['billing', 'statement', 'receipt', 'invoice', 'tax'],
        'information_request': ['what', 'how', 'when', 'where', 'do you', 'can you', 'info', 'tell me', 'policy'],
        'complaint': ['unhappy', 'disappointed', 'worst', 'terrible', 'bad', 'poor', 'unsatisfied', 'angry', 'frustrated', 'horrible', 'awful'],
    }

    # Count messages per intent
    intent_counts = Counter()
    for msg in customer_messages:
        msg_lower = msg.lower()
        for intent, keywords in intent_keywords.items():
            if any(kw in msg_lower for kw in keywords):
                intent_counts[intent] += 1
                break

    # Build taxonomy
    intents = []
    for intent, count in intent_counts.most_common():
        pct = count / len(customer_messages) * 100
        intents.append({
            'name': intent,
            'description': f'{intent.replace("_", " ").title()} issues ({count} examples, {pct:.1f}%)',
            'positive_examples': [msg for msg in customer_messages if any(kw in msg.lower() for kw in intent_keywords[intent])][:5],
            'confusing_related_intents': [],
            'escalation_considerations': f'{intent.replace("_", " ").title()} may require escalation in complex cases',
        })

    # Add 'other' intent
    other_count = len(customer_messages) - sum(intent_counts.values())
    if other_count > 0:
        intents.append({
            'name': 'other',
            'description': f'Messages that do not clearly fit other categories ({other_count} examples)',
            'positive_examples': [msg for msg in customer_messages if not any(any(kw in msg.lower() for kw in kws) for kws in intent_keywords.values())][:5],
            'confusing_related_intents': [],
            'escalation_considerations': 'Ambiguous messages should be escalated for human review',
        })

    logger.info(f"Derived {len(intents)} intents from {len(customer_messages)} customer messages")
    for intent in intents:
        logger.info(f"  {intent['name']}: {intent['description'][:60]}")

    return intents


def main():
    """Main function to load and prepare real data."""
    parser = argparse.ArgumentParser(description="Load and prepare real Kaggle dataset")
    parser.add_argument("--brand", type=str, help="Brand account to focus on (e.g., @AmazonHelp)")
    parser.add_argument("--golden-size", type=int, default=200, help="Size of golden evaluation set")
    parser.add_argument("--profile-only", action="store_true", help="Only profile the dataset, don't process")
    args = parser.parse_args()

    print("=" * 60)
    print("REAL DATA LOADER - Kaggle Customer Support on Twitter")
    print("=" * 60)

    # Step 1: Load raw dataset
    try:
        df = load_raw_dataset()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease download the dataset first. See data/raw/README.md for instructions.")
        return

    # Step 2: Profile dataset
    profile = profile_dataset(df)
    print("\n=== Dataset Profile ===")
    print(f"Total rows: {profile['total_rows']}")
    print(f"Columns: {', '.join(profile['columns'])}")
    if 'inbound_distribution' in profile:
        d = profile['inbound_distribution']
        print(f"Customer messages: {d['customer_messages']}")
        print(f"Brand messages: {d['brand_messages']}")
    if 'unique_brands' in profile:
        print(f"Unique brands: {profile['unique_brands']}")
    if 'date_range' in profile:
        print(f"Date range: {profile['date_range']['start']} to {profile['date_range']['end']} ({profile['date_range']['span_days']} days)")

    if args.profile_only:
        # Save profile
        profile_path = Path("experiments/real_data_profile.json")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        print(f"\nProfile saved to {profile_path}")
        return

    # Step 3: Select brand
    if args.brand:
        brand = args.brand
        print(f"\nUsing specified brand: {brand}")
    else:
        brand = select_brand(df)
        print(f"\nSelected brand: {brand}")

    # Step 4: Build conversations
    conversations = build_conversations(df, brand)

    # Step 5: Conversation-aware split
    train, validation, golden = conversation_aware_split(
        conversations, golden_size=args.golden_size
    )

    # Step 6: Derive intent taxonomy
    taxonomy = derive_intent_taxonomy(train + validation)

    # Step 7: Save everything
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    pd.DataFrame(train).to_csv(output_dir / "real_train.csv", index=False)
    pd.DataFrame(validation).to_csv(output_dir / "real_validation.csv", index=False)
    pd.DataFrame(golden).to_csv(output_dir / "real_golden.csv", index=False)

    # Save taxonomy
    with open("configs/intents_real.yaml", 'w') as f:
        yaml.dump({'intents': taxonomy}, f, default_flow_style=False, indent=2)

    # Save brand config
    brand_config = {
        'brand_name': brand,
        'confidence_threshold': 0.7,
        'dataset_path': 'data/processed',
        'escalation_threshold': 0.3,
        'random_seed': SEED,
        'sample_size': len(conversations),
        'top_k_retrieval': 5,
    }
    with open("configs/brand_real.yaml", 'w') as f:
        yaml.dump(brand_config, f, default_flow_style=False, indent=2)

    # Save metadata
    metadata = {
        'brand': brand,
        'total_conversations': len(conversations),
        'train_size': len(train),
        'validation_size': len(validation),
        'golden_size': len(golden),
        'taxonomy_size': len(taxonomy),
        'data_source': 'kaggle_customer_support_on_twitter',
        'labeling_method': 'pending_human_annotation',
        'caveat': 'Golden set labels are NOT human-annotated yet',
    }
    with open(output_dir / "real_split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Saved ===")
    print(f"Train: {len(train)} conversations -> {output_dir}/real_train.csv")
    print(f"Validation: {len(validation)} conversations -> {output_dir}/real_validation.csv")
    print(f"Golden: {len(golden)} conversations -> {output_dir}/real_golden.csv")
    print(f"Taxonomy: {len(taxonomy)} intents -> configs/intents_real.yaml")
    print(f"Brand config: configs/brand_real.yaml")

    print(f"\n=== Next Steps ===")
    print(f"1. Review the taxonomy in configs/intents_real.yaml")
    print(f"2. Prepare golden set annotation: python evaluation/prepare_golden_annotation.py")
    print(f"3. Manually label the golden set examples")
    print(f"4. Update configs/brand.yaml to point to real data")
    print(f"5. Run evaluation: python evaluation/run_eval.py")


if __name__ == "__main__":
    main()
