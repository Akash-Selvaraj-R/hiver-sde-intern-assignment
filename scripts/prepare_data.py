"""
Data preparation script for the customer support agent.
Handles loading, splitting, and preparing data for training and evaluation.

IMPORTANT: This project uses a SYNTHETIC dataset because the Kaggle
"Customer Support on Twitter" dataset requires API credentials that are
not available. The synthetic data is generated to mimic the structure
of the real dataset. All results should be interpreted with this caveat.
"""
import pandas as pd
import numpy as np
import yaml
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_config() -> dict:
    """Load brand configuration."""
    config_path = Path("configs/brand.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_full_dataset() -> pd.DataFrame:
    """Load the full synthetic dataset."""
    data_path = Path("data/sample/twitter_support_synthetic.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['text'] = df['text'].astype(str)
    df['text_length'] = df['text'].str.len()
    
    logger.info(f"Loaded {len(df)} rows from synthetic dataset")
    return df


def load_brand_data() -> pd.DataFrame:
    """Load brand-specific filtered data."""
    config = load_config()
    brand_name = config['brand_name']
    data_path = Path(config['dataset_path']) / f"{brand_name}_conversations.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Brand data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    logger.info(f"Loaded {len(df)} rows for brand {brand_name}")
    return df


def create_keyword_labeler():
    """Create a keyword-based pseudo-labeler for intent classification."""
    intent_keywords = {
        'order_status': ['order', 'delivery', 'shipping', 'tracking', 'package', 'arrive', 'ship'],
        'refund_request': ['refund', 'money back', 'return', 'chargeback'],
        'payment_issue': ['charge', 'payment', 'bill', 'invoice', 'cost', 'price', 'paid', 'double'],
        'account_login': ['login', 'password', 'account', 'access', 'locked', 'sign in', 'reset'],
        'technical_issue': ['website', 'app', 'error', 'crash', 'loading', 'broken', 'not working', 'page'],
        'cancellation': ['cancel', 'stop', 'unsubscribe', 'membership', 'subscription', 'recurring'],
        'billing_problem': ['billing', 'statement', 'charge', 'invoice', 'receipt'],
        'information_request': ['what', 'how', 'when', 'where', 'do you', 'can you', 'info', 'tell me', 'policy'],
        'complaint': ['unhappy', 'disappointed', 'worst', 'terrible', 'bad', 'poor', 'unsatisfied', 'angry', 'frustrated', 'horrible'],
        'other': []
    }
    
    def label_message(text: str) -> str:
        text_lower = text.lower()
        scores = {}
        for intent, keywords in intent_keywords.items():
            if not keywords:
                continue
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent] = score
        
        if scores:
            return max(scores, key=scores.get)
        return 'other'
    
    return label_message, intent_keywords


def create_escalation_labeler():
    """Create a keyword-based escalation labeler."""
    fraud_indicators = ['fraud', 'unauthorized', 'stolen', 'hacked', 'compromised', 'suspicious']
    legal_indicators = ['lawsuit', 'legal', 'attorney', 'sue', 'court']
    security_indicators = ['hacked', 'stolen', 'compromised', 'unauthorized access']
    severity_indicators = ['worst', 'terrible', 'horrible', 'disgusting', 'unacceptable']
    
    def should_escalate(text: str, intent: str) -> Tuple[bool, str]:
        text_lower = text.lower()
        
        if any(ind in text_lower for ind in fraud_indicators):
            return True, "Potential fraud detected"
        if any(ind in text_lower for ind in legal_indicators):
            return True, "Message references legal action"
        if intent == 'account_login' and any(ind in text_lower for ind in security_indicators):
            return True, "Security incident requiring immediate attention"
        if intent == 'complaint' and any(ind in text_lower for ind in severity_indicators):
            return True, "Severe complaint requiring de-escalation"
        
        # Low-confidence cases for the 'other' intent
        if intent == 'other':
            return True, "Ambiguous intent requiring human review"
        
        return False, ""
    
    return should_escalate


def split_data(df: pd.DataFrame, golden_size: int = 200, test_size: int = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, golden evaluation, and test sets.
    
    Ensures NO text leakage by deduplicating at the text level.
    Each unique text appears in exactly one split.
    """
    customer_msgs = df[df['inbound'] == 1].copy().reset_index(drop=True)
    
    # Label all messages first
    labeler, _ = create_keyword_labeler()
    customer_msgs['pseudo_intent'] = customer_msgs['text'].apply(labeler)
    
    # Deduplicate by text - keep first occurrence only
    unique_msgs = customer_msgs.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
    logger.info(f"Unique customer messages: {len(unique_msgs)} (from {len(customer_msgs)} total)")
    
    # Stratified sample for golden set from unique messages
    golden_indices = []
    intent_groups = unique_msgs.groupby('pseudo_intent')
    
    for intent_name, group in intent_groups:
        # Sample proportionally, ensuring at least 2 per intent if possible
        n_samples = min(max(2, golden_size // len(intent_groups)), len(group))
        if n_samples > 0:
            sampled = group.sample(n=n_samples, random_state=SEED)
            golden_indices.extend(sampled.index.tolist())
    
    # Trim to golden_size
    if len(golden_indices) > golden_size:
        golden_indices = golden_indices[:golden_size]
    
    golden_df = unique_msgs.loc[golden_indices].copy()
    remaining_df = unique_msgs.drop(golden_indices)
    
    # Split remaining into train and test
    test_n = max(1, int(len(remaining_df) * test_size))
    test_df = remaining_df.sample(n=test_n, random_state=SEED)
    train_df = remaining_df.drop(test_df.index)
    
    logger.info(f"Split: train={len(train_df)}, golden={len(golden_df)}, test={len(test_df)}")
    
    # Verify no text leakage
    train_texts = set(train_df['text'].tolist())
    golden_texts = set(golden_df['text'].tolist())
    test_texts = set(test_df['text'].tolist())
    
    overlap_tg = len(train_texts & golden_texts)
    overlap_tt = len(train_texts & test_texts)
    overlap_gt = len(golden_texts & test_texts)
    
    if overlap_tg or overlap_tt or overlap_gt:
        logger.error(f"DATA LEAKAGE DETECTED: train-golden={overlap_tg}, train-test={overlap_tt}, golden-test={overlap_gt}")
    else:
        logger.info("No data leakage detected - all texts are unique across splits")
    
    logger.info(f"Split: train={len(train_df)}, golden={len(golden_df)}, test={len(test_df)}")
    return train_df, golden_df, test_df


def create_golden_set(golden_df: pd.DataFrame) -> pd.DataFrame:
    """Create properly annotated golden set from held-out data."""
    labeler, intent_keywords = create_keyword_labeler()
    escalation_fn = create_escalation_labeler()
    
    golden_examples = []
    for idx, row in golden_df.iterrows():
        text = str(row['text'])
        intent = labeler(text)
        escalate, esc_reason = escalation_fn(text, intent)
        
        example = {
            'example_id': len(golden_examples) + 1,
            'conversation_id': str(row['tweet_id']),
            'customer_message': text,
            'intent': intent,
            'should_escalate': escalate,
            'escalation_reason': esc_reason,
            'expected_resolution': f"Address {intent.replace('_', ' ')} issue",
            'annotator_notes': f"Pseudo-labeled via keyword matching. Intent: {intent}",
            'source': 'synthetic_held_out'
        }
        golden_examples.append(example)
    
    golden_set_df = pd.DataFrame(golden_examples)
    return golden_set_df


def save_splits(train_df: pd.DataFrame, golden_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save data splits to CSV files."""
    # Save training data
    train_path = Path("data/processed/train.csv")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    
    # Save golden set
    golden_path = Path("evaluation/golden_set.csv")
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_df.to_csv(golden_path, index=False)
    
    # Save test data
    test_path = Path("data/processed/test.csv")
    test_df.to_csv(test_path, index=False)
    
    # Save metadata
    metadata = {
        'seed': SEED,
        'train_size': len(train_df),
        'golden_size': len(golden_df),
        'test_size': len(test_df),
        'total_size': len(train_df) + len(golden_df) + len(test_df),
        'labeling_method': 'keyword_pseudo_labeling',
        'data_source': 'synthetic_dataset',
        'caveat': 'All labels are machine-generated via keyword matching, not human-annotated'
    }
    
    metadata_path = Path("data/processed/split_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved train={len(train_df)}, golden={len(golden_df)}, test={len(test_df)}")
    logger.info(f"Metadata saved to {metadata_path}")


def print_data_report(df: pd.DataFrame, golden_df: pd.DataFrame):
    """Print a data report."""
    print("\n" + "=" * 60)
    print("DATA REPORT")
    print("=" * 60)
    print(f"\nFull dataset: {len(df)} rows")
    print(f"Customer messages: {len(df[df['inbound'] == 1])}")
    print(f"Brand messages: {len(df[df['inbound'] == 0])}")
    print(f"Unique authors: {df['author_id'].nunique()}")
    
    if 'created_at' in df.columns:
        print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    
    print(f"\nText length stats:")
    print(f"  Mean: {df['text_length'].mean():.1f}")
    print(f"  Median: {df['text_length'].median():.1f}")
    print(f"  Min: {df['text_length'].min()}")
    print(f"  Max: {df['text_length'].max()}")
    
    labeler, _ = create_keyword_labeler()
    intents = [labeler(str(t)) for t in df[df['inbound'] == 1]['text']]
    intent_dist = pd.Series(intents).value_counts()
    print(f"\nIntent distribution (customer messages):")
    for intent, count in intent_dist.items():
        print(f"  {intent}: {count} ({count/len(intents)*100:.1f}%)")
    
    print(f"\nGolden set: {len(golden_df)} examples")
    golden_intents = golden_df['intent'].value_counts()
    print("Golden set intent distribution:")
    for intent, count in golden_intents.items():
        print(f"  {intent}: {count}")
    
    print(f"\nIMPORTANT: This is a SYNTHETIC dataset.")
    print(f"Results should be interpreted with this caveat.")


def main():
    """Main data preparation pipeline."""
    print("Preparing data...")
    
    # Load data
    df = load_full_dataset()
    brand_df = load_brand_data()
    
    # Use brand-specific data for splits
    # Use golden_size=40 to leave enough for training (89 unique - 40 golden = 49 remaining)
    train_df, golden_df, test_df = split_data(brand_df, golden_size=40)
    
    # Create golden set with annotations
    golden_annotated = create_golden_set(golden_df)
    
    # Save everything
    save_splits(train_df, golden_annotated, test_df)
    
    # Print report
    print_data_report(brand_df, golden_annotated)
    
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
