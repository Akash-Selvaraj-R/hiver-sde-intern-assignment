"""
Analyze brands in the real Kaggle dataset and recommend the best one.

Usage:
    python scripts/analyze_brands_real.py
"""
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_real_dataset() -> pd.DataFrame:
    """Load the real dataset from data/raw/."""
    raw_dir = Path("data/raw")
    candidates = ["twcs.csv", "customer_support_on_twitter.csv"]
    for fname in candidates:
        path = raw_dir / fname
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError("No dataset found in data/raw/. Run the download first.")


def analyze_brands(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Analyze brand accounts in the dataset."""
    if 'inbound' not in df.columns or 'author_id' not in df.columns:
        raise ValueError("Dataset must have 'inbound' and 'author_id' columns")

    brand_df = df[df['inbound'] == 0]  # Brand messages
    customer_df = df[df['inbound'] == 1]  # Customer messages

    # Count brand messages
    brand_msg_counts = brand_df['author_id'].value_counts().head(top_n)

    # For each brand, count customer responses
    results = []
    for brand, brand_count in brand_msg_counts.items():
        brand_tweet_ids = set(brand_df[brand_df['author_id'] == brand]['tweet_id'].tolist()) if 'tweet_id' in brand_df.columns else set()

        if 'in_response_to_tweet_id' in df.columns:
            customer_count = len(customer_df[customer_df['in_response_to_tweet_id'].isin(brand_tweet_ids)])
        else:
            customer_count = brand_count

        # Conversation diversity (unique customer authors)
        if 'in_response_to_tweet_id' in df.columns:
            responding_customers = customer_df[customer_df['in_response_to_tweet_id'].isin(brand_tweet_ids)]['author_id'].nunique()
        else:
            responding_customers = customer_count

        results.append({
            'brand': brand,
            'brand_messages': brand_count,
            'customer_responses': customer_count,
            'unique_customers': responding_customers,
            'score': brand_count + customer_count * 0.5,
        })

    return pd.DataFrame(results).sort_values('score', ascending=False)


def main():
    """Main function."""
    print("Analyzing brands in real dataset...")

    try:
        df = load_real_dataset()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Download the dataset first. See data/raw/README.md")
        return

    print(f"Loaded {len(df)} rows")

    analysis = analyze_brands(df)
    print("\n=== Top 20 Brands ===")
    print(analysis.to_string(index=False))

    # Save analysis
    output_path = Path("experiments/brand_analysis_real.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
