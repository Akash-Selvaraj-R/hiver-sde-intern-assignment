"""
Filter dataset to focus on selected brand for intent taxonomy derivation.
"""
import pandas as pd
import json
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    with open("configs/brand.yaml", 'r') as f:
        return yaml.safe_load(f)

def filter_brand_conversations(df, brand_name):
    """
    Filter conversations to only include those involving the selected brand.

    Args:
        df: DataFrame with tweet data
        brand_name: The brand account ID to focus on

    Returns:
        Filtered DataFrame containing only brand-related conversations
    """
    logger.info(f"Filtering conversations for brand: {brand_name}")

    # Get all tweetIDs that are either from the brand or are responses to the brand
    brand_tweets = df[df['author_id'] == brand_name]['tweet_id'].tolist()

    # For simplicity in this synthetic dataset, we'll consider all messages
    # where the brand is involved (either sending or receiving)
    # In a real dataset, we would use conversation/thread IDs

    # Since we don't have explicit conversation IDs in our synthetic data,
    # we'll use a simplified approach: assume that messages close in time
    # from the same general area are part of conversations
    # But for now, let's just get all brand messages and a sample of customer messages

    brand_messages = df[df['author_id'] == brand_name].copy()
    logger.info(f"Found {len(brand_messages)} messages from brand {brand_name}")

    # For customer messages, let's get those that mention common brand-related terms
    # or are simply a sample to pair with brand messages
    customer_messages = df[(df['inbound'] == 1) & (df['author_id'] != brand_name)].copy()

    # Take a proportional sample of customer messages
    sample_size = min(len(customer_messages), len(brand_messages) * 3)  # 3:1 ratio
    if len(customer_messages) > sample_size:
        customer_messages = customer_messages.sample(n=sample_size, random_state=42)

    logger.info(f"Selected {len(customer_messages)} customer messages for analysis")

    # Combine brand and customer messages
    filtered_df = pd.concat([brand_messages, customer_messages], ignore_index=True)

    # Sort by timestamp if available
    if 'created_at' in filtered_df.columns:
        filtered_df['created_at'] = pd.to_datetime(filtered_df['created_at'], errors='coerce')
        filtered_df = filtered_df.sort_values('created_at').reset_index(drop=True)

    logger.info(f"Total filtered dataset size: {len(filtered_df)} messages")
    return filtered_df

def save_filtered_data(df, output_path):
    """Save filtered dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Filtered dataset saved to {output_path}")

def analyze_filtered_data(df):
    """Perform basic analysis on filtered data."""
    logger.info("=== Filtered Data Analysis ===")
    logger.info(f"Total messages: {len(df)}")
    logger.info(f"Brand messages: {len(df[df['author_id'] == 'brand_account_5'])}")
    logger.info(f"Customer messages: {len(df[df['inbound'] == 1])}")

    if 'created_at' in df.columns:
        logger.info(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")

    # Text length stats
    df['text_length'] = df['text'].astype(str).apply(len)
    logger.info(f"Avg message length: {df['text_length'].mean():.1f} chars")

    # Sample messages
    logger.info("\nSample brand messages:")
    brand_sample = df[df['author_id'] == 'brand_account_5']['text'].head(3)
    for i, msg in enumerate(brand_sample, 1):
        logger.info(f"{i}. {msg}")

    logger.info("\nSample customer messages:")
    customer_sample = df[df['inbound'] == 1]['text'].head(3)
    for i, msg in enumerate(customer_sample, 1):
        logger.info(f"{i}. {msg}")

def main():
    # Load configuration
    config = load_config()
    brand_name = config['brand_name']

    # Load the full dataset
    input_path = Path(config['dataset_path']) / "twitter_support_synthetic.csv"
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)

    # Filter for brand data
    filtered_df = filter_brand_conversations(df, brand_name)

    # Analyze filtered data
    analyze_filtered_data(filtered_df)

    # Save filtered dataset
    output_dir = Path("data") / "brand_specific"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{brand_name}_conversations.csv"
    save_filtered_data(filtered_df, output_path)

    # Update config with brand-specific dataset path
    config['dataset_path'] = str(output_dir)
    with open("configs/brand.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Updated brand.yaml with brand-specific dataset path: {output_dir}")

if __name__ == "__main__":
    main()