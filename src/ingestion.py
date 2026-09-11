"""
Data ingestion module for loading and processing the Twitter customer support dataset.
"""
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_twitter_support_data(data_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load the Twitter customer support dataset.

    Args:
        data_path: Path to the dataset CSV file
        sample_size: Optional sample size for development

    Returns:
        DataFrame with columns: tweet_id, author_id, inbound, text, created_at
    """
    logger.info(f"Loading data from {data_path}")

    # The dataset has specific column names based on Kaggle description
    df = pd.read_csv(data_path)

    # Rename columns to match expected format
    # Based on Kaggle dataset description: tweet_id, author_id, inbound, text, created_at
    expected_columns = ['tweet_id', 'author_id', 'inbound', 'text', 'created_at']

    # Check if we need to rename columns
    if len(df.columns) >= 5:
        df.columns = expected_columns[:len(df.columns)]

    # Ensure we have the required columns
    required_columns = ['tweet_id', 'author_id', 'inbound', 'text', 'created_at']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}. Available columns: {df.columns.tolist()}")
        # Create missing columns with default values
        for col in missing_columns:
            if col == 'inbound':
                df[col] = 1  # Assume inbound (customer message) if not specified
            elif col == 'created_at':
                df[col] = pd.Timestamp.now()
            else:
                df[col] = ''

    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} rows from dataset")

    # Basic cleaning
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)

    logger.info(f"Loaded {len(df)} tweets")
    return df


def extract_conversations(df: pd.DataFrame) -> List[Dict]:
    """
    Extract conversation threads from the dataset.

    Args:
        df: DataFrame with tweet data

    Returns:
        List of conversation dictionaries
    """
    logger.info("Extracting conversations from tweet data")

    # Group by author_id to simulate conversations (simplified approach)
    # In reality, we'd need to use conversation IDs or reply structures
    conversations = []

    # For now, we'll treat each tweet as a potential conversation start
    # This is a simplification - in a real system we'd use actual thread structure
    for _, row in df.iterrows():
        conversation = {
            'conversation_id': str(row['tweet_id']),
            'author_id': str(row['author_id']),
            'messages': [{
                'tweet_id': str(row['tweet_id']),
                'author_id': str(row['author_id']),
                'inbound': bool(row['inbound']),
                'text': str(row['text']),
                'created_at': str(row['created_at']) if pd.notna(row['created_at']) else None
            }]
        }
        conversations.append(conversation)

    logger.info(f"Extracted {len(conversations)} conversations")
    return conversations


def separate_customer_brand_messages(conversations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Separate customer (inbound) and brand (outbound) messages.

    Args:
        conversations: List of conversation dictionaries

    Returns:
        Tuple of (customer_messages, brand_messages)
    """
    customer_messages = []
    brand_messages = []

    for conv in conversations:
        for msg in conv['messages']:
            msg_copy = msg.copy()
            msg_copy['conversation_id'] = conv['conversation_id']

            if msg['inbound']:
                customer_messages.append(msg_copy)
            else:
                brand_messages.append(msg_copy)

    logger.info(f"Separated {len(customer_messages)} customer messages and {len(brand_messages)} brand messages")
    return customer_messages, brand_messages


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    df = load_twitter_support_data("data/sample/twitter_support.csv", sample_size=1000)
    conversations = extract_conversations(df)
    customer_msgs, brand_msgs = separate_customer_brand_messages(conversations)
    print(f"Loaded {len(conversations)} conversations")
    print(f"Customer messages: {len(customer_msgs)}")
    print(f"Brand messages: {len(brand_msgs)}")