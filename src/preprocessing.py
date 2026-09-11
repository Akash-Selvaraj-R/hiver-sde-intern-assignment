"""
Text preprocessing utilities for customer support messages.
"""
import re
import string
from typing import Dict, List, Union
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.

    Args:
        text: Input text string

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user mentions and hashtags (but keep the text)
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove punctuation (optional - might remove meaningful info)
    # text = text.translate(str.maketrans('', '', string.punctuation))

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def preprocess_conversation(messages: List[Dict]) -> List[Dict]:
    """
    Preprocess a list of conversation messages.

    Args:
        messages: List of message dictionaries with 'text' field

    Returns:
        List of preprocessed message dictionaries
    """
    processed_messages = []

    for msg in messages:
        processed_msg = msg.copy()
        processed_msg['text_clean'] = clean_text(msg.get('text', ''))
        processed_msg['text_length'] = len(processed_msg['text_clean'])
        processed_messages.append(processed_msg)

    return processed_messages


def extract_conversation_context(conversation: List[Dict],
                                current_message_index: int,
                                context_window: int = 2) -> str:
    """
    Extract context window around a current message.

    Args:
        conversation: List of message dictionaries in chronological order
        current_message_index: Index of the current message
        context_window: Number of messages before and after to include

    Returns:
        Concatenated context string
    """
    start_idx = max(0, current_message_index - context_window)
    end_idx = min(len(conversation), current_message_index + context_window + 1)

    context_messages = []
    for i in range(start_idx, end_idx):
        msg = conversation[i]
        prefix = "CUSTOMER: " if msg.get('inbound', False) else "BRAND: "
        context_messages.append(f"{prefix}{msg.get('text_clean', '')}")

    return " [SEP] ".join(context_messages)


def batch_preprocess_texts(texts: List[str]) -> List[str]:
    """
    Preprocess a batch of texts efficiently.

    Args:
        texts: List of input text strings

    Returns:
        List of cleaned text strings
    """
    return [clean_text(text) for text in texts]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    sample_texts = [
        "Hello! I need help with my order #12345. https://example.com @support",
        "WHERE IS MY REFUND??? This is unacceptable!!!",
        "Thanks for the help! Great service."
    ]

    cleaned = batch_preprocess_texts(sample_texts)
    for original, cleaned_text in zip(sample_texts, cleaned):
        print(f"Original: {original}")
        print(f"Cleaned:  {cleaned_text}")
        print()