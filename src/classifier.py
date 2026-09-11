"""
Intent classification module using TF-IDF and Logistic Regression.
"""
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Intent classifier using TF-IDF vectorization and Logistic Regression.
    """

    def __init__(self, config_path: str = "configs/brand.yaml"):
        """Initialize classifier with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.brand_name = self.config['brand_name']
        self.confidence_threshold = self.config['confidence_threshold']

        # Initialize model components
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', self.classifier)
        ])

        self.is_fitted = False
        self.classes_ = None
        self.intent_descriptions = {}  # Will be loaded from intents.yaml

        # Load intent descriptions for reference
        self._load_intent_descriptions()

    def _load_intent_descriptions(self):
        """Load intent descriptions from intents.yaml."""
        try:
            with open("configs/intents.yaml", 'r') as f:
                data = yaml.safe_load(f)
                for intent in data.get("intents", []):
                    self.intent_descriptions[intent['name']] = intent['description']
        except FileNotFoundError:
            logger.warning("intents.yaml not found, proceeding without descriptions")

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing for consistency."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def fit(self, texts: List[str], labels: List[str]) -> 'IntentClassifier':
        """
        Fit the classifier on training data.

        Args:
            texts: Training text samples
            labels: Corresponding intent labels

        Returns:
            Self for method chaining
        """
        if not texts or not labels:
            raise ValueError("No training data provided")

        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have same length")

        logger.info(f"Fitting classifier on {len(texts)} samples")

        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]

        # Fit the pipeline
        self.pipeline.fit(processed_texts, labels)
        self.classes_ = self.pipeline.named_steps['clf'].classes_
        self.is_fitted = True

        logger.info(f"Classifier fitted. Classes: {list(self.classes_)}")
        return self

    def predict_intent(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict intent for given texts with confidence scores.

        Args:
            texts: List of text samples to classify

        Returns:
            List of dictionaries with 'intent' and 'confidence' keys
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")

        if not texts:
            return []

        processed_texts = [self._preprocess_text(text) for text in texts]

        # Get predictions and probabilities
        predictions = self.pipeline.predict(processed_texts)
        probabilities = self.pipeline.predict_proba(processed_texts)

        # Format results
        results = []
        for pred, probs in zip(predictions, probabilities):
            confidence = float(np.max(probs))
            results.append({
                'intent': str(pred),
                'confidence': confidence
            })

        return results

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict intent for a single text sample.

        Args:
            text: Text sample to classify

        Returns:
            Dictionary with 'intent' and 'confidence' keys
        """
        results = self.predict_intent([text])
        return results[0]

    def get_feature_importance(self, intent: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top features for a specific intent.

        Args:
            intent: Intent name to get features for
            top_n: Number of top features to return

        Returns:
            List of (feature, weight) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before getting feature importance")

        if intent not in self.classes_:
            raise ValueError(f"Intent '{intent}' not found in classes: {list(self.classes_)}")

        # Get coefficient for the intent
        intent_idx = list(self.classes_).index(intent)
        coef = self.pipeline.named_steps['clf'].coef_[intent_idx]
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()

        # Get top features
        top_indices = np.argsort(np.abs(coef))[::-1][:top_n]
        top_features = [(feature_names[i], float(coef[i])) for i in top_indices]

        return top_features


def load_training_data() -> Tuple[List[str], List[str]]:
    """
    Load training data for the classifier.
    Uses the prepared training split with keyword pseudo-labels.
    """
    train_path = Path("data/processed/train.csv")
    
    if train_path.exists():
        df = pd.read_csv(train_path)
        customer_messages = df['text'].tolist()
        
        # If the CSV already has intent labels, use them
        if 'intent' in df.columns:
            labels = df['intent'].tolist()
        else:
            labels = _pseudo_label_batch(customer_messages)
    else:
        # Fallback to brand-specific data
        config_path = Path("configs/brand.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        brand_name = config['brand_name']
        data_path = Path(config['dataset_path']) / f"{brand_name}_conversations.csv"
        
        df = pd.read_csv(data_path)
        customer_messages = df[df['inbound'] == 1]['text'].tolist()
        labels = _pseudo_label_batch(customer_messages)

    logger.info(f"Loaded {len(customer_messages)} training samples")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    return customer_messages, labels


def _pseudo_label_batch(texts: List[str]) -> List[str]:
    """Assign pseudo-labels to a batch of texts using keyword matching."""
    labels = []
    for msg in texts:
        msg_lower = msg.lower()
        if any(word in msg_lower for word in ['driver', 'ride', 'trip', 'pickup', 'dropoff', 'eta', 'waiting', 'cancelled', 'cancel on', 'no show', 'arrived', 'destination', 'route', 'stranded', 'where is', 'how long']):
            labels.append('ride_status')
        elif any(word in msg_lower for word in ['charge', 'fare', 'surge', 'price', 'cost', 'double', 'receipt', 'charged', 'billing', 'overcharged', 'amount', 'estimate']):
            labels.append('trip_fare_dispute')
        elif any(word in msg_lower for word in ['login', 'password', 'account', 'banned', 'deactivated', 'locked', 'sign in', 'reset', 'verify', 'verification', 'otp', 'phone number', 'email']):
            labels.append('account_access')
        elif any(word in msg_lower for word in ['app', 'crash', 'bug', 'gps', 'map', 'loading', 'broken', 'not working', 'glitch', 'update', 'download', 'install', 'error', 'notification']):
            labels.append('app_functionality')
        elif any(word in msg_lower for word in ['unhappy', 'disappointed', 'worst', 'terrible', 'bad', 'poor', 'unsatisfied', 'angry', 'frustrated', 'horrible', 'awful', 'rude', 'never again', 'scam', 'fraud', 'pathetic', 'joke']):
            labels.append('complaint')
        elif any(word in msg_lower for word in ['refund', 'money back', 'return', 'give me back', 'credit back', 'reimburse']):
            labels.append('refund_request')
        elif any(word in msg_lower for word in ['cancel my', 'cancel account', 'unsubscribe', 'subscription', 'delete account', 'stop subscription', 'cancellation fee']):
            labels.append('cancellation')
        else:
            labels.append('other')
    return labels


def main():
    """Main function to demonstrate classifier usage."""
    print("Loading training data...")
    train_texts, train_labels = load_training_data()

    print(f"Loaded {len(train_texts)} training examples")

    # Initialize and fit classifier
    print("\nTraining intent classifier...")
    classifier = IntentClassifier()
    classifier.fit(train_texts, train_labels)

    # Test with some examples
    test_examples = [
        "Where is my order #12345?",
        "I want a refund for my purchase",
        "I was charged twice for the same item",
        "I can't log into my account",
        "The website keeps crashing when I try to checkout",
        "How do I cancel my subscription?",
        "What is your return policy?",
        "I'm very unhappy with the service I received",
        "Thanks for your help!",
        "My payment didn't go through"
    ]

    print("\n=== Test Predictions ===")
    for example in test_examples:
        result = classifier.predict_single(example)
        print(f"Text: {example}")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
        print()


if __name__ == "__main__":
    main()