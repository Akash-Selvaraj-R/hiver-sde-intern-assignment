"""
Baseline implementations for the customer support agent evaluation.
"""
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrivialBaseline:
    """Baseline 0: Always predict majority class, never escalate."""

    def __init__(self):
        self.majority_class = None
        self.classes_ = None

    def fit(self, texts: List[str], labels: List[str]):
        """Fit the baseline by finding the majority class."""
        if not labels:
            raise ValueError("No labels provided for fitting")

        # Find majority class
        label_counts = pd.Series(labels).value_counts()
        self.majority_class = label_counts.index[0]
        self.classes_ = sorted(list(set(labels)))
        logger.info(f"Trivial baseline fitted. Majority class: {self.majority_class}")

    def predict_intent(self, texts: List[str]) -> List[str]:
        """Always predict the majority class."""
        return [self.majority_class] * len(texts)

    def predict_proba(self, texts: List[str]) -> List[float]:
        """Return confidence scores (1.0 for majority class, 0.0 otherwise)."""
        # For simplicity, return fixed confidence
        return [1.0] * len(texts)

    def predict_escalation(self, texts: List[str]) -> List[bool]:
        """Never escalate."""
        return [False] * len(texts)

    def predict_escalation_reason(self, texts: List[str]) -> List[str]:
        """No escalation reason since never escalating."""
        return [None] * len(texts)


class TfIdfBaseline:
    """Baseline 1: TF-IDF + Logistic Regression for intent classification, TF-IDF retrieval."""

    def __init__(self):
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
        self.is_fitted = False
        self.classes_ = None
        self.tfidf_matrix = None
        self.retrieval_texts = None

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def fit(self, texts: List[str], labels: List[str], retrieval_texts: List[str] = None):
        """
        Fit the baseline model.

        Args:
            texts: Training texts for classification
            labels: Corresponding intent labels
            retrieval_texts: Texts to use for retrieval index (if None, uses texts)
        """
        if not texts or not labels:
            raise ValueError("No training data provided")

        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have same length")

        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]

        # Fit TF-IDF vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_texts)

        # Fit classifier
        self.classifier.fit(self.tfidf_matrix, labels)
        self.classes_ = sorted(list(set(labels)))
        self.is_fitted = True

        # Store retrieval texts (for getting responses)
        if retrieval_texts is None:
            self.retrieval_texts = texts
        else:
            self.retrieval_texts = [self._preprocess_text(text) for text in retrieval_texts]

        logger.info(f"TF-IDF baseline fitted. Classes: {self.classes_}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def predict_intent(self, texts: List[str]) -> List[str]:
        """Predict intent for given texts."""
        if not self.is_fitted:
            raise RuntimeError("Baseline must be fitted before prediction")

        processed_texts = [self._preprocess_text(text) for text in texts]
        tfidf_features = self.vectorizer.transform(processed_texts)
        predictions = self.classifier.predict(tfidf_features)
        return list(predictions)

    def predict_proba(self, texts: List[str]) -> List[float]:
        """Get confidence scores for predictions."""
        if not self.is_fitted:
            raise RuntimeError("Baseline must be fitted before prediction")

        processed_texts = [self._preprocess_text(text) for text in texts]
        tfidf_features = self.vectorizer.transform(processed_texts)
        probabilities = self.classifier.predict_proba(tfidf_features)
        # Return max probability for each prediction
        max_probs = np.max(probabilities, axis=1)
        return list(max_probs)

    def retrieve_similar_cases(self, query_text: str, k: int = 5) -> List[Dict]:
        """
        Retrieve similar cases using TF-IDF cosine similarity.

        Args:
            query_text: The query text to find similar cases for
            k: Number of similar cases to retrieve

        Returns:
            List of dictionaries with similarity scores and metadata
        """
        if not self.is_fitted:
            raise RuntimeError("Baseline must be fitted before retrieval")

        if self.retrieval_texts is None or len(self.retrieval_texts) == 0:
            return []

        # Preprocess query
        processed_query = self._preprocess_text(query_text)
        query_vector = self.vectorizer.transform([processed_query])

        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            if idx < len(self.retrieval_texts):
                results.append({
                    'index': int(idx),
                    'similarity': float(similarities[idx]),
                    'text': self.retrieval_texts[idx] if idx < len(self.retrieval_texts) else ""
                })

        return results

    def predict_escalation(self, texts: List[str]) -> List[bool]:
        """Simple escalation based on low confidence."""
        if not self.is_fitted:
            raise RuntimeError("Baseline must be fitted before prediction")

        confidences = self.predict_proba(texts)
        # Escalate if confidence is below threshold
        threshold = 0.5  # Simple threshold
        escalations = [conf < threshold for conf in confidences]
        return escalations

    def predict_escalation_reason(self, texts: List[str]) -> List[str]:
        """Provide escalation reasons."""
        if not self.is_fitted:
            raise RuntimeError("Baseline must be fitted before prediction")

        confidences = self.predict_proba(texts)
        reasons = []
        for conf in confidences:
            if conf < 0.5:
                reasons.append("Low confidence in intent prediction")
            else:
                reasons.append(None)
        return reasons


def load_training_data() -> Tuple[List[str], List[str], List[str]]:
    """
    Load training data for baselines from the prepared split.
    Avoids golden set to prevent leakage.
    """
    train_path = Path("data/processed/train.csv")
    
    if train_path.exists():
        df = pd.read_csv(train_path)
        customer_messages = df['text'].tolist()
        
        if 'intent' in df.columns:
            pseudo_labels = df['intent'].tolist()
        else:
            pseudo_labels = _pseudo_label_batch(customer_messages)
        
        retrieval_texts = customer_messages
    else:
        # Fallback
        config_path = Path("configs/brand.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        brand_name = config['brand_name']
        data_path = Path(config['dataset_path']) / f"{brand_name}_conversations.csv"

        df = pd.read_csv(data_path)
        customer_messages = df[df['inbound'] == 1]['text'].tolist()
        pseudo_labels = _pseudo_label_batch(customer_messages)
        retrieval_texts = customer_messages

    return customer_messages, pseudo_labels, retrieval_texts


def _pseudo_label_batch(texts):
    """Assign pseudo-labels via keyword matching."""
    labels = []
    for msg in texts:
        msg_lower = msg.lower()
        if any(word in msg_lower for word in ['driver', 'ride', 'trip', 'pickup', 'dropoff', 'eta', 'waiting', 'cancelled', 'cancel on', 'no show', 'arrived', 'destination', 'route', 'stranded']):
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


def evaluate_baseline(baseline, golden_set_path: str = "evaluation/golden_set.json") -> Dict[str, Any]:
    """
    Evaluate a baseline on the golden set.

    Args:
        baseline: The baseline model to evaluate
        golden_set_path: Path to the golden set (JSON or CSV)

    Returns:
        Dictionary with evaluation metrics
    """
    # Load golden set (support both JSON and CSV)
    golden_path = Path(golden_set_path)
    if golden_path.suffix == '.json':
        with open(golden_path, 'r') as f:
            golden_df = pd.DataFrame(json.load(f))
    else:
        golden_df = pd.read_csv(golden_path)

    # Get test data
    test_texts = golden_df['customer_message'].tolist()
    true_intents = golden_df['intent'].tolist()
    true_escalation = golden_df['should_escalate'].tolist()

    # Make predictions
    pred_intents = baseline.predict_intent(test_texts)
    pred_confidences = baseline.predict_proba(test_texts)
    pred_escalation = baseline.predict_escalation(test_texts)
    pred_escalation_reasons = baseline.predict_escalation_reason(test_texts)

    # Calculate intent metrics
    intent_accuracy = accuracy_score(true_intents, pred_intents)
    intent_f1_macro = f1_score(true_intents, pred_intents, average='macro')
    intent_f1_weighted = f1_score(true_intents, pred_intents, average='weighted')

    # Calculate escalation metrics
    escalation_accuracy = accuracy_score(true_escalation, pred_escalation)
    # Handle case where all predictions are same class
    try:
        escalation_f1 = f1_score(true_escalation, pred_escalation)
    except:
        escalation_f1 = 0.0

    # Per-intent metrics
    try:
        intent_report = classification_report(true_intents, pred_intents, output_dict=True)
    except:
        intent_report = {}

    return {
        'intent_accuracy': intent_accuracy,
        'intent_f1_macro': intent_f1_macro,
        'intent_f1_weighted': intent_f1_weighted,
        'intent_classification_report': intent_report,
        'escalation_accuracy': escalation_accuracy,
        'escalation_f1': escalation_f1,
        'predictions': {
            'intents': pred_intents,
            'confidences': pred_confidences,
            'escalation': pred_escalation,
            'escalation_reasons': pred_escalation_reasons
        }
    }


def run_baselines():
    """Run both baselines and print results."""
    print("Loading training data...")
    train_texts, train_labels, retrieval_texts = load_training_data()

    print(f"Loaded {len(train_texts)} training examples")
    print(f"Label distribution: {pd.Series(train_labels).value_counts().to_dict()}")

    # Baseline 0: Trivial
    print("\n=== Training Trivial Baseline ===")
    trivial_baseline = TrivialBaseline()
    trivial_baseline.fit(train_texts, train_labels)

    print("\n=== Evaluating Trivial Baseline ===")
    trivial_results = evaluate_baseline(trivial_baseline)

    # Baseline 1: TF-IDF
    print("\n=== Training TF-IDF Baseline ===")
    tfidf_baseline = TfIdfBaseline()
    tfidf_baseline.fit(train_texts, train_labels, retrieval_texts)

    print("\n=== Evaluating TF-IDF Baseline ===")
    tfidf_results = evaluate_baseline(tfidf_baseline)

    # Print results
    print("\n" + "="*50)
    print("BASELINE RESULTS")
    print("="*50)

    print("\n--- Trivial Baseline ---")
    print(f"Intent Accuracy: {trivial_results['intent_accuracy']:.3f}")
    print(f"Intent Macro F1: {trivial_results['intent_f1_macro']:.3f}")
    print(f"Intent Weighted F1: {trivial_results['intent_f1_weighted']:.3f}")
    print(f"Escalation Accuracy: {trivial_results['escalation_accuracy']:.3f}")
    print(f"Escalation F1: {trivial_results['escalation_f1']:.3f}")

    print("\n--- TF-IDF Baseline ---")
    print(f"Intent Accuracy: {tfidf_results['intent_accuracy']:.3f}")
    print(f"Intent Macro F1: {tfidf_results['intent_f1_macro']:.3f}")
    print(f"Intent Weighted F1: {tfidf_results['intent_f1_weighted']:.3f}")
    print(f"Escalation Accuracy: {tfidf_results['escalation_accuracy']:.3f}")
    print(f"Escalation F1: {tfidf_results['escalation_f1']:.3f}")

    # Save results for later comparison
    results = {
        'trivial': trivial_results,
        'tfidf': tfidf_results
    }

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    # Save to file
    import json
    with open('evaluation/baseline_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print("\nBaseline results saved to evaluation/baseline_results.json")

    return results


if __name__ == "__main__":
    run_baselines()