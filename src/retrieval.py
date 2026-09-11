"""
Historical case retrieval module using sentence transformers and FAISS.
"""
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalRetriever:
    """
    Retrieves historical support cases using sentence embeddings and FAISS index.
    """

    def __init__(self, config_path: str = "configs/brand.yaml"):
        """Initialize retriever with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.brand_name = self.config['brand_name']
        self.top_k = self.config['top_k_retrieval']

        # Initialize sentence transformer model
        logger.info("Loading sentence transformer model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = self.encoder.get_embedding_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.is_fitted = False

        # Storage for metadata
        self.cases = []  # List of case dictionaries
        self.case_ids = []  # List of case IDs for mapping

        # Cache directory for embeddings and index
        self.cache_dir = Path("data/embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_create_cache_key(self, texts: List[str]) -> str:
        """Create a cache key based on the input texts."""
        # Create a hash of the texts for caching
        text_string = "|".join(sorted(texts))  # Sort to ensure consistent ordering
        return hashlib.md5(text_string.encode()).hexdigest()

    def _get_cache_paths(self, cache_key: str) -> Tuple[Path, Path]:
        """Get paths for cached embeddings and index."""
        embeddings_path = self.cache_dir / f"embeddings_{cache_key}.npy"
        index_path = self.cache_dir / f"index_{cache_key}.faiss"
        metadata_path = self.cache_dir / f"metadata_{cache_key}.pkl"
        return embeddings_path, index_path, metadata_path

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = text.strip()
        return text

    def fit(self, cases: List[Dict[str, Any]]) -> 'HistoricalRetriever':
        """
        Fit the retriever on historical cases.

        Args:
            cases: List of case dictionaries with keys:
                  - 'conversation_id': str
                  - 'customer_message': str
                  - 'brand_response': str (optional)
                  - 'resolution': str (optional)
                  - 'intent': str
                  - ... other metadata

        Returns:
            Self for method chaining
        """
        if not cases:
            raise ValueError("No cases provided for fitting")

        logger.info(f"Fitting retriever on {len(cases)} historical cases")

        # Store cases
        self.cases = cases
        self.case_ids = [str(case.get('conversation_id', f'case_{i}'))
                        for i, case in enumerate(cases)]

        # Extract texts for embedding (combine customer message and context if available)
        texts_to_encode = []
        for case in cases:
            customer_msg = self._preprocess_text(case.get('customer_message', ''))
            # Optionally include brand response or resolution for richer context
            brand_resp = self._preprocess_text(case.get('brand_response', ''))
            resolution = self._preprocess_text(case.get('resolution', ''))

            # Combine texts with separator
            combined_text = customer_msg
            if brand_resp:
                combined_text += " [RESP] " + brand_resp
            if resolution:
                combined_text += " [RES] " + resolution

            texts_to_encode.append(combined_text)

        # Try to load from cache
        cache_key = self._load_or_create_cache_key(texts_to_encode)
        embeddings_path, index_path, metadata_path = self._get_cache_paths(cache_key)

        if embeddings_path.exists() and index_path.exists() and metadata_path.exists():
            logger.info("Loading cached embeddings and index...")
            try:
                # Load embeddings
                embeddings = np.load(embeddings_path)

                # Load index
                self.index = faiss.read_index(str(index_path))

                # Load metadata
                with open(metadata_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.cases = cached_data['cases']
                    self.case_ids = cached_data['case_ids']

                logger.info(f"Loaded cached index with {len(self.cases)} cases")
                self.is_fitted = True
                return self
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # Compute embeddings
        logger.info("Computing embeddings for historical cases...")
        embeddings = self.encoder.encode(
            texts_to_encode,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity with inner product
        )

        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        self.is_fitted = True

        # Save to cache
        logger.info("Saving embeddings and index to cache...")
        try:
            np.save(embeddings_path, embeddings)
            faiss.write_index(self.index, str(index_path))
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'cases': self.cases,
                    'case_ids': self.case_ids
                }, f)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        logger.info(f"Retriever fitted with {len(self.cases)} cases")
        return self

    def retrieve(self, query_text: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar historical cases for a query text.

        Args:
            query_text: The customer message to find similar cases for
            k: Number of cases to retrieve (uses self.top_k if None)

        Returns:
            List of dictionaries with case information and similarity scores
        """
        if not self.is_fitted:
            raise RuntimeError("Retriever must be fitted before retrieval")

        if k is None:
            k = self.top_k

        # Preprocess and encode query
        processed_query = self._preprocess_text(query_text)
        query_embedding = self.encoder.encode(
            [processed_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Search index
        similarities, indices = self.index.search(
            query_embedding.astype('float32'),
            k
        )

        # Format results
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.cases):  # Valid index
                case = self.cases[idx].copy()
                case['similarity'] = float(similarity)
                case['case_id'] = self.case_ids[idx]
                results.append(case)

        return results

    def retrieve_by_intent(self, query_text: str, intent: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar cases filtered by intent.

        Args:
            query_text: The customer message to find similar cases for
            intent: The intent to filter cases by
            k: Number of cases to retrieve

        Returns:
            List of dictionaries with case information and similarity scores
        """
        if not self.is_fitted:
            raise RuntimeError("Retriever must be fitted before retrieval")

        # First get all similar cases
        all_results = self.retrieve(query_text, k=min(len(self.cases), 50))  # Get more to filter

        # Filter by intent
        intent_results = [case for case in all_results if case.get('intent') == intent]

        # Return top k
        return intent_results[:k] if k else intent_results

    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific case by its ID.

        Args:
            case_id: The case ID to retrieve

        Returns:
            Case dictionary or None if not found
        """
        try:
            idx = self.case_ids.index(case_id)
            return self.cases[idx].copy()
        except ValueError:
            return None


def load_historical_cases() -> List[Dict[str, Any]]:
    """
    Load historical cases from brand-specific training data.
    Only loads from the training split to prevent data leakage.
    """
    train_path = Path("data/processed/train.csv")
    cases = []
    
    if train_path.exists():
        df = pd.read_csv(train_path)
        logger.info(f"Loaded {len(df)} training examples for historical cases")
        
        has_brand_response = 'brand_response' in df.columns
        if not has_brand_response:
            logger.warning("train.csv missing brand_response column; responses will be empty")
        
        for _, row in df.iterrows():
            text = str(row.get('customer_message', row.get('text', '')))
            brand_resp = str(row.get('brand_response', '')) if has_brand_response else ''
            intent = str(row.get('intent', 'other'))
            
            case = {
                'conversation_id': str(row.get('conversation_id', row.get('tweet_id', f'case_{len(cases)}'))),
                'customer_message': text,
                'brand_response': brand_resp,
                'resolution': f"Resolved {intent.replace('_', ' ')} issue",
                'intent': intent,
                'created_at': str(row.get('timestamp', row.get('created_at', '')))
            }
            cases.append(case)
    else:
        # Fallback to brand-specific data
        config_path = Path("configs/brand.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        brand_name = config['brand_name']
        data_path = Path(config['dataset_path']) / f"{brand_name}_conversations.csv"

        df = pd.read_csv(data_path)

        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df = df.sort_values('created_at').reset_index(drop=True)

        for i, row in df.iterrows():
            if row['inbound'] == 1:
                case = {
                    'conversation_id': str(row['tweet_id']),
                    'customer_message': str(row['text']),
                    'created_at': str(row['created_at']) if pd.notna(row['created_at']) else None
                }

                brand_response = ""
                for j in range(i+1, min(i+5, len(df))):
                    if df.iloc[j]['inbound'] == 0 and df.iloc[j]['author_id'] == config['brand_name']:
                        brand_response = str(df.iloc[j]['text'])
                        break

                if brand_response:
                    case['brand_response'] = brand_response

                case['resolution'] = "Issue addressed according to standard procedures"
                case['intent'] = _pseudo_label_intent(str(row['text']))
                cases.append(case)

    logger.info(f"Loaded {len(cases)} historical cases for retrieval")
    return cases


def _pseudo_label_intent(text: str) -> str:
    """Assign a pseudo-label intent based on keywords (same as in classifier)."""
    text_lower = text.lower()
    if any(word in text_lower for word in ['driver', 'ride', 'trip', 'pickup', 'dropoff', 'eta', 'waiting', 'cancelled', 'cancel on', 'no show', 'arrived', 'destination', 'route', 'stranded']):
        return 'ride_status'
    elif any(word in text_lower for word in ['charge', 'fare', 'surge', 'price', 'cost', 'double', 'receipt', 'charged', 'billing', 'overcharged', 'amount', 'estimate']):
        return 'trip_fare_dispute'
    elif any(word in text_lower for word in ['login', 'password', 'account', 'banned', 'deactivated', 'locked', 'sign in', 'reset', 'verify', 'verification', 'otp', 'phone number', 'email']):
        return 'account_access'
    elif any(word in text_lower for word in ['app', 'crash', 'bug', 'gps', 'map', 'loading', 'broken', 'not working', 'glitch', 'update', 'download', 'install', 'error', 'notification']):
        return 'app_functionality'
    elif any(word in text_lower for word in ['unhappy', 'disappointed', 'worst', 'terrible', 'bad', 'poor', 'unsatisfied', 'angry', 'frustrated', 'horrible', 'awful', 'rude', 'never again', 'scam', 'fraud', 'pathetic', 'joke']):
        return 'complaint'
    elif any(word in text_lower for word in ['refund', 'money back', 'return', 'give me back', 'credit back', 'reimburse']):
        return 'refund_request'
    elif any(word in text_lower for word in ['cancel my', 'cancel account', 'unsubscribe', 'subscription', 'delete account', 'stop subscription', 'cancellation fee']):
        return 'cancellation'
    else:
        return 'other'


def _create_synthetic_cases(brand_name: str, count: int) -> List[Dict[str, Any]]:
    """Create synthetic historical cases for demonstration."""
    import random

    synthetic_templates = [
        {
            'customer_message': "Where is my order #{}?",
            'brand_response': "I've checked your order #{} and it's currently {}",
            'resolution': "Provided tracking information and estimated delivery date",
            'intent': 'order_status'
        },
        {
            'customer_message': "Can I get a refund for this item?",
            'brand_response': "Refund processed! Should appear in 3-5 business days",
            'resolution': "Initiated refund process",
            'intent': 'refund_request'
        },
        {
            'customer_message': "I was charged twice for the same item",
            'brand_response': "I see the duplicate charge. Issuing refund for the extra amount",
            'resolution': "Corrected billing error and issued refund",
            'intent': 'payment_issue'
        },
        {
            'customer_message': "I can't log into my account",
            'brand_response': "Let me help you reset your password",
            'resolution': "Assisted with password recovery",
            'intent': 'account_login'
        },
        {
            'customer_message': "The website keeps crashing when I try to checkout",
            'brand_response': "Our technical team is aware of the issue and working on a fix",
            'resolution': "Provided workaround and escalated to technical team",
            'intent': 'technical_issue'
        },
        {
            'customer_message': "How do I cancel my subscription?",
            'brand_response': "To cancel your subscription, please visit your account settings",
            'resolution': "Provided cancellation instructions",
            'intent': 'cancellation'
        },
        {
            'customer_message': "I don't understand this charge on my bill",
            'brand_response': "Let me review that charge for you",
            'resolution': "Explained the charge and corrected if necessary",
            'intent': 'billing_problem'
        },
        {
            'customer_message': "What is your return policy?",
            'brand_response': "Our return policy allows returns within 30 days with original receipt",
            'resolution': "Provided return policy information",
            'intent': 'information_request'
        },
        {
            'customer_message': "I'm very unhappy with the service I received",
            'brand_response': "I'm sorry to hear that. Let me see how we can make this right",
            'resolution': "Apologized and offered compensatory action",
            'intent': 'complaint'
        },
        {
            'customer_message': "Thanks for your help!",
            'brand_response': "You're welcome! Is there anything else I can help you with today?",
            'resolution': "Ended conversation positively",
            'intent': 'other'
        }
    ]

    cases = []
    statuses = ['shipped', 'processing', 'out for delivery', 'delayed']

    for i in range(count):
        template = random.choice(synthetic_templates)

        # Fill in placeholders
        customer_msg = template['customer_message']
        brand_resp = template['brand_response']

        if '{}' in customer_msg:
            if '#' in customer_msg:
                customer_msg = customer_msg.format(random.randint(10000, 99999))
            else:
                # For templates like "Where is my order {}?"
                customer_msg = customer_msg.format(random.randint(10000, 99999))

        if '{}' in brand_resp:
            if '#' in brand_resp:
                # Replace both instances of {}
                order_num = random.randint(10000, 99999)
                status = random.choice(statuses)
                brand_resp = brand_resp.format(order_num, status)
            else:
                brand_resp = brand_resp.format(random.choice(['shipped', 'processing', 'delivered']))

        case = {
            'conversation_id': f'synthetic_{i}',
            'customer_message': customer_msg,
            'brand_response': brand_resp,
            'resolution': template['resolution'],
            'intent': template['intent'],
            'created_at': f'2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}'
        }

        cases.append(case)

    return cases


def main():
    """Main function to demonstrate retriever usage."""
    print("Loading historical cases...")
    cases = load_historical_cases()

    print(f"Loaded {len(cases)} historical cases")

    # Initialize and fit retriever
    print("\nTraining historical retriever...")
    retriever = HistoricalRetriever()
    retriever.fit(cases)

    # Test with some examples
    test_queries = [
        "Where is my order #12345?",
        "I want a refund for my purchase",
        "I was charged twice for the same item",
        "I can't log into my account",
        "The website keeps crashing",
        "How do I cancel my subscription?",
        "What is your return policy?",
        "I'm very unhappy with the service",
        "Thanks for your help!",
        "My payment didn't go through"
    ]

    print("\n=== Retrieval Examples ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, k=3)
        for i, result in enumerate(results, 1):
            print(f"  Result {i}:")
            print(f"    Similarity: {result['similarity']:.3f}")
            print(f"    Intent: {result.get('intent', 'unknown')}")
            print(f"    Customer: {result.get('customer_message', '')[:50]}...")
            if result.get('brand_response'):
                print(f"    Response: {result['brand_response'][:50]}...")
    print()


if __name__ == "__main__":
    main()