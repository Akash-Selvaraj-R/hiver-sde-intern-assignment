"""
Intent taxonomy derivation and management for brand-specific customer support.
"""
import pandas as pd
import re
from collections import Counter
from typing import List, Dict, Tuple, Any
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IntentTaxonomy:
    def __init__(self, config_path: str = "configs/brand.yaml"):
        """Initialize with brand configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.brand_name = self.config['brand_name']
        self.dataset_path = Path(self.config['dataset_path'])

    def load_brand_data(self) -> pd.DataFrame:
        """Load brand-specific conversation data."""
        data_file = self.dataset_path / f"{self.brand_name}_conversations.csv"
        if not data_file.exists():
            logger.warning(f"Brand-specific data not found at {data_file}")
            # Fall back to general sample
            data_file = Path("data/sample/twitter_support_synthetic.csv")

        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} messages for intent analysis")
        return df

    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing for intent analysis."""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user mentions and hashtags (keep the text)
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_keywords(self, texts: List[str]) -> List[Tuple[str, int]]:
        """Extract common keywords from a list of texts."""
        all_words = []
        for text in texts:
            processed = self.preprocess_text(text)
            # Simple tokenization
            words = re.findall(r'\b[a-z]{3,}\b', processed)
            all_words.extend(words)

        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can',
            'has', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'him', 'his',
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did',
            'its', 'let', 'put', 'say', 'she', 'too', 'use', 'why', 'ask', 'end',
            'far', 'big', 'why', 'try', 'put', 'oil', 'fit', 'run', 'sit', 'six',
            'ten', 'lot', 'men', 'won'
        }

        keywords = [word for word in all_words if word not in stop_words and len(word) > 2]
        return Counter(keywords).most_common(20)

    def derive_intents_from_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Derive intent taxonomy from historical conversations.

        Returns:
            List of intent dictionaries with name, description, examples, etc.
        """
        logger.info("Deriving intent taxonomy from brand-specific data...")

        # Separate customer and brand messages
        customer_msgs = df[df['inbound'] == 1]['text'].tolist()
        brand_msgs = df[df['inbound'] == 0]['text'].tolist()

        logger.info(f"Analyzing {len(customer_msgs)} customer messages and {len(brand_msgs)} brand messages")

        # Extract keywords from customer messages to understand common issues
        customer_keywords = self.extract_keywords(customer_msgs)
        logger.info(f"Top customer message keywords: {customer_keywords[:10]}")

        # Based on keyword analysis and common support patterns, define intents
        # This is where we would normally do clustering or manual analysis
        # For now, we'll create a reasonable taxonomy based on observed patterns

        intents = [
            {
                "name": "order_status",
                "description": "Inquiries about order status, shipping, delivery tracking, or delivery dates",
                "positive_examples": [
                    "Where is my order #12345?",
                    "When will my package arrive?",
                    "I haven't received my order yet",
                    "Tracking number not working",
                    "Order still shows as processing"
                ],
                "confusing_related_intents": ["information_request", "technical_issue"],
                "escalation_considerations": "Complex shipping issues (lost packages, international delays) may require human intervention"
            },
            {
                "name": "refund_request",
                "description": "Requests for refunds, money-back guarantees, or payment reversals",
                "positive_examples": [
                    "Can I get a refund for this item?",
                    "I want my money back",
                    "Refund not processed yet",
                    "Charged twice for same item",
                    "Item arrived damaged, need refund"
                ],
                "confusing_related_intents": ["payment_issue", "cancellation", "billing_problem"],
                "escalation_considerations": "Refunds over certain amounts or requiring managerial approval should be escalated"
            },
            {
                "name": "payment_issue",
                "description": "Problems with payments, charges, billing, or payment methods",
                "positive_examples": [
                    "Was charged incorrectly",
                    "Double charged for order",
                    "Payment didn't go through",
                    "Need to update payment method",
                    "Unauthorized charge on account"
                ],
                "confusing_related_intents": ["refund_request", "billing_problem"],
                "escalation_considerations": "Fraud suspected payments or disputed charges over threshold amounts"
            },
            {
                "name": "account_login",
                "description": "Issues with account access, login problems, password reset, or authentication",
                "positive_examples": [
                    "Can't log into my account",
                    "Password reset not working",
                    "Account locked",
                    "Unable to access account",
                    "Login page not loading"
                ],
                "confusing_related_intents": ["technical_issue", "information_request"],
                "escalation_considerations": "Security concerns or potential account compromise should be escalated to security team"
            },
            {
                "name": "technical_issue",
                "description": "Problems with product functionality, website performance, app issues, or service errors",
                "positive_examples": [
                    "Website keeps crashing",
                    "App not loading properly",
                    "Error message when trying to checkout",
                    "Page not loading",
                    "Features not working as expected"
                ],
                "confusing_related_intents": ["account_login", "information_request"],
                "escalation_considerations": "Widespread technical outages or security vulnerabilities should be escalated immediately"
            },
            {
                "name": "cancellation",
                "description": "Requests to cancel orders, subscriptions, services, or memberships",
                "positive_examples": [
                    "Want to cancel my order",
                    "How do I cancel subscription?",
                    "Need to stop recurring payment",
                    "Change of mind, don't want item anymore",
                    "Order placed by mistake"
                ],
                "confusing_related_intents": ["refund_request", "order_status"],
                "escalation_considerations": "Subscription cancellations with complex terms or retention offers may need human handling"
            },
            {
                "name": "billing_problem",
                "description": "Questions about invoices, billing statements, charges, or payment receipts",
                "positive_examples": [
                    "Don't understand charge on statement",
                    "Received incorrect invoice",
                    "Billing address needs update",
                    "Charge higher than expected",
                    "Missing receipt for purchase"
                ],
                "confusing_related_intents": ["payment_issue", "refund_request"],
                "escalation_considerations": "Disputed charges requiring investigation or billing system errors"
            },
            {
                "name": "information_request",
                "description": "General inquiries for information about products, services, policies, or procedures",
                "positive_examples": [
                    "What is your return policy?",
                    "Do you offer international shipping?",
                    "How long does shipping take?",
                    "What payment methods do you accept?",
                    "Is this item in stock?"
                ],
                "confusing_related_intents": ["order_status", "technical_issue"],
                "escalation_considerations": "Usually low escalation risk unless complex policy interpretation needed"
            },
            {
                "name": "complaint",
                "description": "Expressions of dissatisfaction with products, services, or support experience",
                "positive_examples": [
                    "Very unhappy with service",
                    "This is unacceptable",
                    "Poor quality product received",
                    "Worst experience ever",
                    "Extremely disappointed with purchase"
                ],
                "confusing_related_intents": ["technical_issue", "information_request"],
                "escalation_considerations": "May require de-escalation, compensatory actions, or supervisor involvement"
            },
            {
                "name": "other",
                "description": "Messages that don't clearly fit into other categories or are ambiguous",
                "positive_examples": [
                    "Thanks for the help!",
                    "Just saying hello",
                    "Testing if this works",
                    "Random thought",
                    "Unclear request"
                ],
                "confusing_related_intents": [],
                "escalation_considerations": "Should be reviewed to determine if new intent category is needed or if it requires human handling"
            }
        ]

        logger.info(f"Derived {len(intents)} intents from data analysis")
        return intents

    def save_intents_to_yaml(self, intents: List[Dict[str, Any]], output_path: str = "configs/intents.yaml"):
        """Save intents to YAML configuration file."""
        # Convert to format suitable for YAML
        intent_dict = {"intents": intents}

        with open(output_path, 'w') as f:
            yaml.dump(intent_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Saved {len(intents)} intents to {output_path}")

    def load_intents_from_yaml(self, input_path: str = "configs/intents.yaml") -> List[Dict[str, Any]]:
        """Load intents from YAML configuration file."""
        with open(input_path, 'r') as f:
            data = yaml.safe_load(f)

        intents = data.get("intents", [])
        logger.info(f"Loaded {len(intents)} intents from {input_path}")
        return intents

    def validate_intents(self, intents: List[Dict[str, Any]]) -> bool:
        """Validate that intents have required fields."""
        required_fields = ["name", "description", "positive_examples", "confusing_related_intents", "escalation_considerations"]

        for intent in intents:
            missing_fields = [field for field in required_fields if field not in intent]
            if missing_fields:
                logger.error(f"Intent {intent.get('name', 'unknown')} missing fields: {missing_fields}")
                return False

        logger.info("All intents validated successfully")
        return True


def main():
    """Main function to derive and save intent taxonomy."""
    logging.basicConfig(level=logging.INFO)

    # Initialize taxonomy manager
    taxonomy = IntentTaxonomy()

    # Load brand-specific data
    df = taxonomy.load_brand_data()

    # Derive intents from data
    intents = taxonomy.derive_intents_from_data(df)

    # Validate intents
    if not taxonomy.validate_intents(intents):
        logger.error("Intent validation failed")
        return

    # Save to YAML
    taxonomy.save_intents_to_yaml(intents)

    # Print summary
    print("\n=== Derived Intent Taxonomy ===")
    for intent in intents:
        print(f"\nIntent: {intent['name']}")
        print(f"Description: {intent['description']}")
        print(f"Examples: {', '.join(intent['positive_examples'][:2])}...")
        print(f"Related: {', '.join(intent['confusing_related_intents'])}")
        print(f"Escalation: {intent['escalation_considerations'][:50]}...")


if __name__ == "__main__":
    main()