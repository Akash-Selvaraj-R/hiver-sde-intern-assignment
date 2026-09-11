"""
Comprehensive tests for the customer support agent.
"""
import unittest
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))


class TestPreprocessing(unittest.TestCase):
    """Tests for text preprocessing."""

    def test_clean_text_basic(self):
        from preprocessing import clean_text
        result = clean_text("Hello World!")
        self.assertEqual(result, "hello world!")

    def test_clean_text_urls(self):
        from preprocessing import clean_text
        result = clean_text("Visit https://example.com for info")
        self.assertNotIn("https://", result)

    def test_clean_text_mentions(self):
        from preprocessing import clean_text
        result = clean_text("Hello @support team")
        self.assertNotIn("@support", result)

    def test_clean_text_empty(self):
        from preprocessing import clean_text
        result = clean_text("")
        self.assertEqual(result, "")

    def test_clean_text_none(self):
        from preprocessing import clean_text
        result = clean_text(None)
        self.assertEqual(result, "")


class TestTaxonomy(unittest.TestCase):
    """Tests for intent taxonomy loading."""

    def test_load_intents(self):
        import yaml
        with open("configs/intents.yaml", 'r') as f:
            data = yaml.safe_load(f)
        intents = data.get("intents", [])
        self.assertGreater(len(intents), 0)
        self.assertLessEqual(len(intents), 15)

    def test_intent_structure(self):
        import yaml
        with open("configs/intents.yaml", 'r') as f:
            data = yaml.safe_load(f)
        for intent in data["intents"]:
            self.assertIn("name", intent)
            self.assertIn("description", intent)
            self.assertIn("positive_examples", intent)
            self.assertIsInstance(intent["positive_examples"], list)


class TestClassifier(unittest.TestCase):
    """Tests for intent classifier."""

    def setUp(self):
        from classifier import IntentClassifier
        self.classifier = IntentClassifier()
        self.train_texts = [
            "Where is my order?",
            "I want a refund",
            "I can't log in",
            "The website is down",
            "How do I cancel?",
            "What's your return policy?"
        ]
        self.train_labels = [
            "order_status",
            "refund_request",
            "account_login",
            "technical_issue",
            "cancellation",
            "information_request"
        ]

    def test_initialization(self):
        self.assertFalse(self.classifier.is_fitted)

    def test_fit(self):
        self.classifier.fit(self.train_texts, self.train_labels)
        self.assertTrue(self.classifier.is_fitted)

    def test_predict(self):
        self.classifier.fit(self.train_texts, self.train_labels)
        results = self.classifier.predict_intent(["Where is my order?"])
        self.assertEqual(len(results), 1)
        self.assertIn("intent", results[0])
        self.assertIn("confidence", results[0])

    def test_predict_single(self):
        self.classifier.fit(self.train_texts, self.train_labels)
        result = self.classifier.predict_single("I want a refund")
        self.assertEqual(result["intent"], "refund_request")

    def test_empty_input(self):
        self.classifier.fit(self.train_texts, self.train_labels)
        results = self.classifier.predict_intent([])
        self.assertEqual(len(results), 0)


class TestRetrieval(unittest.TestCase):
    """Tests for historical case retrieval."""

    def test_load_cases(self):
        from retrieval import load_historical_cases
        cases = load_historical_cases()
        self.assertGreater(len(cases), 0)

    def test_case_structure(self):
        from retrieval import load_historical_cases
        cases = load_historical_cases()
        for case in cases[:5]:
            self.assertIn("customer_message", case)
            self.assertIn("intent", case)


class TestEscalation(unittest.TestCase):
    """Tests for escalation policy."""

    def setUp(self):
        from escalation import EscalationPolicy
        self.policy = EscalationPolicy()

    def test_no_escalation_high_confidence(self):
        result = self.policy.should_escalate(
            intent_result={"intent": "order_status", "confidence": 0.95},
            retrieved_cases=[
                {"similarity": 0.9, "intent": "order_status", "resolution": "Provided tracking"}
            ]
        )
        self.assertFalse(result["should_escalate"])

    def test_escalation_low_confidence(self):
        result = self.policy.should_escalate(
            intent_result={"intent": "order_status", "confidence": 0.3},
            retrieved_cases=[]
        )
        self.assertTrue(result["should_escalate"])

    def test_escalation_has_reason(self):
        result = self.policy.should_escalate(
            intent_result={"intent": "other", "confidence": 0.4},
            retrieved_cases=[]
        )
        if result["should_escalate"]:
            self.assertIsNotNone(result["reason"])


class TestPipeline(unittest.TestCase):
    """Tests for the full pipeline."""

    def test_pipeline_initialization(self):
        from pipeline import SupportAgentPipeline
        pipeline = SupportAgentPipeline()
        self.assertFalse(pipeline.is_fitted)

    def test_pipeline_fit_and_predict(self):
        from pipeline import SupportAgentPipeline
        from classifier import load_training_data
        from retrieval import load_historical_cases

        train_texts, train_labels = load_training_data()
        historical_cases = load_historical_cases()

        pipeline = SupportAgentPipeline()
        pipeline.fit(
            training_data=(train_texts[:50], train_labels[:50]),
            historical_cases=historical_cases[:50]
        )

        result = pipeline.predict("Where is my order?")
        self.assertIn("intent", result)
        self.assertIn("reply", result)
        self.assertIn("should_escalate", result)


class TestDataLeakage(unittest.TestCase):
    """Tests to ensure no data leakage between splits."""

    def test_no_leakage(self):
        """Verify golden set examples don't appear in training set."""
        import pandas as pd

        train_path = Path("data/processed/train.csv")
        golden_path = Path("evaluation/golden_set.csv")

        if not train_path.exists() or not golden_path.exists():
            self.skipTest("Data splits not created yet")

        train_df = pd.read_csv(train_path)
        golden_df = pd.read_csv(golden_path)

        train_texts = set(train_df['text'].tolist())
        golden_texts = set(golden_df['customer_message'].tolist())

        overlap = train_texts & golden_texts
        self.assertEqual(len(overlap), 0, f"Found {len(overlap)} leaked examples")


class TestGoldenSet(unittest.TestCase):
    """Tests for golden set properties."""

    def test_golden_set_size(self):
        import pandas as pd
        golden_df = pd.read_csv("evaluation/golden_set.csv")
        # Golden set size depends on available unique messages in synthetic data
        self.assertGreaterEqual(len(golden_df), 10)
        self.assertLessEqual(len(golden_df), 250)

    def test_golden_set_columns(self):
        import pandas as pd
        golden_df = pd.read_csv("evaluation/golden_set.csv")
        required_cols = ['example_id', 'customer_message', 'intent', 'should_escalate']
        for col in required_cols:
            self.assertIn(col, golden_df.columns)

    def test_golden_set_intent_distribution(self):
        import pandas as pd
        golden_df = pd.read_csv("evaluation/golden_set.csv")
        intent_counts = golden_df['intent'].value_counts()
        # Each intent should have at least some examples
        self.assertGreater(len(intent_counts), 3)


class TestBaselines(unittest.TestCase):
    """Tests for baseline implementations."""

    def test_trivial_baseline(self):
        from baselines import TrivialBaseline
        baseline = TrivialBaseline()
        baseline.fit(["a", "b", "c"], ["x", "y", "x"])
        predictions = baseline.predict_intent(["a", "b"])
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(p == "x" for p in predictions))

    def test_tfidf_baseline(self):
        from baselines import TfIdfBaseline
        baseline = TfIdfBaseline()
        texts = ["where is my order", "i want a refund", "website is down"]
        labels = ["order", "refund", "technical"]
        baseline.fit(texts, labels)
        predictions = baseline.predict_intent(["where is my order"])
        self.assertEqual(len(predictions), 1)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_empty_message(self):
        from classifier import IntentClassifier
        classifier = IntentClassifier()
        # Need at least 2 classes for LogisticRegression
        classifier.fit(["test message", "another message"], ["other", "order_status"])
        result = classifier.predict_single("")
        self.assertIn("intent", result)

    def test_long_message(self):
        from classifier import IntentClassifier
        classifier = IntentClassifier()
        classifier.fit(["test message", "another message"], ["other", "order_status"])
        long_msg = "word " * 1000
        result = classifier.predict_single(long_msg)
        self.assertIn("intent", result)

    def test_special_characters(self):
        from classifier import IntentClassifier
        classifier = IntentClassifier()
        classifier.fit(["test message", "another message"], ["other", "order_status"])
        result = classifier.predict_single("!!!@@@###$$$%%%")
        self.assertIn("intent", result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
