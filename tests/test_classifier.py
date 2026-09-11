"""
Unit tests for the intent classifier.
"""
import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from classifier import IntentClassifier


class TestIntentClassifier(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = IntentClassifier()

        # Simple training data for testing
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

    def test_classifier_initialization(self):
        """Test that classifier initializes correctly."""
        self.assertIsInstance(self.classifier, IntentClassifier)
        self.assertFalse(self.classifier.is_fitted)

    def test_classifier_fit(self):
        """Test that classifier can be fitted."""
        self.classifier.fit(self.train_texts, self.train_labels)
        self.assertTrue(self.classifier.is_fitted)
        self.assertIsNotNone(self.classifier.classes_)

    def test_classifier_predict(self):
        """Test that classifier makes predictions."""
        self.classifier.fit(self.train_texts, self.train_labels)

        test_texts = ["Where is my order #12345?", "I need a refund"]
        predictions = self.classifier.predict_intent(test_texts)

        self.assertEqual(len(predictions), 2)
        for pred in predictions:
            self.assertIn('intent', pred)
            self.assertIn('confidence', pred)
            self.assertIsInstance(pred['intent'], str)
            self.assertIsInstance(pred['confidence'], float)
            self.assertGreaterEqual(pred['confidence'], 0.0)
            self.assertLessEqual(pred['confidence'], 1.0)

    def test_classifier_predict_single(self):
        """Test single prediction."""
        self.classifier.fit(self.train_texts, self.train_labels)

        result = self.classifier.predict_single("Where is my order?")
        self.assertIn('intent', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['intent'], str)
        self.assertIsInstance(result['confidence'], float)


if __name__ == '__main__':
    unittest.main()