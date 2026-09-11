"""
Main pipeline that integrates all components for the customer support agent.
"""
import pandas as pd
import numpy as np
import yaml
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import asdict

# Import our components
try:
    # Try relative imports (when used as package)
    from .classifier import IntentClassifier
    from .retrieval import HistoricalRetriever
    from .generator import ResponseGenerator, GenerationResult
    from .escalation import EscalationPolicy
except ImportError:
    # Fall back to absolute imports (when run directly or from elsewhere)
    from classifier import IntentClassifier
    from retrieval import HistoricalRetriever
    from generator import ResponseGenerator, GenerationResult
    from escalation import EscalationPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupportAgentPipeline:
    """
    Main pipeline integrating intent classification, historical retrieval,
    response generation, and escalation policy.
    """

    def __init__(self, config_path: str = "configs/brand.yaml"):
        """Initialize the pipeline with all components."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.brand_name = self.config['brand_name']

        # Initialize components
        logger.info("Initializing pipeline components...")
        self.classifier = IntentClassifier(config_path)
        self.retriever = HistoricalRetriever(config_path)
        self.generator = ResponseGenerator(config_path)
        self.escalation_policy = EscalationPolicy(config_path)

        # Track if components are fitted
        self.is_fitted = False

        logger.info("Pipeline initialized")

    def fit(self, training_data: Optional[Tuple[List[str], List[str]]] = None,
            historical_cases: Optional[List[Dict[str, Any]]] = None) -> 'SupportAgentPipeline':
        """
        Fit all pipeline components.

        Args:
            training_data: Tuple of (texts, labels) for classifier training.
                          If None, will load from brand-specific data.
            historical_cases: List of historical cases for retriever.
                            If None, will load from brand-specific data.

        Returns:
            Self for method chaining
        """
        logger.info("Fitting pipeline components...")

        # Load training data if not provided
        if training_data is None:
            from .classifier import load_training_data
            training_data = load_training_data()

        train_texts, train_labels = training_data

        # Load historical cases if not provided
        if historical_cases is None:
            from .retrieval import load_historical_cases
            historical_cases = load_historical_cases()

        # Fit components
        logger.info("Fitting intent classifier...")
        self.classifier.fit(train_texts, train_labels)

        logger.info("Fitting historical retriever...")
        self.retriever.fit(historical_cases)

        self.is_fitted = True
        logger.info("Pipeline fitting completed")

        return self

    def predict(self, message: str) -> Dict[str, Any]:
        """
        Process a customer message through the complete pipeline.

        Args:
            message: The customer message to process

        Returns:
            Dictionary with all pipeline outputs
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before making predictions")

        logger.info(f"Processing message: '{message[:100]}{'...' if len(message) > 100 else ''}'")

        # Step 1: Intent Classification
        logger.debug("Step 1: Intent classification")
        intent_result = self.classifier.predict_single(message)
        intent_name = intent_result['intent']
        intent_confidence = intent_result['confidence']

        # Step 2: Historical Case Retrieval
        logger.debug("Step 2: Historical retrieval")
        retrieved_cases = self.retriever.retrieve(message, k=self.config['top_k_retrieval'])

        # Optionally, retrieve more cases of the predicted intent for better grounding
        intent_specific_cases = self.retriever.retrieve_by_intent(
            message, intent_name, k=self.config['top_k_retrieval']
        )
        # Combine and deduplicate (prioritize intent-specific cases)
        all_case_ids = set()
        combined_cases = []

        # Add intent-specific cases first
        for case in intent_specific_cases:
            case_id = str(case.get('conversation_id', ''))
            if case_id not in all_case_ids:
                all_case_ids.add(case_id)
                combined_cases.append(case)

        # Add other retrieved cases
        for case in retrieved_cases:
            case_id = str(case.get('conversation_id', ''))
            if case_id not in all_case_ids:
                all_case_ids.add(case_id)
                combined_cases.append(case)

        retrieved_cases = combined_cases[:self.config['top_k_retrieval'] * 2]  # Limit total

        # Step 3: Response Generation
        logger.debug("Step 3: Response generation")
        generation_result = self.generator.generate(
            query=message,
            intent=intent_name,
            confidence=intent_confidence,
            retrieved_cases=retrieved_cases
        )

        # Step 4: Escalation Decision
        logger.debug("Step 4: Escalation decision")
        escalation_result = self.escalation_policy.should_escalate(
            intent_result=intent_result,
            retrieved_cases=retrieved_cases,
            generated_response=asdict(generation_result) if hasattr(generation_result, '__dict__') else None
        )

        # Compile final result
        result = {
            'intent': {
                'name': intent_name,
                'confidence': intent_confidence
            },
            'retrieved_cases': [
                {
                    'conversation_id': str(case.get('conversation_id', '')),
                    'similarity': float(case.get('similarity', 0.0)),
                    'intent': str(case.get('intent', '')),
                    'customer_message': str(case.get('customer_message', ''))[:100] + ('...' if len(str(case.get('customer_message', ''))) > 100 else ''),
                    'brand_response': str(case.get('brand_response', ''))[:100] + ('...' if len(str(case.get('brand_response', ''))) > 100 else ''),
                    'resolution': str(case.get('resolution', ''))[:100] + ('...' if len(str(case.get('resolution', ''))) > 100 else '')
                }
                for case in retrieved_cases
            ],
            'reply': self._clean_reply(generation_result.reply),
            'should_escalate': escalation_result['should_escalate'],
            'escalation_reason': escalation_result['reason'],
            'metadata': {
                'grounding_confidence': generation_result.grounding_confidence,
                'model_used': generation_result.model_used,
                'evidence_ids': generation_result.evidence_ids,
                'escalation_score': escalation_result.get('escalation_score', 0.0),
                'factor_scores': escalation_result.get('factor_scores', {})
            }
        }

        logger.info(f"Pipeline completed. Intent: {intent_name} ({intent_confidence:.3f}), "
                   f"Escalate: {escalation_result['should_escalate']}")

        return result

    def predict_batch(self, messages: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of customer messages.

        Args:
            messages: List of customer messages to process

        Returns:
            List of pipeline output dictionaries
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before making predictions")

        logger.info(f"Processing batch of {len(messages)} messages")
        results = []

        for i, message in enumerate(messages):
            logger.debug(f"Processing message {i+1}/{len(messages)}")
            result = self.predict(message)
            results.append(result)

        return results

    def get_component_status(self) -> Dict[str, bool]:
        """Get the fitting status of each component."""
        return {
            'classifier_fitted': self.classifier.is_fitted,
            'retriever_fitted': self.retriever.is_fitted,
            'pipeline_fitted': self.is_fitted
        }

    def _clean_reply(self, reply: str) -> str:
        """Clean up reply text: strip leaked conversation IDs and fix broken fragments."""
        if not reply:
            return reply
        # Remove leaked Twitter conversation IDs like @275897 or @115873 or @378450
        cleaned = re.sub(r'@\d{3,}', '', reply).strip()
        # Remove truncated URLs
        cleaned = re.sub(r'https?://\S+', '', cleaned).strip()
        # Fix broken sentences from URL stripping
        cleaned = re.sub(r'\bvia\s+and\b', 'and', cleaned)
        cleaned = re.sub(r'\bat\s+and\b', 'and', cleaned)
        cleaned = re.sub(r'\bvia\s+\.', '.', cleaned)
        cleaned = re.sub(r'\bat\s+\.', '.', cleaned)
        cleaned = re.sub(r'\bhere[;:]\s+and\b', 'and', cleaned)
        # Remove dangling prepositions/conjunctions after URL removal
        cleaned = re.sub(r'\bvia\s*$', '', cleaned).strip()
        cleaned = re.sub(r'\bat\s+(?=so |our |the )', '', cleaned).strip()
        cleaned = re.sub(r'\bhere[;:]\s+(?=so |our |the )', '', cleaned).strip()
        cleaned = re.sub(r'\bhere[;:]\s*$', '', cleaned).strip()
        cleaned = re.sub(r'\bhere\s*$', '', cleaned).strip()
        # Remove dangling name leaks like "Hi Brian," anywhere in the text
        cleaned = re.sub(r',?\s*Hi \w+,\s*', ' ', cleaned).strip()
        # Fix lowercase after period (e.g. ". requesting" -> ". Requesting")
        cleaned = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), cleaned)
        # Remove trailing punctuation artifacts
        cleaned = re.sub(r'[,\s]+$', '', cleaned).strip()
        # Collapse multiple spaces
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
        return cleaned


def main():
    """Main function to demonstrate the complete pipeline."""
    print("Initializing support agent pipeline...")
    pipeline = SupportAgentPipeline()

    # Load and fit with brand-specific data
    print("\nLoading training data and historical cases...")
    from .classifier import load_training_data
    from .retrieval import load_historical_cases

    train_texts, train_labels = load_training_data()
    historical_cases = load_historical_cases()

    print(f"Loaded {len(train_texts)} training examples")
    print(f"Loaded {len(historical_cases)} historical cases")

    # Fit the pipeline
    print("\nFitting pipeline components...")
    pipeline.fit(
        training_data=(train_texts, train_labels),
        historical_cases=historical_cases
    )

    # Test with example messages
    test_messages = [
        "Where is my order #12345?",
        "I want a refund for my purchase",
        "I was charged twice for the same item",
        "I can't log into my account",
        "The website keeps crashing when I try to checkout",
        "How do I cancel my subscription?",
        "What is your return policy?",
        "I'm very unhappy with the service I received",
        "Thanks for your help!",
        "My payment didn't go through",
        "I think there might be fraudulent activity on my account",
        "The item arrived damaged and I need a replacement"
    ]

    print("\n=== Pipeline Results ===")
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. Message: {message}")
        result = pipeline.predict(message)

        print(f"   Intent: {result['intent']['name']} "
              f"(confidence: {result['intent']['confidence']:.3f})")
        print(f"   Reply: {result['reply']}")
        print(f"   Retrieved {len(result['retrieved_cases'])} cases")
        if result['retrieved_cases']:
            top_case = result['retrieved_cases'][0]
            print(f"   Top case similarity: {top_case['similarity']:.3f}")
            print(f"   Top case intent: {top_case['intent']}")
        print(f"   Should escalate: {result['should_escalate']}")
        if result['escalation_reason']:
            print(f"   Escalation reason: {result['escalation_reason']}")
        print(f"   Grounding confidence: {result['metadata']['grounding_confidence']:.3f}")

    # Show component status
    print("\n=== Component Status ===")
    status = pipeline.get_component_status()
    for component, fitted in status.items():
        print(f"{component}: {'✓' if fitted else '✗'}")


if __name__ == "__main__":
    main()


def cli_main():
    """Command-line interface for the support agent."""
    import argparse
    import sys
    import json

    parser = argparse.ArgumentParser(description="Hiver Customer Support Agent CLI")
    parser.add_argument("--message", "-m", type=str, help="Customer message to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    if args.interactive:
        # Interactive mode
        print("Hiver Customer Support Agent - Interactive Mode")
        print("Type 'quit' or 'exit' to stop")
        print("-" * 50)

        # Initialize pipeline
        pipeline = SupportAgentPipeline()
        from .classifier import load_training_data
        from .retrieval import load_historical_cases

        print("Loading data and fitting pipeline...")
        train_texts, train_labels = load_training_data()
        historical_cases = load_historical_cases()
        pipeline.fit(
            training_data=(train_texts, train_labels),
            historical_cases=historical_cases
        )
        print("Ready!\n")

        while True:
            try:
                message = input("\nCustomer message: ").strip()
                if message.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not message:
                    continue

                result = pipeline.predict(message)
                print(f"\nIntent: {result['intent']['name']} ({result['intent']['confidence']:.3f})")
                print(f"Reply: {result['reply']}")
                print(f"Retrieved {len(result['retrieved_cases'])} historical cases")
                if result['should_escalate']:
                    print(f"⚠️  ESCALATION RECOMMENDED: {result['escalation_reason']}")
                else:
                    print("✅ Safe to auto-handle")
                print("-" * 50)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    elif args.message:
        # Single message mode
        pipeline = SupportAgentPipeline()
        from .classifier import load_training_data
        from .retrieval import load_historical_cases

        print("Loading data and fitting pipeline...")
        train_texts, train_labels = load_training_data()
        historical_cases = load_historical_cases()
        pipeline.fit(
            training_data=(train_texts, train_labels),
            historical_cases=historical_cases
        )

        result = pipeline.predict(args.message)
        print(json.dumps(result, indent=2))

    else:
        # No arguments - show help
        parser.print_help()


