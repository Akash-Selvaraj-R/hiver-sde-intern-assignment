"""
Escalation policy module for determining when to escalate to human agents.
"""
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EscalationPolicy:
    """
    Implements a conservative escalation policy based on multiple factors.
    """

    def __init__(self, config_path: str = "configs/brand.yaml"):
        """Initialize escalation policy with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.brand_name = self.config['brand_name']
        self.confidence_threshold = self.config['confidence_threshold']
        self.escalation_threshold = self.config['escalation_threshold']

        # Load intent-specific escalation considerations
        self.intent_escalation_rules = self._load_intent_escalation_rules()

        # Define escalation factors and their weights
        self.factors = {
            'low_intent_confidence': 0.25,
            'low_evidence_quality': 0.20,
            'conflicting_historical_resolutions': 0.15,
            'unsupported_requested_action': 0.15,
            'high_risk_content': 0.15,
            'ambiguous_message': 0.10
        }

    def _load_intent_escalation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load intent-specific escalation considerations from intents.yaml."""
        try:
            with open("configs/intents.yaml", 'r') as f:
                data = yaml.safe_load(f)
                rules = {}
                for intent in data.get("intents", []):
                    rules[intent['name']] = {
                        'considerations': intent.get('escalation_considerations', ''),
                        'related_intents': intent.get('confusing_related_intents', [])
                    }
                return rules
        except FileNotFoundError:
            logger.warning("intents.yaml not found, using default escalation rules")
            return self._get_default_escalation_rules()

    def _get_default_escalation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Default escalation rules if intents.yaml is not available."""
        return {
            'ride_status': {
                'considerations': 'Safety concerns, driver behavior incidents, lost items, repeated cancellations, and rides with no brand response should be escalated',
                'related_intents': ['trip_fare_dispute', 'app_functionality']
            },
            'trip_fare_dispute': {
                'considerations': 'High-value fare disputes, suspected fraud, duplicate charges, and charges without ride records should be escalated',
                'related_intents': ['ride_status', 'refund_request']
            },
            'account_access': {
                'considerations': 'Account security concerns, deactivation appeals, and identity verification failures should be escalated to specialized teams',
                'related_intents': ['app_functionality']
            },
            'app_functionality': {
                'considerations': 'Widespread app outages, security vulnerabilities, and issues affecting driver earnings should be escalated to engineering teams',
                'related_intents': ['account_access', 'ride_status']
            },
            'complaint': {
                'considerations': 'Severe customer anger, safety-related complaints, threats of legal action, and public reputation risk should be escalated for de-escalation',
                'related_intents': ['ride_status', 'trip_fare_dispute']
            },
            'refund_request': {
                'considerations': 'Refunds over threshold amounts, repeat refund requests, and refund denials should be escalated to supervisors',
                'related_intents': ['trip_fare_dispute']
            },
            'cancellation': {
                'considerations': 'Cancellation fee disputes, account deletion requests with active subscriptions, and retention-sensitive cases should be escalated',
                'related_intents': ['ride_status', 'account_access']
            },
            'other': {
                'considerations': 'Ambiguous messages should be reviewed to determine if a new intent category is needed or if human handling is appropriate',
                'related_intents': []
            }
        }

    def should_escalate(self,
                       intent_result: Dict[str, Any],
                       retrieved_cases: List[Dict[str, Any]],
                       generated_response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Determine if a message should be escalated based on multiple factors.

        Args:
            intent_result: Dictionary with 'intent' and 'confidence' from classifier
            retrieved_cases: List of retrieved historical cases
            generated_response: Optional dict with response generation results

        Returns:
            Dictionary with 'should_escalate' (bool) and 'reason' (str)
        """
        intent_name = intent_result.get('intent', 'other')
        intent_confidence = intent_result.get('confidence', 0.0)

        logger.debug(f"Evaluating escalation for intent '{intent_name}' with confidence {intent_confidence:.3f}")

        # Calculate escalation scores for each factor
        factor_scores = {}

        # 1. Low intent confidence
        factor_scores['low_intent_confidence'] = max(0.0, 1.0 - (intent_confidence / self.confidence_threshold))

        # 2. Low evidence quality
        factor_scores['low_evidence_quality'] = self._calculate_evidence_quality_score(retrieved_cases, intent_name)

        # 3. Conflicting historical resolutions
        factor_scores['conflicting_historical_resolutions'] = self._calculate_conflict_score(retrieved_cases)

        # 4. Unsupported requested action (simplified - would need action extraction in reality)
        factor_scores['unsupported_requested_action'] = self._calculate_unsupported_action_score(
            intent_name, retrieved_cases
        )

        # 5. High risk content
        factor_scores['high_risk_content'] = self._calculate_high_risk_score(intent_name, retrieved_cases)

        # 6. Ambiguous message
        factor_scores['ambiguous_message'] = self._calculate_ambiguity_score(intent_name, retrieved_cases)

        # Apply intent-specific adjustments
        factor_scores = self._apply_intent_specific_rules(factor_scores, intent_name, retrieved_cases)

        # Calculate weighted escalation score
        weighted_score = sum(
            factor_scores[factor] * weight
            for factor, weight in self.factors.items()
            if factor in factor_scores
        )

        # Determine if escalation is warranted
        should_escalate = weighted_score >= self.escalation_threshold

        # Generate reason
        reason = self._generate_escalation_reason(factor_scores, intent_name, weighted_score)

        result = {
            'should_escalate': bool(should_escalate),
            'reason': reason if should_escalate else None,
            'escalation_score': float(weighted_score),
            'factor_scores': {k: float(v) for k, v in factor_scores.items()}
        }

        logger.debug(f"Escalation decision: {result}")
        return result

    def _calculate_evidence_quality_score(self, retrieved_cases: List[Dict[str, Any]], intent: str) -> float:
        """Calculate score for low evidence quality (higher = worse quality)."""
        if not retrieved_cases:
            return 1.0  # Maximum score for no evidence

        # Check similarity scores
        similarities = [case.get('similarity', 0.0) for case in retrieved_cases]
        max_similarity = max(similarities) if similarities else 0.0
        avg_similarity = np.mean(similarities) if similarities else 0.0

        # Check intent alignment in retrieved cases
        intent_matches = [1.0 if case.get('intent') == intent else 0.0
                         for case in retrieved_cases]
        intent_alignment = np.mean(intent_matches) if intent_matches else 0.0

        # Evidence quality is poor if similarity is low or intent alignment is poor
        quality_score = 1.0 - ((max_similarity * 0.6) + (intent_alignment * 0.4))
        return min(1.0, max(0.0, quality_score))

    def _calculate_conflict_score(self, retrieved_cases: List[Dict[str, Any]]) -> float:
        """Calculate score for conflicting historical resolutions."""
        if len(retrieved_cases) < 2:
            return 0.0  # Can't have conflict with less than 2 cases

        # Extract resolutions
        resolutions = [case.get('resolution', '').lower().strip()
                      for case in retrieved_cases
                      if case.get('resolution')]

        if len(resolutions) < 2:
            return 0.0

        # Simple conflict detection: check if resolutions are significantly different
        # In reality, would use semantic similarity or manual labeling
        unique_resolutions = set(resolutions)
        if len(unique_resolutions) > 1:
            # Basic heuristic: if we have very different resolutions, there might be conflict
            # For now, return a moderate score if there's more than one unique resolution
            return min(0.8, len(unique_resolutions) / len(resolutions))

        return 0.0

    def _calculate_unsupported_action_score(self, intent: str, retrieved_cases: List[Dict[str, Any]]) -> float:
        """Calculate score for unsupported requested actions."""
        # This would require intent parsing and action extraction
        # For now, use a simplified heuristic based on intent and content

        high_risk_intents = ['refund_request', 'payment_issue', 'account_login']
        if intent in high_risk_intents:
            # Check if any retrieved cases suggest limitations
            limitation_indicators = [
                'managerial approval', 'supervisor', 'restriction', 'limit',
                'policy prevents', 'cannot', 'not allowed'
            ]

            limitation_count = 0
            for case in retrieved_cases:
                resolution = str(case.get('resolution', '')).lower()
                if any(indicator in resolution for indicator in limitation_indicators):
                    limitation_count += 1

            # If many cases show limitations, requested action might not be supported
            limitation_ratio = limitation_count / len(retrieved_cases) if retrieved_cases else 0
            return min(0.8, limitation_ratio * 2)  # Scale up the score

        return 0.2  # Low base score for other intents

    def _calculate_high_risk_score(self, intent: str, retrieved_cases: List[Dict[str, Any]]) -> float:
        """Calculate score for high-risk content that should be escalated."""
        high_risk_indicators = [
            'fraud', 'unauthorized', 'stolen', 'hacked', 'compromised',
            'lawsuit', 'legal', 'attorney', 'sue', 'court',
            'threat', 'abuse', 'harassment', 'violence'
        ]

        # Check if intent itself is high-risk
        high_risk_intents = ['account_login', 'payment_issue', 'refund_request']
        intent_risk = 0.5 if intent in high_risk_intents else 0.0

        # Check retrieved cases for high-risk content
        case_risk = 0.0
        if retrieved_cases:
            risk_count = 0
            total_cases = len(retrieved_cases)

            for case in retrieved_cases:
                # Check customer message, brand response, and resolution
                text_fields = [
                    str(case.get('customer_message', '')),
                    str(case.get('brand_response', '')),
                    str(case.get('resolution', ''))
                ]
                combined_text = ' '.join(text_fields).lower()

                if any(indicator in combined_text for indicator in high_risk_indicators):
                    risk_count += 1

            case_risk = risk_count / total_cases if total_cases > 0 else 0.0

        # Combine intent risk and case risk
        return min(1.0, intent_risk + (case_risk * 0.5))

    def _calculate_ambiguity_score(self, intent: str, retrieved_cases: List[Dict[str, Any]]) -> float:
        """Calculate score for message ambiguity."""
        # Ambiguity indicators
        ambiguity_indicators = [
            'maybe', 'perhaps', 'i think', 'i guess', 'possibly',
            'not sure', 'unclear', 'confused', 'help me understand',
            'what do you mean', 'can you explain'
        ]

        # Very short messages are often ambiguous
        length_penalty = 0.0

        # Check if intent is ambiguous (the 'other' intent or high confusion)
        intent_ambiguity = 0.0
        if intent == 'other':
            intent_ambiguity = 0.6
        elif intent in self.intent_escalation_rules:
            related_count = len(self.intent_escalation_rules[intent].get('related_intents', []))
            intent_ambiguity = min(0.5, related_count * 0.1)  # More related intents = more ambiguity

        # For retrieved cases, check if they show mixed intents (indicating ambiguity in query)
        case_ambiguity = 0.0
        if retrieved_cases:
            intents_in_cases = [case.get('intent', '') for case in retrieved_cases]
            unique_intents = set(intents_in_cases)
            if len(unique_intents) > 1:
                # Cases have different intents, suggesting the query is ambiguous
                case_ambiguity = min(0.8, (len(unique_intents) - 1) * 0.3)

        return min(1.0, length_penalty + intent_ambiguity + case_ambiguity)

    def _apply_intent_specific_rules(self, factor_scores: Dict[str, float],
                                   intent: str, retrieved_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """Apply intent-specific rules to adjust factor scores."""
        if intent not in self.intent_escalation_rules:
            return factor_scores

        considerations = self.intent_escalation_rules[intent].get('considerations', '').lower()

        # Adjust scores based on specific considerations
        if 'refund' in considerations and 'amount' in considerations:
            # For refund requests, increase score if we suspect high value
            if intent == 'refund_request':
                # Look for high value indicators in cases or query (simplified)
                high_value_indicators = ['expensive', 'over $', 'above $', 'thousand', 'significant']
                # This would normally check the actual query, but we'll use a placeholder
                factor_scores['high_risk_content'] = min(1.0, factor_scores.get('high_risk_content', 0.0) + 0.3)

        if 'security' in considerations:
            if intent == 'account_login':
                factor_scores['high_risk_content'] = min(1.0, factor_scores.get('high_risk_content', 0.0) + 0.4)

        if 'widespread' in considerations or 'outage' in considerations:
            if intent == 'technical_issue':
                # Would check for indicators of widespread issues
                factor_scores['conflicting_historical_resolutions'] = min(
                    1.0, factor_scores.get('conflicting_historical_resolutions', 0.0) + 0.3)

        if 'complex' in considerations:
            if intent in ['cancellation', 'refund_request']:
                factor_scores['unsupported_requested_action'] = min(
                    1.0, factor_scores.get('unsupported_requested_action', 0.0) + 0.2)

        if 'de-escalation' in considerations or 'compensatory' in considerations:
            if intent == 'complaint':
                factor_scores['high_risk_content'] = min(1.0, factor_scores.get('high_risk_content', 0.0) + 0.3)

        return factor_scores

    def _generate_escalation_reason(self, factor_scores: Dict[str, float],
                                  intent: str, weighted_score: float) -> str:
        """Generate a human-readable reason for the escalation decision."""
        # Get top contributing factors
        sorted_factors = sorted(
            factor_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_factors = [factor for factor, score in sorted_factors[:3] if score > 0.1]

        if not top_factors:
            return f"Overall escalation score ({weighted_score:.2f}) exceeds threshold"

        # Map factors to readable reasons
        factor_reasons = {
            'low_intent_confidence': 'low confidence in intent classification',
            'low_evidence_quality': 'insufficient or poor quality historical evidence',
            'conflicting_historical_resolutions': 'conflicting resolutions in similar historical cases',
            'unsupported_requested_action': 'requested action may not be supported by policy',
            'high_risk_content': 'message contains potentially high-risk or sensitive content',
            'ambiguous_message': 'message is ambiguous or unclear'
        }

        reasons = [factor_reasons.get(factor, factor) for factor in top_factors]

        if len(reasons) == 1:
            base_reason = reasons[0]
        elif len(reasons) == 2:
            base_reason = f"{reasons[0]} and {reasons[1]}"
        else:
            base_reason = f"{', '.join(reasons[:-1])}, and {reasons[-1]}"

        return f"Escalation due to {base_reason}"


def main():
    """Main function to demonstrate escalation policy usage."""
    print("Initializing escalation policy...")
    policy = EscalationPolicy()

    # Test cases
    test_scenarios = [
        {
            'name': 'High confidence, good evidence',
            'intent_result': {'intent': 'order_status', 'confidence': 0.92},
            'retrieved_cases': [
                {'similarity': 0.85, 'intent': 'order_status', 'resolution': 'Provided tracking info'},
                {'similarity': 0.78, 'intent': 'order_status', 'resolution': 'Gave ETA'}
            ]
        },
        {
            'name': 'Low confidence, poor evidence',
            'intent_result': {'intent': 'refund_request', 'confidence': 0.45},
            'retrieved_cases': [
                {'similarity': 0.30, 'intent': 'payment_issue', 'resolution': 'Issued partial refund'},
                {'similarity': 0.25, 'intent': 'billing_problem', 'resolution': 'Explained charges'}
            ]
        },
        {
            'name': 'High risk content',
            'intent_result': {'intent': 'account_login', 'confidence': 0.80},
            'retrieved_cases': [
                {'similarity': 0.70, 'intent': 'account_login', 'resolution': 'Reset password'},
                {'similarity': 0.65, 'intent': 'account_login', 'resolution': 'Account unlocked'}
            ]
            # In reality, we'd add high-risk indicators to the cases or check the query
        },
        {
            'name': 'Conflicting resolutions',
            'intent_result': {'intent': 'technical_issue', 'confidence': 0.75},
            'retrieved_cases': [
                {'similarity': 0.80, 'intent': 'technical_issue', 'resolution': 'Provided workaround'},
                {'similarity': 0.75, 'intent': 'technical_issue', 'resolution': 'Escalated to senior team'},
                {'similarity': 0.70, 'intent': 'technical_issue', 'resolution': 'Issue cannot be resolved remotely'}
            ]
        }
    ]

    print("\n=== Escalation Examples ===")
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        result = policy.should_escalate(
            scenario['intent_result'],
            scenario['retrieved_cases']
        )
        print(f"Intent: {scenario['intent_result']['intent']} "
              f"(confidence: {scenario['intent_result']['confidence']:.3f})")
        print(f"Should escalate: {result['should_escalate']}")
        if result['reason']:
            print(f"Reason: {result['reason']}")
        print(f"Escalation score: {result.get('escalation_score', 0.0):.3f}")
        print(f"Top factors: {sorted(result['factor_scores'].items(), key=lambda x: x[1], reverse=True)[:3]}")


if __name__ == "__main__":
    main()