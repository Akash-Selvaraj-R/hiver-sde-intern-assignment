"""
Response generation module using LLMs with grounding constraints.
"""
import pandas as pd
import numpy as np
import yaml
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import os
import random
from dataclasses import dataclass

# Try to import OpenAI, but provide fallback for when API key is not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not available. Will use template-based generation.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from response generation."""
    reply: str
    evidence_ids: List[str]
    grounding_confidence: float
    model_used: str
    tokens_used: Optional[int] = None


class ResponseGenerator:
    """
    Generates customer-support responses grounded in historical evidence.
    """

    def __init__(self, config_path: str = "configs/brand.yaml"):
        """Initialize generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.brand_name = self.config['brand_name']

        # Generation parameters
        self.max_length = 150
        self.temperature = 0.3
        self.top_p = 0.9

        # Check for OpenAI API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.use_openai = OPENAI_AVAILABLE and bool(self.openai_api_key)

        if self.use_openai:
            openai.api_key = self.openai_api_key
            logger.info("OpenAI API configured for response generation")
        else:
            logger.info("Using template-based response generation (OpenAI not available or not configured)")

        # Load intent-specific response templates
        self.response_templates = self._load_response_templates()

    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load or create response templates for each intent."""
        # In a real system, these would come from analyzing historical brand responses
        templates = {
            'ride_status': [
                "I've looked into your ride status. {resolution_details}",
                "I can see your driver's location. {resolution_details}",
                "Let me check on your current ride. {resolution_details}"
            ],
            'trip_fare_dispute': [
                "I've reviewed the fare for your trip. {resolution_details}",
                "I can see the charge details for your ride. {resolution_details}",
                "Let me look into the fare calculation for your trip. {resolution_details}"
            ],
            'account_access': [
                "I can help with your account access. {resolution_details}",
                "I've checked your account status. {resolution_details}",
                "Let me assist with your account issue. {resolution_details}"
            ],
            'app_functionality': [
                "I apologize for the technical difficulty with the app. {resolution_details}",
                "I've noted the app issue you're experiencing. {resolution_details}",
                "Our team is aware of this issue. {resolution_details}"
            ],
            'complaint': [
                "I'm sorry to hear about your experience. {resolution_details}",
                "I understand your frustration. {resolution_details}",
                "Thank you for bringing this to our attention. {resolution_details}"
            ],
            'refund_request': [
                "I've reviewed your refund request. {resolution_details}",
                "I can help with your refund. {resolution_details}",
                "Let me process your refund request. {resolution_details}"
            ],
            'cancellation': [
                "I can help with your cancellation request. {resolution_details}",
                "I've processed your cancellation. {resolution_details}",
                "Let me assist with canceling your {cancellation_type}. {resolution_details}"
            ],
            'other': [
                "Thanks for reaching out. How can I assist you today?",
                "I'm here to help. What can I do for you?",
                "Thanks for contacting Uber Support. How may I help?"
            ]
        }

        return templates

    def _fill_template(self, template: str, case: Dict[str, Any], query: str) -> str:
        """Fill a response template with information from the case and query."""
        filled = template

        # Extract order ID or similar identifiers from query if present
        import re
        order_match = re.search(r'#?(\d{4,})', query)
        order_id = order_match.group(1) if order_match else "12345"

        # Common replacements
        replacements = {
            '{order_id}': order_id,
            '{status}': 'shipped',  # Would come from actual case data in reality
            '{date}': 'within 3-5 business days',
            '{information}': 'The information you requested is available in your account',
            '{policy_info}': 'Our standard policy applies to this situation',
            '{confirmation_details}': 'This is correct according to our records',
            '{workaround}': 'Please try clearing your browser cache and cookies',
            '{resolution}': 'We can offer a discount on your next purchase or expedited shipping'
        }

        for placeholder, value in replacements.items():
            filled = filled.replace(placeholder, value)

        # If we have brand response from the case, use that as base
        if case.get('brand_response'):
            # Use the actual brand response but ground it in the evidence
            base_response = case['brand_response']
            # Apply same template replacements to the actual response
            for placeholder, value in replacements.items():
                base_response = base_response.replace(placeholder, value)
            filled = base_response

        return filled

    def generate(self,
                 query: str,
                 intent: str,
                 confidence: float,
                 retrieved_cases: List[Dict[str, Any]]) -> GenerationResult:
        """
        Generate a response grounded in retrieved historical evidence.

        Args:
            query: The customer message/query
            intent: The predicted intent
            confidence: Confidence in the intent prediction
            retrieved_cases: List of retrieved historical cases

        Returns:
            GenerationResult with reply, evidence IDs, and grounding confidence
        """
        logger.info(f"Generating response for intent '{intent}' with {len(retrieved_cases)} evidence cases")

        # Calculate grounding confidence based on retrieval quality
        grounding_confidence = self._calculate_grounding_confidence(retrieved_cases, intent)

        # Extract resolution actions from historical brand responses
        resolution_actions = self._extract_resolution_actions(retrieved_cases)

        # If we have good evidence with brand responses, use it to ground the response
        if retrieved_cases and grounding_confidence > 0.3:
            evidence_ids = [str(case.get('conversation_id', f'case_{i}'))
                           for i, case in enumerate(retrieved_cases)]

            # Try to use OpenAI if available, otherwise use grounded templates
            if self.use_openai and len(retrieved_cases) > 0:
                reply = self._generate_with_openai(query, intent, retrieved_cases)
            else:
                reply = self._generate_grounded(query, intent, retrieved_cases, resolution_actions)
        else:
            # Fallback to generic response when evidence is poor
            logger.warning("Poor retrieval quality, using fallback response")
            evidence_ids = []
            reply = self._generate_fallback(intent)

        # Ensure we don't exceed max length
        if len(reply) > self.max_length:
            reply = reply[:self.max_length-3] + "..."

        result = GenerationResult(
            reply=reply,
            evidence_ids=evidence_ids,
            grounding_confidence=grounding_confidence,
            model_used="openai" if self.use_openai else "template",
            tokens_used=None
        )

        logger.info(f"Generated response: {reply[:100]}...")
        return result

    def _extract_resolution_actions(self, cases: List[Dict[str, Any]]) -> List[str]:
        """Extract resolution actions from historical brand responses."""
        actions = []
        for case in cases:
            brand_resp = case.get('brand_response', '')
            resolution = case.get('resolution', '')
            if brand_resp and len(brand_resp) > 10:
                # Extract actionable content from brand response
                action = self._parse_action_from_response(brand_resp)
                if action:
                    actions.append(action)
            elif resolution and len(resolution) > 10:
                actions.append(resolution)
        return actions

    def _parse_action_from_response(self, response: str) -> str:
        """Parse the key action/resolution from a brand response text."""
        response_lower = response.lower()
        
        # Extract common resolution patterns
        action_patterns = [
            (r'(send|email|contact)\s+us\s+(at|via|through)', 'contact support via provided channel'),
            (r'(please|kindly)\s+(share|provide|send)', 'requesting additional information'),
            (r'(refund|credit|reimburse)', 'processing refund/credit'),
            (r'(cancel|cancelled|cancellation)', 'handling cancellation request'),
            (r'(reset|update|change)\s+(your\s+)?(password|account|email)', 'account modification assistance'),
            (r'(investigate|review|look into)', 'investigating the issue'),
            (r'(safe|safety)', 'addressing safety concern'),
            (r'(lost|missing)\s+(item|phone|bag)', 'lost item retrieval process'),
        ]
        
        for pattern, action_desc in action_patterns:
            if re.search(pattern, response_lower):
                return action_desc
        
        # If no specific pattern found, use a summary of the response
        if len(response) > 20:
            # Take the first sentence as a summary
            first_sentence = response.split('.')[0].strip()
            if len(first_sentence) > 10:
                return first_sentence[:100]
        
        return ''

    def _generate_grounded(self, query: str, intent: str, cases: List[Dict[str, Any]],
                          resolution_actions: List[str]) -> str:
        """Generate a response grounded in historical resolution patterns."""
        templates = self.response_templates.get(intent, self.response_templates.get('other', []))
        
        # Get the top case for direct evidence
        top_case = cases[0] if cases else {}
        brand_response = top_case.get('brand_response', '')
        
        # If we have a real brand response, adapt it for this query
        if brand_response and len(brand_response) > 15:
            # Clean and adapt the historical response
            adapted = self._adapt_historical_response(brand_response, query, intent)
            if adapted:
                return adapted
        
        # If we have resolution actions, incorporate them
        if resolution_actions:
            primary_action = resolution_actions[0]
            template = random.choice(templates) if 'random' in globals() else templates[0]
            # Fill template with the actual resolution action
            filled = template.replace('{resolution_details}', primary_action)
            return filled
        
        # Fallback to standard template
        template = random.choice(templates) if 'random' in globals() else templates[0]
        return self._fill_template(template, top_case, query)

    def _adapt_historical_response(self, historical_response: str, query: str, intent: str) -> str:
        """Adapt a historical brand response for the current query."""
        # Remove Twitter-specific formatting
        cleaned = re.sub(r'@\w+', '', historical_response).strip()
        cleaned = re.sub(r'https?://\S+', '', cleaned).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Fix broken sentences left after URL removal (e.g. "Send us a note via  and we'll...")
        cleaned = re.sub(r'\bvia\s+and\b', 'and', cleaned)
        cleaned = re.sub(r'\bat\s+and\b', 'and', cleaned)
        cleaned = re.sub(r'\bvia\s+\.', '.', cleaned)
        cleaned = re.sub(r'\bat\s+\.', '.', cleaned)
        cleaned = re.sub(r'\bvia\s*$', '', cleaned).strip()
        cleaned = re.sub(r'\bat\s*$', '', cleaned).strip()
        cleaned = re.sub(r'\bhere:\s*$', '', cleaned).strip()
        cleaned = re.sub(r'\bhere\s*$', '', cleaned).strip()
        # Remove trailing punctuation artifacts
        cleaned = re.sub(r'[,\s]+$', '', cleaned).strip()

        # If the cleaned response is substantial, use it
        if len(cleaned) > 15:
            # Add a brief intro if the response jumps directly to action
            intro_map = {
                'ride_status': "Regarding your ride, ",
                'trip_fare_dispute': "Regarding your fare concern, ",
                'account_access': "Regarding your account, ",
                'app_functionality': "Regarding the app issue, ",
                'complaint': "Thank you for your feedback. ",
                'refund_request': "Regarding your refund request, ",
                'cancellation': "Regarding your cancellation, ",
                'other': ""
            }
            intro = intro_map.get(intent, "")

            # Only add intro if the response doesn't already start with one
            if not any(cleaned.lower().startswith(w) for w in ['we ', 'i ', 'our ', 'let ', 'please ']):
                return intro + cleaned

        return ''

    def _calculate_grounding_confidence(self, retrieved_cases: List[Dict[str, Any]], intent: str) -> float:
        """Calculate confidence in the grounding based on retrieval quality."""
        if not retrieved_cases:
            return 0.0

        # Base confidence on similarity scores
        similarities = [case.get('similarity', 0.0) for case in retrieved_cases]
        avg_similarity = np.mean(similarities) if similarities else 0.0

        # Check intent alignment
        intent_matches = [1.0 if case.get('intent') == intent else 0.0
                         for case in retrieved_cases]
        intent_alignment = np.mean(intent_matches) if intent_matches else 0.0

        # Combine scores
        grounding_confidence = (avg_similarity * 0.6) + (intent_alignment * 0.4)
        return min(1.0, max(0.0, grounding_confidence))

    def _generate_with_openai(self, query: str, intent: str, retrieved_cases: List[Dict[str, Any]]) -> str:
        """Generate response using OpenAI API with grounding constraints."""
        try:
            # Prepare evidence context
            evidence_context = ""
            for i, case in enumerate(retrieved_cases[:3]):  # Use top 3 cases
                evidence_context += f"Evidence {i+1}:\n"
                evidence_context += f"Customer said: {case.get('customer_message', '')}\n"
                if case.get('brand_response'):
                    evidence_context += f"Brand responded: {case.get('brand_response')}\n"
                evidence_context += f"Resolution: {case.get('resolution', '')}\n\n"

            # Construct prompt with strict grounding instructions
            prompt = f"""You are a customer support agent for {self.brand_name}.
Generate a helpful, professional response to the customer's message based ONLY on the provided evidence.
Do not invent information, make promises, or guarantee outcomes not supported by the evidence.
Be concise and follow the brand's historical support style evident in the examples.

Customer message: "{query}"

Relevant historical evidence:
{evidence_context}

Instructions:
1. Answer the customer's actual issue
2. Follow the brand's historical support style from the evidence
3. Use ONLY information supported by the retrieved evidence
4. Never invent policies, timelines, refunds, guarantees, or actions
5. Be concise and professional
6. Ask for clarification only if absolutely necessary

Response:"""

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful customer support agent that strictly adheres to provided evidence."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p
            )

            reply = response.choices[0].message.content.strip()
            return reply

        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            # Fallback to template-based generation
            return self._generate_with_templates(query, intent, retrieved_cases[0] if retrieved_cases else {})

    def _generate_with_templates(self, query: str, intent: str, case: Dict[str, Any]) -> str:
        """Generate response using template-based approach."""
        if intent not in self.response_templates:
            intent = 'other'  # Fallback to other

        templates = self.response_templates[intent]
        template = random.choice(templates) if 'random' in globals() else templates[0]

        # Fill the template
        filled = self._fill_template(template, case, query)

        # If we have a good brand response from the case, prefer to use/adapt it
        if case.get('brand_response') and len(case['brand_response']) > 10:
            # Use the actual response but ensure it's appropriate
            base_response = case['brand_response']
            # Apply light templating to make it more specific to the query
            filled = self._fill_template(base_response, case, query)

        return filled

    def _generate_fallback(self, intent: str) -> str:
        """Generate a fallback response when evidence quality is poor."""
        fallback_responses = {
            'ride_status': "I can see you're having an issue with your ride. Could you share your trip details so I can look into this?",
            'trip_fare_dispute': "I understand you have a concern about your fare. Let me review the charge details for your trip.",
            'account_access': "I can help with your account issue. Could you share more details about what's happening?",
            'app_functionality': "I apologize for the technical difficulty you're experiencing with the app. Our team is working on it.",
            'complaint': "I'm sorry to hear about your experience. I want to help make this right.",
            'refund_request': "I can help with your refund request. Could you share the details of the trip in question?",
            'cancellation': "I can help with your cancellation request. Please let me know what you need to cancel.",
            'other': "How can I assist you today?"
        }

        return fallback_responses.get(intent, fallback_responses['other'])


def main():
    """Main function to demonstrate generator usage."""
    print("Initializing response generator...")
    generator = ResponseGenerator()

    # Create some mock retrieved cases for demonstration
    mock_cases = [
        {
            'conversation_id': 'case_001',
            'customer_message': 'Where is my order #12345?',
            'brand_response': "I've checked your order #12345 and it's currently shipped with tracking #ABC123",
            'resolution': 'Provided tracking information',
            'intent': 'order_status',
            'similarity': 0.85
        },
        {
            'conversation_id': 'case_002',
            'customer_message': 'I want a refund for my purchase',
            'brand_response': 'I\'ve initiated a refund for your purchase. It should appear in 3-5 business days.',
            'resolution': 'Refund processed',
            'intent': 'refund_request',
            'similarity': 0.78
        }
    ]

    test_queries = [
        ("Where is my order #67890?", "order_status", 0.92),
        ("Can I get a refund?", "refund_request", 0.87),
        ("I need help with my account", "account_login", 0.75)
    ]

    print("\n=== Generation Examples ===")
    for query, intent, confidence in test_queries:
        print(f"\nQuery: {query}")
        print(f"Intent: {intent} (confidence: {confidence:.3f})")

        result = generator.generate(query, intent, confidence, mock_cases[:1])
        print(f"Response: {result.reply}")
        print(f"Evidence IDs: {result.evidence_ids}")
        print(f"Grounding confidence: {result.grounding_confidence:.3f}")
        print(f"Model used: {result.model_used}")


if __name__ == "__main__":
    main()