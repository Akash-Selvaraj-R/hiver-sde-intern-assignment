"""
Generate golden evaluation set template for hand-labelling.

This script creates a 200-example golden set with:
- Realistic intent distribution (derived from data, not perfectly balanced)
- Escalation cases (~25% of examples)
- Diverse message types (short, long, noisy, formal, informal)
- Clear instructions for manual annotators

The auto-generated labels are PLACEHOLDERS that must be verified/overridden by a human annotator.
"""
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import List, Dict
import random

np.random.seed(42)
random.seed(42)

# Escalation-triggering patterns
ESCALATION_KEYWORDS = [
    'lawyer', 'legal', 'sue', 'attorney', 'fraud', 'stolen', 'unauthorized',
    'disappointed', 'worst', 'terrible', 'horrible', 'unacceptable', 'furious',
    'angry', 'outraged', 'scam', 'rip off', 'ripped off', 'never again',
    'cancel everything', 'report', 'BBB', 'better business', 'social media',
    'viral', 'expose', 'manager', 'supervisor', 'escalate', 'complaint'
]

# Intents with realistic frequency weights (from typical e-commerce support)
INTENT_WEIGHTS = {
    'order_status': 0.22,
    'refund_request': 0.15,
    'technical_issue': 0.14,
    'payment_issue': 0.12,
    'account_login': 0.10,
    'complaint': 0.09,
    'cancellation': 0.07,
    'billing_problem': 0.05,
    'information_request': 0.04,
    'other': 0.02,
}

# Message templates per intent
MESSAGE_TEMPLATES = {
    'order_status': [
        "Where is my order #{order_id}?",
        "My package hasn't arrived yet",
        "Order still shows processing, it's been {days} days",
        "Tracking number {tracking} not updating",
        "When will my order ship?",
        "Need to update shipping address for order #{order_id}",
        "My order arrived but item is missing",
        "Wrong item delivered for order #{order_id}",
        "Package marked delivered but I didn't receive it",
        "Can I change the delivery date?",
    ],
    'refund_request': [
        "I want a refund for my order",
        "How do I get my money back?",
        "Refund not processed yet, it's been {days} days",
        "Item arrived damaged, need full refund",
        "Charged twice, need refund for duplicate charge",
        "I returned the item {days} days ago, where's my refund?",
        "This product is defective, I want a complete refund",
        "Can I get a partial refund for the damaged part?",
        "Refund was denied but I disagree with the decision",
        "Need refund for incorrect item received",
    ],
    'payment_issue': [
        "Was charged ${amount} twice for the same order",
        "Payment didn't go through but money was deducted",
        "Need to update my payment method",
        "Unauthorized charge on my account",
        "My coupon code didn't apply at checkout",
        "Why was I charged ${amount} when the item was free?",
        "Payment failed but order shows as placed",
        "Can I split payment between two cards?",
        "Charge shows different amount than expected",
        "Promo code not working at checkout",
    ],
    'account_login': [
        "Can't log into my account",
        "Password reset email not received",
        "Account locked after too many attempts",
        "Unable to access account, getting error",
        "Two-factor authentication code not working",
        "Need to change email on my account",
        "Account shows wrong information",
        "Someone else accessed my account",
        "Can't remember which email I used to sign up",
        "Deactivate my account permanently",
    ],
    'technical_issue': [
        "Website keeps crashing when I try to checkout",
        "App not loading properly on my phone",
        "Error message when trying to place order",
        "Page not loading, keeps spinning",
        "Features not working as expected",
        "Can't upload photos for my review",
        "Search function not returning results",
        "Checkout page broken, can't complete purchase",
        "Mobile app crashes every time I open it",
        "Notifications not working in the app",
    ],
    'cancellation': [
        "Want to cancel my order before it ships",
        "How do I cancel my subscription?",
        "Need to stop recurring payment",
        "Order placed by mistake, need to cancel",
        "Cancel my membership immediately",
        "I want to cancel but keep my account",
        "Cancellation request from {days} days ago not processed",
        "Need to cancel and get a refund",
        "Stop sending me promotional emails",
        "Cancel my premium subscription",
    ],
    'billing_problem': [
        "Don't understand charge on my statement",
        "Received incorrect invoice amount",
        "Billing address needs to be updated",
        "Charge higher than the advertised price",
        "Missing receipt for my purchase",
        "Invoice shows items I didn't order",
        "Tax calculation seems wrong on my bill",
        "Need itemized receipt for expense report",
        "Billing cycle changed without notice",
        "Account shows balance but I already paid",
    ],
    'information_request': [
        "What is your return policy?",
        "Do you offer international shipping?",
        "How long does standard shipping take?",
        "What payment methods do you accept?",
        "Is this item currently in stock?",
        "Can I get a price match if it goes on sale?",
        "What are the store hours?",
        "Do you offer gift wrapping?",
        "How do I use my loyalty points?",
        "What's the warranty on this product?",
    ],
    'complaint': [
        "Very unhappy with the service I received",
        "This is completely unacceptable",
        "Poor quality product for the price",
        "Worst customer service experience ever",
        "Extremely disappointed with my purchase",
        "Your competitor offers better service",
        "I've been waiting {days} days with no response",
        "This is the third time I'm contacting you",
        "Your product broke after one week of use",
        "Not worth the money at all",
    ],
    'other': [
        "Thanks for the help!",
        "Just saying hello to the team",
        "Testing if this support chat works",
        "Love your products!",
        "Happy customer here",
        "Do you have any job openings?",
        "Can I speak to someone in Spanish?",
        "What's your social media handle?",
        "Do you collaborate with influencers?",
        "My cat walked on my keyboard and sent a message",
    ],
}


def should_escalate_heuristic(message: str, intent: str) -> tuple:
    """Determine if a message should likely be escalated based on heuristics."""
    msg_lower = message.lower()
    
    # Explicit escalation language
    has_escalation_keywords = any(kw in msg_lower for kw in ESCALATION_KEYWORDS)
    
    # Financial risk indicators
    has_fraud_language = any(w in msg_lower for w in ['fraud', 'stolen', 'unauthorized', 'scam'])
    
    # High-risk intents that often need escalation
    high_risk_intents = {'complaint', 'payment_issue'}
    is_high_risk = intent in high_risk_intents
    
    # Should escalate if explicit keywords or fraud language
    should_esc = has_escalation_keywords or has_fraud_language
    
    # Also escalate some high-risk intents with moderate probability
    if is_high_risk and not should_esc:
        should_esc = random.random() < 0.4
    
    # Also escalate some ambiguous cases
    if intent == 'other' and random.random() < 0.3:
        should_esc = True
    
    reason = ""
    if should_esc:
        if has_fraud_language:
            reason = "Potential fraud or unauthorized activity requiring investigation"
        elif 'unhappy' in msg_lower or 'disappointed' in msg_lower or 'worst' in msg_lower or 'terrible' in msg_lower:
            reason = "Strong customer dissatisfaction requiring de-escalation"
        elif any(w in msg_lower for w in ['manager', 'supervisor', 'escalate']):
            reason = "Customer explicitly requested escalation"
        elif any(w in msg_lower for w in ['lawyer', 'legal', 'sue', 'report']):
            reason = "Legal/compliance risk requiring immediate attention"
        elif any(w in msg_lower for w in ['cancel everything', 'never again', 'report']):
            reason = "Customer at risk of churn with strong negative sentiment"
        elif is_high_risk:
            reason = f"High-risk intent ({intent}) with ambiguous resolution path"
        else:
            reason = "Requires human review due to complexity"
    
    return should_esc, reason


def generate_golden_set(target_size: int = 200) -> List[Dict]:
    """Generate golden set examples with realistic distribution."""
    examples = []
    
    # Calculate number of examples per intent based on weights
    intent_counts = {}
    remaining = target_size
    for intent, weight in sorted(INTENT_WEIGHTS.items(), key=lambda x: -x[1]):
        count = max(5, round(target_size * weight))
        intent_counts[intent] = min(count, remaining)
        remaining -= intent_counts[intent]
    
    # Distribute remaining examples
    intents = list(INTENT_WEIGHTS.keys())
    idx = 0
    while remaining > 0:
        intent = intents[idx % len(intents)]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        remaining -= 1
        idx += 1
    
    example_id = 1
    for intent, count in intent_counts.items():
        templates = MESSAGE_TEMPLATES[intent]
        for i in range(count):
            template = templates[i % len(templates)]
            
            # Fill in template variables
            message = template.format(
                order_id=random.randint(10000, 99999),
                tracking=f"TRK{random.randint(100000, 999999)}",
                days=random.randint(1, 30),
                amount=random.choice([9.99, 19.99, 29.99, 49.99, 99.99, 149.99]),
            )
            
            should_esc, reason = should_escalate_heuristic(message, intent)
            
            # Add some ambiguity flags
            notes_parts = []
            if intent in ['information_request', 'technical_issue']:
                notes_parts.append("Potential ambiguity with related intents")
            if should_esc:
                notes_parts.append("Requires escalation review")
            if len(message) < 20:
                notes_parts.append("Short message - may lack context")
            
            examples.append({
                'example_id': example_id,
                'conversation_id': f"conv_{example_id:04d}",
                'customer_message': message,
                'intent': intent,  # PLACEHOLDER - must be verified by human annotator
                'should_escalate': should_esc,  # PLACEHOLDER - must be verified by human annotator
                'escalation_reason': reason if should_esc else '',
                'expected_resolution': f"Resolve {intent.replace('_', ' ')} issue",  # PLACEHOLDER
                'annotator_notes': '; '.join(notes_parts) if notes_parts else '',
                'human_labelled': False,  # Must be set to True after manual review
            })
            example_id += 1
    
    # Shuffle to avoid order bias
    random.shuffle(examples)
    
    # Re-assign example_ids after shuffle
    for i, ex in enumerate(examples):
        ex['example_id'] = i + 1
    
    return examples


def main():
    """Generate and save the golden set."""
    print("Generating golden evaluation set template...")
    
    examples = generate_golden_set(200)
    
    # Save as JSON
    output_path = Path("evaluation/golden_set.json")
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} examples to {output_path}")
    
    # Save annotation template (without pre-filled labels)
    template_path = Path("evaluation/golden_set_annotation_template.csv")
    template_data = []
    for ex in examples:
        template_data.append({
            'example_id': ex['example_id'],
            'conversation_id': ex['conversation_id'],
            'customer_message': ex['customer_message'],
            'intent': '',  # Empty - annotator fills this
            'should_escalate': '',  # Empty - annotator fills this
            'escalation_reason': '',  # Empty - annotator fills this
            'expected_resolution': '',  # Empty - annotator fills this
            'annotator_notes': '',
        })
    template_df = pd.DataFrame(template_data)
    template_df.to_csv(template_path, index=False)
    print(f"Saved annotation template to {template_path}")
    
    # Print statistics
    intents = {}
    escalations = {'True': 0, 'False': 0}
    for ex in examples:
        intents[ex['intent']] = intents.get(ex['intent'], 0) + 1
        escalations[str(ex['should_escalate'])] += 1
    
    print(f"\nStatistics:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Intent distribution:")
    for intent, count in sorted(intents.items(), key=lambda x: -x[1]):
        print(f"    {intent}: {count} ({count/len(examples)*100:.1f}%)")
    print(f"  Escalation distribution:")
    for esc, count in escalations.items():
        print(f"    {esc}: {count} ({count/len(examples)*100:.1f}%)")


if __name__ == "__main__":
    main()
