"""
Failure analysis for the Hiver support agent using human-labelled golden set.

Identifies the top 5 failure modes based on actual human-vs-system disagreement.
Saves results to experiments/failure_analysis.json
"""
import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANNOTATION_PATH = Path(__file__).parent / "human_annotation" / "annotation.csv"


def load_data():
    """Load golden set and run TF-IDF baseline to get predictions."""
    golden_df = pd.read_csv(ANNOTATION_PATH)
    logger.info(f"Loaded {len(golden_df)} examples")
    
    # Load training data
    train_df = pd.read_csv("data/processed/train.csv")
    train_texts = train_df["customer_message"].tolist()
    if "intent" in train_df.columns:
        train_labels = train_df["intent"].tolist()
    else:
        from classifier import _pseudo_label_batch
        train_labels = _pseudo_label_batch(train_texts)
    
    # Use TF-IDF baseline for fast intent predictions
    from baselines import TfIdfBaseline
    baseline = TfIdfBaseline()
    baseline.fit(train_texts, train_labels, train_texts)
    
    golden_texts = golden_df["customer_message"].tolist()
    pred_intents = baseline.predict_intent(golden_texts)
    pred_escalation = baseline.predict_escalation(golden_texts)
    
    # Convert to pipeline-like format
    predictions = []
    for i, (_, row) in enumerate(golden_df.iterrows()):
        predictions.append({
            'intent': {'name': pred_intents[i], 'confidence': 0.8},
            'should_escalate': pred_escalation[i],
            'reply': '',
            'retrieved_cases': [],
            'metadata': {'grounding_confidence': 0.0}
        })
    
    return golden_df, predictions


def categorize_failure(message: str, true_intent: str, pred_intent: str) -> str:
    """Categorize a failure into a named mode."""
    msg = message.lower()
    
    # Check for specific confusion patterns
    if true_intent == 'cancellation' and pred_intent == 'ride_status':
        return 'cancellation_as_ride_status'
    
    if true_intent == 'refund_request' and pred_intent == 'ride_status':
        return 'refund_request_as_ride_status'
    
    if true_intent == 'complaint' and pred_intent in ['ride_status', 'other']:
        return 'complaint_as_ride_status_or_other'
    
    if true_intent == 'trip_fare_dispute' and pred_intent == 'ride_status':
        return 'fare_dispute_as_ride_status'
    
    if true_intent in ['account_access', 'app_functionality'] and \
       pred_intent in ['account_access', 'app_functionality']:
        return 'access_vs_functionality_confusion'
    
    if true_intent == 'trip_fare_dispute' and pred_intent == 'other':
        return 'fare_dispute_as_other'
    
    if true_intent == 'complaint' and pred_intent == 'other':
        return 'complaint_as_other'
    
    if true_intent == 'refund_request' and pred_intent == 'trip_fare_dispute':
        return 'refund_request_as_fare_dispute'
    
    if true_intent == 'account_access' and pred_intent == 'ride_status':
        return 'account_access_as_ride_status'
    
    if true_intent == 'app_functionality' and pred_intent == 'ride_status':
        return 'app_functionality_as_ride_status'
    
    # Generic fallback
    return f'{true_intent}_as_{pred_intent}'


def get_hypothesis(mode: str) -> str:
    """Generate hypothesis for why this failure mode occurs."""
    hypotheses = {
        'cancellation_as_ride_status': 'Cancellation messages often mention rides, causing the classifier to default to ride_status',
        'refund_request_as_ride_status': 'Refund requests mention rides/trips, and the classifier lacks strong refund-specific features',
        'complaint_as_ride_status_or_other': 'Complaints about service quality are broad and lack specific intent signals',
        'fare_dispute_as_ride_status': 'Fare disputes involve ride-related vocabulary, confusing the classifier',
        'access_vs_functionality_confusion': 'Account access and app functionality both involve "not working" language',
        'fare_dispute_as_other': 'Fare disputes without clear keywords get classified as other',
        'complaint_as_other': 'General complaints lack specific intent markers',
        'refund_request_as_fare_dispute': 'Both involve financial aspects of rides',
        'account_access_as_ride_status': 'Account issues sometimes mention rides or trips',
        'app_functionality_as_ride_status': 'App issues during rides get classified as ride_status',
    }
    return hypotheses.get(mode, f'Classifier struggles to distinguish {mode.replace("_as_", " from ")}')


def get_fix(mode: str) -> str:
    """Generate proposed fix for this failure mode."""
    fixes = {
        'cancellation_as_ride_status': 'Add stronger cancellation-specific features (cancel, cancellation, delete account)',
        'refund_request_as_ride_status': 'Add refund-specific keywords and hierarchical classification',
        'complaint_as_ride_status_or_other': 'Add complaint-specific features or multi-label classification',
        'fare_dispute_as_ride_status': 'Better distinguish fare disputes from general ride status',
        'access_vs_functionality_confusion': 'Add more specific account vs app keywords',
        'fare_dispute_as_other': 'Add fare-related keywords to prevent fallthrough to other',
        'complaint_as_other': 'Add complaint-specific features',
        'refund_request_as_fare_dispute': 'Add hierarchical classification for financial intents',
        'account_access_as_ride_status': 'Add account-specific features',
        'app_functionality_as_ride_status': 'Add app-specific features',
    }
    return fixes.get(mode, 'Collect more labeled data for this confusion pattern')


def run_failure_analysis():
    """Run complete failure analysis."""
    golden_df, predictions = load_data()
    
    n = len(golden_df)
    intent_failures = []
    escalation_failures = []
    mode_counts = defaultdict(list)
    
    for i in range(n):
        true_intent = golden_df.iloc[i]['human_intent']
        pred_intent = predictions[i]['intent']['name']
        true_escalate = bool(golden_df.iloc[i]['human_should_escalate'])
        pred_escalate = bool(predictions[i]['should_escalate'])
        message = golden_df.iloc[i]['customer_message']
        
        if true_intent != pred_intent:
            mode = categorize_failure(message, true_intent, pred_intent)
            intent_failures.append({
                'index': i,
                'example_id': int(golden_df.iloc[i]['example_id']),
                'customer_message': message[:200],
                'true_intent': true_intent,
                'pred_intent': pred_intent,
                'mode': mode
            })
            mode_counts[mode].append(intent_failures[-1])
        
        if true_escalate != pred_escalate:
            escalation_failures.append({
                'index': i,
                'example_id': int(golden_df.iloc[i]['example_id']),
                'customer_message': message[:200],
                'true_escalate': true_escalate,
                'pred_escalate': pred_escalate,
                'direction': 'missed_escalation' if true_escalate and not pred_escalate else 'false_escalation'
            })
    
    # Top 5 failure modes
    top_modes = sorted(mode_counts.items(), key=lambda x: -len(x[1]))[:5]
    
    # Confusion matrix
    confusion_pairs = Counter()
    for f in intent_failures:
        pair = (f['true_intent'], f['pred_intent'])
        confusion_pairs[pair] += 1
    
    # Build analysis
    analysis = {
        'total_examples': n,
        'intent_failures': len(intent_failures),
        'intent_error_rate': round(len(intent_failures) / n, 3),
        'escalation_failures': len(escalation_failures),
        'escalation_error_rate': round(len(escalation_failures) / n, 3),
        'top_confusion_pairs': [
            {'true': pair[0], 'predicted': pair[1], 'count': count}
            for pair, count in confusion_pairs.most_common(10)
        ],
        'failure_modes': []
    }
    
    for mode, examples in top_modes:
        mode_analysis = {
            'name': mode,
            'count': len(examples),
            'frequency': f"{len(examples)/n*100:.1f}%",
            'examples': [
                {
                    'example_id': ex['example_id'],
                    'customer_message': ex['customer_message'],
                    'true_intent': ex['true_intent'],
                    'pred_intent': ex['pred_intent']
                }
                for ex in examples[:3]
            ],
            'root_cause': get_hypothesis(mode),
            'proposed_fix': get_fix(mode),
            'component': _identify_component(mode)
        }
        analysis['failure_modes'].append(mode_analysis)
    
    # Escalation failure breakdown
    missed_escalations = [f for f in escalation_failures if f['direction'] == 'missed_escalation']
    false_escalations = [f for f in escalation_failures if f['direction'] == 'false_escalation']
    
    analysis['escalation_breakdown'] = {
        'missed_escalations': len(missed_escalations),
        'false_escalations': len(false_escalations),
        'examples_missed': [
            {'example_id': f['example_id'], 'message': f['customer_message'][:100]}
            for f in missed_escalations[:5]
        ],
        'examples_false': [
            {'example_id': f['example_id'], 'message': f['customer_message'][:100]}
            for f in false_escalations[:5]
        ]
    }
    
    return analysis


def _identify_component(mode: str) -> str:
    """Identify which component is primarily responsible."""
    if 'confusion' in mode or 'as_' in mode:
        return 'classifier'
    elif 'escalation' in mode:
        return 'escalation'
    elif 'grounding' in mode:
        return 'retrieval/generation'
    else:
        return 'classifier'


def save_analysis(analysis: Dict[str, Any]):
    """Save analysis to JSON and print summary."""
    output_path = Path("experiments/failure_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS (Human-labelled golden set)")
    print("=" * 60)
    print(f"Total examples: {analysis['total_examples']}")
    print(f"Intent failures: {analysis['intent_failures']} ({analysis['intent_error_rate']*100:.1f}%)")
    print(f"Escalation failures: {analysis['escalation_failures']} ({analysis['escalation_error_rate']*100:.1f}%)")
    
    print(f"\nTop 5 Failure Modes:")
    for i, mode in enumerate(analysis['failure_modes'], 1):
        print(f"\n  {i}. {mode['name']}")
        print(f"     Count: {mode['count']} ({mode['frequency']})")
        print(f"     Root cause: {mode['root_cause']}")
        print(f"     Component: {mode['component']}")
        print(f"     Fix: {mode['proposed_fix']}")
        if mode['examples']:
            ex = mode['examples'][0]
            print(f"     Example: \"{ex['customer_message'][:60]}...\"")
            print(f"       True: {ex['true_intent']} -> Predicted: {ex['pred_intent']}")
    
    print(f"\nTop Confusion Pairs:")
    for pair in analysis['top_confusion_pairs'][:5]:
        print(f"  {pair['true']} -> {pair['predicted']}: {pair['count']}")
    
    print(f"\nEscalation Breakdown:")
    eb = analysis['escalation_breakdown']
    print(f"  Missed escalations: {eb['missed_escalations']}")
    print(f"  False escalations: {eb['false_escalations']}")
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    analysis = run_failure_analysis()
    save_analysis(analysis)
