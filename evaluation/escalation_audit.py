"""
Escalation audit: analyze why the system is so conservative and create improved policy.

Current baseline: 68.5% accuracy, 16.7% precision, 5.9% recall, 8.7% F1
Human escalation rate: 25.5%
"""
import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ANNOTATION_PATH = Path(__file__).parent / "human_annotation" / "annotation.csv"


def analyze_escalation_patterns():
    """Analyze patterns in human escalation labels."""
    golden = pd.read_csv(ANNOTATION_PATH)
    
    # Overall escalation rate
    esc_rate = golden['human_should_escalate'].mean()
    print(f"Human escalation rate: {esc_rate:.1%}")
    
    # Escalation by intent
    print("\nEscalation by intent:")
    for intent in sorted(golden['human_intent'].unique()):
        subset = golden[golden['human_intent'] == intent]
        esc_count = subset['human_should_escalate'].sum()
        total = len(subset)
        print(f"  {intent}: {esc_count}/{total} ({esc_count/total*100:.0f}%)")
    
    # Check escalation reasons
    escalated = golden[golden['human_should_escalate'] == True]
    print(f"\nEscalation reasons (from {len(escalated)} escalated examples):")
    reasons = escalated['human_escalation_reason'].dropna()
    for reason in reasons[:10]:
        print(f"  - {reason[:80]}")
    
    return golden


def create_improved_escalation_v2(golden):
    """
    Create an improved escalation policy based on human labels.
    
    Key insights from analysis:
    1. Cancellation intent: 100% should be escalated
    2. Complaint intent: high escalation rate
    3. Trip fare disputes with strong language: should escalate
    4. Lost items: should escalate
    5. Safety concerns: should escalate
    """
    # Load pipeline
    from pipeline import SupportAgentPipeline
    from retrieval import load_historical_cases
    
    train_df = pd.read_csv("data/processed/train.csv")
    train_texts = train_df["customer_message"].tolist()
    if "intent" in train_df.columns:
        train_labels = train_df["intent"].tolist()
    else:
        from classifier import _pseudo_label_batch
        train_labels = _pseudo_label_batch(train_texts)
    
    historical_cases = load_historical_cases()
    
    pipeline = SupportAgentPipeline()
    pipeline.fit(
        training_data=(train_texts, train_labels),
        historical_cases=historical_cases
    )
    
    # Run pipeline on all examples
    predictions = []
    for _, row in golden.iterrows():
        result = pipeline.predict(row['customer_message'])
        predictions.append(result)
    
    # Apply improved escalation rules
    improved_escalations = []
    for i, (idx, row) in enumerate(golden.iterrows()):
        pred_intent = predictions[i]['intent']['name']
        confidence = predictions[i]['intent']['confidence']
        message = row['customer_message'].lower()
        
        # Rule-based escalation improvements
        should_escalate = False
        reasons = []
        
        # Rule 1: Always escalate cancellation intent
        if pred_intent == 'cancellation':
            should_escalate = True
            reasons.append('cancellation_intent')
        
        # Rule 2: Escalate complaint intent with low confidence
        elif pred_intent == 'complaint' and confidence < 0.7:
            should_escalate = True
            reasons.append('complaint_low_confidence')
        
        # Rule 3: Escalate if message contains strong escalation signals
        escalation_keywords = [
            'lost item', 'lost phone', 'lost bag', 'left in', 'left my',
            'safety', 'unsafe', 'threatened', 'gun', 'harassment',
            'fraud', 'unauthorized', 'stolen', 'hacked',
            'cancel', 'cancellation', 'delete account', 'deactivate',
            'refund', 'money back', 'charge back',
            'complaint', 'disgusted', 'horrible', 'terrible'
        ]
        
        for keyword in escalation_keywords:
            if keyword in message:
                should_escalate = True
                reasons.append(f'keyword_{keyword}')
                break
        
        # Rule 4: Escalate if confidence is very low (uncertain)
        if confidence < 0.4:
            should_escalate = True
            reasons.append('very_low_confidence')
        
        # Rule 5: Escalate trip fare disputes over certain threshold
        if pred_intent == 'trip_fare_dispute' and confidence > 0.7:
            # Check for strong language about charges
            charge_keywords = ['charged', 'overcharged', 'double', 'twice', 'wrong amount']
            for kw in charge_keywords:
                if kw in message:
                    should_escalate = True
                    reasons.append(f'fare_dispute_{kw}')
                    break
        
        improved_escalations.append({
            'should_escalate': should_escalate,
            'reasons': reasons
        })
    
    return predictions, improved_escalations


def evaluate_escalation(golden, predictions, improved_escalations, baseline_escalations=None):
    """Evaluate escalation performance."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    true_escalation = golden['human_should_escalate'].tolist()
    
    # Pipeline baseline
    pipeline_escalations = [p['should_escalate'] for p in predictions]
    
    # Improved
    improved_esc = [e['should_escalate'] for e in improved_escalations]
    
    print("\n" + "=" * 60)
    print("ESCALATION COMPARISON")
    print("=" * 60)
    
    # Pipeline baseline
    acc = accuracy_score(true_escalation, pipeline_escalations)
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_escalation, pipeline_escalations, average='binary', zero_division=0
    )
    print(f"\nPipeline baseline:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall: {rec:.3f}")
    print(f"  F1: {f1:.3f}")
    
    # Improved
    acc2 = accuracy_score(true_escalation, improved_esc)
    prec2, rec2, f12, _ = precision_recall_fscore_support(
        true_escalation, improved_esc, average='binary', zero_division=0
    )
    print(f"\nImproved (v2):")
    print(f"  Accuracy: {acc2:.3f}")
    print(f"  Precision: {prec2:.3f}")
    print(f"  Recall: {rec2:.3f}")
    print(f"  F1: {f12:.3f}")
    
    # Improvement
    print(f"\nImprovement:")
    print(f"  Recall: {rec:.3f} -> {rec2:.3f} (+{rec2-rec:.3f})")
    print(f"  F1: {f1:.3f} -> {f12:.3f} (+{f12-f1:.3f})")
    print(f"  Accuracy: {acc:.3f} -> {acc2:.3f} ({acc2-acc:+.3f})")
    
    return {
        'baseline': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1},
        'improved': {'accuracy': acc2, 'precision': prec2, 'recall': rec2, 'f1': f12}
    }


def main():
    print("ESCALATION AUDIT")
    print("=" * 60)
    
    golden = analyze_escalation_patterns()
    predictions, improved_escalations = create_improved_escalation_v2(golden)
    metrics = evaluate_escalation(golden, predictions, improved_escalations)
    
    # Save results
    output = {
        'baseline_metrics': metrics['baseline'],
        'improved_metrics': metrics['improved'],
        'improvement': {
            'recall_delta': metrics['improved']['recall'] - metrics['baseline']['recall'],
            'f1_delta': metrics['improved']['f1'] - metrics['baseline']['f1'],
            'accuracy_delta': metrics['improved']['accuracy'] - metrics['baseline']['accuracy']
        }
    }
    
    output_path = Path("experiments/escalation_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
