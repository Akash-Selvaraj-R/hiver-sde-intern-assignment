"""
Complete real-data evaluation for Uber_Support customer support agent.

Runs:
1. Trivial (majority class) baseline
2. TF-IDF + Logistic Regression baseline
3. Full pipeline (classifier + retrieval + generation + escalation)
4. Failure analysis
5. LLM judge check
"""
import pandas as pd
import numpy as np
import yaml
import json
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from classifier import IntentClassifier, _pseudo_label_batch
from retrieval import HistoricalRetriever
from generator import ResponseGenerator
from escalation import EscalationPolicy
from pipeline import SupportAgentPipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from baselines import TrivialBaseline, TfIdfBaseline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_real_data():
    """Load real train/dev/golden splits."""
    train_df = pd.read_csv('data/processed/train.csv')
    dev_df = pd.read_csv('data/processed/dev.csv')
    golden_df = pd.read_csv('data/processed/golden_set.csv')
    return train_df, dev_df, golden_df


def prepare_training_data(train_df):
    """Prepare training texts and labels for classifier."""
    texts = train_df['customer_message'].tolist()
    # Use auto_intent for pseudo-labeling (NOT hand-labelled)
    if 'intent' in train_df.columns:
        labels = train_df['intent'].tolist()
    else:
        labels = _pseudo_label_batch(texts)
    return texts, labels


def prepare_golden_data(golden_df):
    """Prepare golden set for evaluation."""
    texts = golden_df['customer_message'].tolist()
    # Golden set has auto_intent (NOT hand-labelled)
    if 'auto_intent' in golden_df.columns:
        true_intents = golden_df['auto_intent'].tolist()
    elif 'intent' in golden_df.columns:
        true_intents = golden_df['intent'].tolist()
    else:
        true_intents = _pseudo_label_batch(texts)

    if 'auto_should_escalate' in golden_df.columns:
        true_escalation = golden_df['auto_should_escalate'].tolist()
    elif 'should_escalate' in golden_df.columns:
        true_escalation = golden_df['should_escalate'].tolist()
    else:
        true_escalation = [False] * len(texts)

    return texts, true_intents, true_escalation


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_trivial_baseline(train_texts, train_labels, golden_texts, true_intents, true_escalation):
    """Run trivial (majority class) baseline."""
    logger.info("Running trivial baseline...")
    baseline = TrivialBaseline()
    baseline.fit(train_texts, train_labels)

    pred_intents = baseline.predict_intent(golden_texts)
    pred_escalation = baseline.predict_escalation(golden_texts)

    intent_acc = accuracy_score(true_intents, pred_intents)
    intent_f1_macro = f1_score(true_intents, pred_intents, average='macro', zero_division=0)
    intent_f1_weighted = f1_score(true_intents, pred_intents, average='weighted', zero_division=0)
    esc_f1 = f1_score(true_escalation, pred_escalation, zero_division=0)
    esc_acc = accuracy_score(true_escalation, pred_escalation)

    try:
        report = classification_report(true_intents, pred_intents, output_dict=True, zero_division=0)
    except:
        report = {}

    return {
        'name': 'Trivial (Majority Class)',
        'intent_accuracy': intent_acc,
        'intent_f1_macro': intent_f1_macro,
        'intent_f1_weighted': intent_f1_weighted,
        'escalation_f1': esc_f1,
        'escalation_accuracy': esc_acc,
        'classification_report': report,
        'predictions': {'intents': pred_intents, 'escalation': pred_escalation}
    }


def run_tfidf_baseline(train_texts, train_labels, golden_texts, true_intents, true_escalation):
    """Run TF-IDF + LogReg baseline."""
    logger.info("Running TF-IDF baseline...")
    baseline = TfIdfBaseline()
    baseline.fit(train_texts, train_labels, train_texts)

    pred_intents = baseline.predict_intent(golden_texts)
    pred_escalation = baseline.predict_escalation(golden_texts)

    intent_acc = accuracy_score(true_intents, pred_intents)
    intent_f1_macro = f1_score(true_intents, pred_intents, average='macro', zero_division=0)
    intent_f1_weighted = f1_score(true_intents, pred_intents, average='weighted', zero_division=0)
    esc_f1 = f1_score(true_escalation, pred_escalation, zero_division=0)
    esc_acc = accuracy_score(true_escalation, pred_escalation)

    try:
        report = classification_report(true_intents, pred_intents, output_dict=True, zero_division=0)
    except:
        report = {}

    return {
        'name': 'TF-IDF + LogReg',
        'intent_accuracy': intent_acc,
        'intent_f1_macro': intent_f1_macro,
        'intent_f1_weighted': intent_f1_weighted,
        'escalation_f1': esc_f1,
        'escalation_accuracy': esc_acc,
        'classification_report': report,
        'predictions': {'intents': pred_intents, 'escalation': pred_escalation}
    }


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(train_texts, train_labels, golden_texts, true_intents, true_escalation):
    """Run the full pipeline (classifier + retrieval + generation + escalation)."""
    logger.info("Running full pipeline...")

    # Build historical cases from training data
    historical_cases = []
    for text, label in zip(train_texts, train_labels):
        historical_cases.append({
            'conversation_id': f'case_{len(historical_cases)}',
            'customer_message': text,
            'brand_response': '',
            'resolution': f'Resolved {label.replace("_", " ")} issue',
            'intent': label,
        })

    # Initialize and fit pipeline
    pipeline = SupportAgentPipeline()
    pipeline.fit(
        training_data=(train_texts, train_labels),
        historical_cases=historical_cases
    )

    # Predict on golden set
    pred_results = pipeline.predict_batch(golden_texts)

    pred_intents = [r['intent']['name'] for r in pred_results]
    pred_confidences = [r['intent']['confidence'] for r in pred_results]
    pred_escalation = [r['should_escalate'] for r in pred_results]
    pred_replies = [r['reply'] for r in pred_results]
    pred_escalation_reasons = [r['escalation_reason'] for r in pred_results]

    intent_acc = accuracy_score(true_intents, pred_intents)
    intent_f1_macro = f1_score(true_intents, pred_intents, average='macro', zero_division=0)
    intent_f1_weighted = f1_score(true_intents, pred_intents, average='weighted', zero_division=0)
    esc_f1 = f1_score(true_escalation, pred_escalation, zero_division=0)
    esc_acc = accuracy_score(true_escalation, pred_escalation)

    try:
        report = classification_report(true_intents, pred_intents, output_dict=True, zero_division=0)
    except:
        report = {}

    return {
        'name': 'Full Pipeline',
        'intent_accuracy': intent_acc,
        'intent_f1_macro': intent_f1_macro,
        'intent_f1_weighted': intent_f1_weighted,
        'escalation_f1': esc_f1,
        'escalation_accuracy': esc_acc,
        'classification_report': report,
        'predictions': {
            'intents': pred_intents,
            'confidences': pred_confidences,
            'escalation': pred_escalation,
            'replies': pred_replies,
            'escalation_reasons': pred_escalation_reasons,
        },
        'pipeline': pipeline,
    }


# ---------------------------------------------------------------------------
# Failure Analysis
# ---------------------------------------------------------------------------

def run_failure_analysis(true_intents, pred_intents, true_escalation, pred_escalation, golden_df):
    """Analyze failures."""
    logger.info("Running failure analysis...")

    n = len(true_intents)
    intent_failures = []
    confusion_pairs = Counter()

    for i in range(n):
        if true_intents[i] != pred_intents[i]:
            confusion_pairs[(true_intents[i], pred_intents[i])] += 1
            intent_failures.append({
                'index': i,
                'message': golden_df.iloc[i]['customer_message'][:120],
                'true_intent': true_intents[i],
                'pred_intent': pred_intents[i],
            })

    escalation_failures = []
    for i in range(min(len(true_escalation), len(pred_escalation))):
        if true_escalation[i] != pred_escalation[i]:
            escalation_failures.append({
                'index': i,
                'message': golden_df.iloc[i]['customer_message'][:120],
                'true_escalate': true_escalation[i],
                'pred_escalate': pred_escalation[i],
            })

    # Top 5 failure modes
    failure_modes = []
    for (true_i, pred_i), count in confusion_pairs.most_common(5):
        failure_modes.append({
            'name': f'{true_i} confused with {pred_i}',
            'count': count,
            'true_intent': true_i,
            'pred_intent': pred_i,
            'example': next((f['message'] for f in intent_failures
                           if f['true_intent'] == true_i and f['pred_intent'] == pred_i), ''),
        })

    return {
        'total': n,
        'intent_failures': len(intent_failures),
        'intent_error_rate': len(intent_failures) / n if n > 0 else 0,
        'escalation_failures': len(escalation_failures),
        'top_confusion_pairs': [
            {'true': p[0], 'predicted': p[1], 'count': c}
            for p, c in confusion_pairs.most_common(10)
        ],
        'top_5_failure_modes': failure_modes,
        'intent_failure_examples': intent_failures[:20],
    }


# ---------------------------------------------------------------------------
# LLM Judge Check
# ---------------------------------------------------------------------------

def check_llm_judge():
    """Check if LLM judge is available."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return {'available': True, 'mode': 'openai', 'note': 'OpenAI API key configured'}
    else:
        return {'available': False, 'mode': 'mock', 'note': 'No OPENAI_API_KEY found. LLM judge NOT RUN. Set OPENAI_API_KEY to enable real LLM judging.'}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("REAL DATA EVALUATION — Uber_Support")
    print("=" * 70)

    # Load data
    train_df, dev_df, golden_df = load_real_data()
    train_texts, train_labels = prepare_training_data(train_df)
    golden_texts, true_intents, true_escalation = prepare_golden_data(golden_df)

    print(f"\nDataset:")
    print(f"  Train: {len(train_df):,} examples")
    print(f"  Dev: {len(dev_df):,} examples")
    print(f"  Golden: {len(golden_df):,} examples")
    print(f"  Golden intent distribution:")
    for intent, count in Counter(true_intents).most_common():
        print(f"    {intent}: {count} ({count/len(true_intents)*100:.1f}%)")
    print(f"  Golden escalation rate: {sum(true_escalation)/len(true_escalation)*100:.1f}%")

    # Run baselines
    results = {}

    trivial = run_trivial_baseline(train_texts, train_labels, golden_texts, true_intents, true_escalation)
    results['trivial'] = trivial

    tfidf = run_tfidf_baseline(train_texts, train_labels, golden_texts, true_intents, true_escalation)
    results['tfidf'] = tfidf

    full = run_full_pipeline(train_texts, train_labels, golden_texts, true_intents, true_escalation)
    results['final'] = full

    # Print results table
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'System':<25} {'Intent Acc':<12} {'Macro-F1':<12} {'Weighted-F1':<13} {'Esc F1':<10}")
    print("-" * 80)
    for name, r in [('Trivial', trivial), ('TF-IDF', tfidf), ('Full Pipeline', full)]:
        print(f"{name:<25} {r['intent_accuracy']:<12.3f} {r['intent_f1_macro']:<12.3f} "
              f"{r['intent_f1_weighted']:<13.3f} {r['escalation_f1']:<10.3f}")

    # Failure analysis
    failure = run_failure_analysis(
        true_intents, full['predictions']['intents'],
        true_escalation, full['predictions']['escalation'],
        golden_df
    )

    print(f"\n--- Failure Analysis ---")
    print(f"Intent errors: {failure['intent_failures']}/{failure['total']} "
          f"({failure['intent_error_rate']*100:.1f}%)")
    print(f"Escalation errors: {failure['escalation_failures']}/{failure['total']}")
    print(f"\nTop 5 failure modes:")
    for i, mode in enumerate(failure['top_5_failure_modes'], 1):
        print(f"  {i}. {mode['name']} ({mode['count']} examples)")
        if mode['example']:
            print(f"     Example: \"{mode['example'][:80]}\"")

    # LLM Judge
    judge_status = check_llm_judge()
    print(f"\n--- LLM Judge ---")
    print(f"Mode: {judge_status['mode']}")
    print(f"Note: {judge_status['note']}")

    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)

    # Convert numpy types
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    save_results = {
        'brand': 'Uber_Support',
        'data_source': 'kaggle_customer_support_on_twitter',
        'golden_set_labels': 'auto_generated_NOT_hand_labelled',
        'evaluation_timestamp': datetime.now().isoformat(),
        'train_size': len(train_df),
        'dev_size': len(dev_df),
        'golden_size': len(golden_df),
        'trivial': {k: v for k, v in trivial.items() if k != 'predictions'},
        'tfidf': {k: v for k, v in tfidf.items() if k != 'predictions'},
        'final': {k: v for k, v in full.items() if k not in ('predictions', 'pipeline')},
        'failure_analysis': failure,
        'llm_judge': judge_status,
    }

    with open(output_dir / 'real_results.json', 'w') as f:
        json.dump(to_serializable(save_results), f, indent=2)
    print(f"\nResults saved to {output_dir / 'real_results.json'}")

    # Save failure report
    failure_lines = [
        "# Failure Analysis Report (Real Data — Uber_Support)\n",
        f"## Overview\n",
        f"- Total examples: {failure['total']}",
        f"- Intent failures: {failure['intent_failures']} ({failure['intent_error_rate']*100:.1f}%)",
        f"- Escalation failures: {failure['escalation_failures']}\n",
        "## Top Confusion Pairs\n",
        "| True Intent | Predicted Intent | Count |",
        "|-------------|-----------------|-------|"
    ]
    for p in failure['top_confusion_pairs']:
        failure_lines.append(f"| {p['true']} | {p['predicted']} | {p['count']} |")

    failure_lines.append("\n## Top 5 Failure Modes\n")
    for i, mode in enumerate(failure['top_5_failure_modes'], 1):
        failure_lines.append(f"### {i}. {mode['name']}")
        failure_lines.append(f"- Count: {mode['count']}")
        failure_lines.append(f"- Example: \"{mode['example'][:100]}\"\n")

    with open(output_dir / 'failure_analysis_real.md', 'w') as f:
        f.write('\n'.join(failure_lines))

    print(f"Failure report saved to {output_dir / 'failure_analysis_real.md'}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
