"""
Complete evaluation script for the customer support agent.
Runs baselines, final system, and generates comparison report.
"""
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import modules directly
from classifier import IntentClassifier
from retrieval import HistoricalRetriever
from generator import ResponseGenerator
from escalation import EscalationPolicy
from pipeline import SupportAgentPipeline
# Import baselines from evaluation directory
import importlib.util
_spec = importlib.util.spec_from_file_location("baselines", str(Path(__file__).parent / "baselines.py"))
_baselines = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baselines)
TrivialBaseline = _baselines.TrivialBaseline
TfIdfBaseline = _baselines.TfIdfBaseline
evaluate_baseline = _baselines.evaluate_baseline
_load_training_data_baselines = _baselines.load_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_golden_set():
    """Load the golden set evaluation data.
    
    Prefers the JSON golden set (200 examples) over the CSV (25 examples).
    """
    golden_json_path = Path("evaluation/golden_set.json")
    golden_csv_path = Path("evaluation/golden_set.csv")
    
    if golden_json_path.exists():
        with open(golden_json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        logger.info(f"Loaded golden set from JSON: {len(df)} examples")
        return df
    elif golden_csv_path.exists():
        df = pd.read_csv(golden_csv_path)
        logger.info(f"Loaded golden set from CSV: {len(df)} examples")
        return df
    else:
        raise FileNotFoundError(f"Golden set not found. Checked: {golden_json_path}, {golden_csv_path}")


def evaluate_system(pipeline, golden_df: pd.DataFrame) -> dict:
    """
    Evaluate the final system on the golden set.

    Args:
        pipeline: The fitted SupportAgentPipeline
        golden_df: DataFrame with golden set examples

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating final system on golden set...")

    # Get test data
    test_messages = golden_df['customer_message'].tolist()
    true_intents = golden_df['intent'].tolist()
    true_escalation = golden_df['should_escalate'].tolist()

    # Make predictions
    logger.info("Making predictions...")
    pred_results = pipeline.predict_batch(test_messages)

    # Extract predictions
    pred_intents = [result['intent']['name'] for result in pred_results]
    pred_confidences = [result['intent']['confidence'] for result in pred_results]
    pred_escalation = [result['should_escalate'] for result in pred_results]
    pred_escalation_reasons = [result['escalation_reason'] for result in pred_results]

    # Calculate intent metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    intent_accuracy = accuracy_score(true_intents, pred_intents)
    intent_f1_macro = f1_score(true_intents, pred_intents, average='macro')
    intent_f1_weighted = f1_score(true_intents, pred_intents, average='weighted')

    # Per-intent metrics
    try:
        intent_report = classification_report(true_intents, pred_intents, output_dict=True, zero_division=0)
    except Exception as e:
        logger.warning(f"Could not generate classification report: {e}")
        intent_report = {}

    # Calculate escalation metrics
    try:
        escalation_accuracy = accuracy_score(true_escalation, pred_escalation)
        escalation_f1 = f1_score(true_escalation, pred_escalation, zero_division=0)
    except Exception as e:
        logger.warning(f"Could not calculate escalation metrics: {e}")
        escalation_accuracy = 0.0
        escalation_f1 = 0.0

    # Calculate reply quality would require LLM judge - placeholder for now
    # In a full implementation, this would use the LLM judge from the spec
    reply_score_placeholder = 0.0  # Would be replaced with actual LLM judging

    # Calculate safe auto-handle rate (would need human evaluation)
    # For now, we'll approximate based on escalation correctness
    safe_auto_handle_placeholder = 0.0  # Would be replaced with human evaluation

    return {
        'intent_accuracy': intent_accuracy,
        'intent_f1_macro': intent_f1_macro,
        'intent_f1_weighted': intent_f1_weighted,
        'intent_classification_report': intent_report,
        'escalation_accuracy': escalation_accuracy,
        'escalation_f1': escalation_f1,
        'reply_score_placeholder': reply_score_placeholder,
        'safe_auto_handle_placeholder': safe_auto_handle_placeholder,
        'predictions': {
            'intents': pred_intents,
            'confidences': pred_confidences,
            'escalation': pred_escalation,
            'reasons': pred_escalation_reasons
        }
    }


def run_complete_evaluation():
    """Run complete evaluation including baselines and final system."""
    print("=" * 60)
    print("HIVER SUPPORT AGENT EVALUATION")
    print("=" * 60)

    # Load golden set
    try:
        golden_df = load_golden_set()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("Please run evaluation/create_golden_set.py first to generate the golden set")
        return

    print(f"\nGolden set loaded: {len(golden_df)} examples")
    print(f"Escalation rate in golden set: {golden_df['should_escalate'].mean()*100:.1f}%")

    # Load training data (avoiding golden set to prevent leakage)
    print("\nLoading training data...")
    train_texts, train_labels, retrieval_texts = _load_training_data_baselines()
    print(f"Training examples: {len(train_texts)}")

    # Load historical cases for retrieval
    print("Loading historical cases...")
    from retrieval import load_historical_cases
    historical_cases = load_historical_cases()
    print(f"Historical cases: {len(historical_cases)}")

    # Evaluate baselines
    print("\n" + "="*50)
    print("EVALUATING BASELINES")
    print("="*50)

    # Baseline 0: Trivial
    print("\n--- Training Trivial Baseline ---")
    trivial_baseline = TrivialBaseline()
    trivial_baseline.fit(train_texts, train_labels)

    print("\n--- Evaluating Trivial Baseline ---")
    trivial_results = evaluate_baseline(trivial_baseline, str(Path("evaluation/golden_set.json")))

    # Baseline 1: TF-IDF
    print("\n--- Training TF-IDF Baseline ---")
    tfidf_baseline = TfIdfBaseline()
    tfidf_baseline.fit(train_texts, train_labels, retrieval_texts)  # Use retrieval texts for retrieval

    print("\n--- Evaluating TF-IDF Baseline ---")
    tfidf_results = evaluate_baseline(tfidf_baseline, str(Path("evaluation/golden_set.json")))

    # Final System
    print("\n" + "="*50)
    print("EVALUATING FINAL SYSTEM")
    print("="*50)

    print("\n--- Training Final System ---")
    pipeline = SupportAgentPipeline()
    pipeline.fit(
        training_data=(train_texts, train_labels),
        historical_cases=historical_cases
    )

    print("\n--- Evaluating Final System ---")
    final_results = evaluate_system(pipeline, golden_df)

    # Print results table
    print("\n" + "="*70)
    print("FINAL EVALUATION RESULTS")
    print("="*70)
    print(f"{'System':<20} {'Intent Macro-F1':<15} {'Escalation F1':<15} {'Reply Score':<12} {'Safe Auto-Handle':<18}")
    print("-" * 70)
    print(f"{'Trivial':<20} {trivial_results['intent_f1_macro']:<15.3f} {trivial_results['escalation_f1']:<15.3f} {'N/A':<12} {'N/A':<18}")
    print(f"{'TF-IDF Baseline':<20} {tfidf_results['intent_f1_macro']:<15.3f} {tfidf_results['escalation_f1']:<15.3f} {'N/A':<12} {'N/A':<18}")
    print(f"{'Final System':<20} {final_results['intent_f1_macro']:<15.3f} {final_results['escalation_f1']:<15.3f} {final_results['reply_score_placeholder']:<12.3f} {final_results['safe_auto_handle_placeholder']:<18.3f}")

    # Save detailed results
    results_dir = Path("experiments")
    results_dir.mkdir(exist_ok=True)

    results = {
        'trivial': trivial_results,
        'tfidf': tfidf_results,
        'final': final_results,
        'evaluation_timestamp': datetime.now().isoformat(),
        'golden_set_size': len(golden_df),
        'training_set_size': len(train_texts),
        'historical_cases_count': len(historical_cases)
    }

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    results_file = results_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    # Also save to CSV for easy viewing
    csv_file = results_dir / "results.csv"
    csv_data = []
    for system_name, system_results in [('trivial', trivial_results), ('tfidf', tfidf_results), ('final', final_results)]:
        row = {
            'experiment': system_name,
            'model': system_name,
            'dataset': 'golden_set',
            'accuracy': system_results.get('intent_accuracy', 0.0),
            'macro_f1': system_results.get('intent_f1_macro', 0.0),
            'weighted_f1': system_results.get('intent_f1_weighted', 0.0),
            'escalation_precision': 0.0,  # Placeholder
            'escalation_recall': system_results.get('escalation_accuracy', 0.0),  # Approximation
            'escalation_f1': system_results.get('escalation_f1', 0.0),
            'reply_score': system_results.get('reply_score_placeholder', 0.0) if system_name == 'final' else 0.0,
            'safe_auto_handle_rate': system_results.get('safe_auto_handle_placeholder', 0.0) if system_name == 'final' else 0.0,
            'notes': 'Baseline' if system_name in ['trivial', 'tfidf'] else 'Final system with retrieval and generation'
        }
        csv_data.append(row)

    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_file, index=False)

    print(f"\nDetailed results saved to:")
    print(f"  JSON: {results_file}")
    print(f"  CSV: {csv_file}")

    # Print some example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS FROM FINAL SYSTEM")
    print("="*50)

    example_indices = [0, 5, 10, 15, 20] if len(golden_df) > 20 else list(range(min(5, len(golden_df))))
    for idx in example_indices:
        if idx < len(golden_df):
            row = golden_df.iloc[idx]
            message = row['customer_message']
            true_intent = row['intent']
            true_escalate = row['should_escalate']

            result = pipeline.predict(message)
            pred_intent = result['intent']['name']
            pred_confidence = result['intent']['confidence']
            pred_escalate = result['should_escalate']

            print(f"\nExample {idx+1}:")
            print(f"  Message: {message[:60]}{'...' if len(message) > 60 else ''}")
            print(f"  True Intent: {true_intent} | Pred: {pred_intent} ({pred_confidence:.3f})")
            print(f"  True Escalate: {true_escalate} | Pred: {pred_escalate}")
            if result['escalation_reason']:
                print(f"  Reason: {result['escalation_reason']}")
            print(f"  Reply: {result['reply'][:80]}{'...' if len(result['reply']) > 80 else ''}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

    return results


if __name__ == "__main__":
    run_complete_evaluation()