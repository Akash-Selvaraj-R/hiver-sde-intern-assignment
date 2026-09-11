"""
Failure analysis script for identifying and analyzing system failures.

Analyzes evaluation results to find:
1. Top failure modes
2. Real examples of failures
3. Expected vs actual results
4. Root causes and hypotheses
5. Proposed fixes
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """
    Analyzes system failures from evaluation results.
    """

    def __init__(self):
        self.failure_modes = {}

    def analyze_failures(self, golden_path: str, predictions_path: str = None,
                         pipeline_results: List[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive failure analysis.
        
        Args:
            golden_path: Path to golden set CSV
            predictions_path: Optional path to predictions JSON
            pipeline_results: Optional list of pipeline prediction results
            
        Returns:
            Dictionary with failure analysis results
        """
        # Load golden set
        golden_df = pd.read_csv(golden_path)
        
        # Load or use predictions
        if predictions_path and Path(predictions_path).exists():
            with open(predictions_path, 'r') as f:
                predictions = json.load(f)
            # Handle nested structure from run_eval.py
            if 'final' in predictions and 'predictions' in predictions['final']:
                final_preds = predictions['final']['predictions']
                pred_intents = final_preds.get('intents', [])
                pred_escalation = final_preds.get('escalation', [])
            else:
                pred_intents = predictions.get('intents', [])
                pred_escalation = predictions.get('escalation', [])
        elif pipeline_results:
            pred_intents = [r['intent']['name'] for r in pipeline_results]
            pred_escalation = [r['should_escalate'] for r in pipeline_results]
        else:
            logger.warning("No predictions available for failure analysis")
            return {}
        
        # Ensure same length
        n = min(len(golden_df), len(pred_intents))
        golden_df = golden_df.iloc[:n]
        pred_intents = pred_intents[:n]
        pred_escalation = pred_escalation[:n]
        
        # Identify failures
        intent_failures = []
        escalation_failures = []
        
        for i in range(n):
            true_intent = golden_df.iloc[i]['intent']
            pred_intent = pred_intents[i]
            true_escalate = bool(golden_df.iloc[i]['should_escalate'])
            pred_escalate = bool(pred_escalation[i]) if i < len(pred_escalation) else False
            
            if true_intent != pred_intent:
                intent_failures.append({
                    'index': i,
                    'customer_message': golden_df.iloc[i]['customer_message'],
                    'true_intent': true_intent,
                    'pred_intent': pred_intent,
                    'type': 'intent_misclassification'
                })
            
            if true_escalate != pred_escalate:
                escalation_failures.append({
                    'index': i,
                    'customer_message': golden_df.iloc[i]['customer_message'],
                    'true_escalate': true_escalate,
                    'pred_escalate': pred_escalate,
                    'type': 'escalation_error'
                })
        
        # Categorize failure modes
        failure_modes = self._categorize_failure_modes(intent_failures, golden_df)
        
        # Compute statistics
        total = n
        intent_errors = len(intent_failures)
        escalation_errors = len(escalation_failures)
        
        # Top confusion pairs
        confusion_pairs = Counter()
        for f in intent_failures:
            pair = (f['true_intent'], f['pred_intent'])
            confusion_pairs[pair] += 1
        
        analysis = {
            'total_examples': total,
            'intent_failures': intent_errors,
            'intent_error_rate': round(intent_errors / total, 3) if total > 0 else 0,
            'escalation_failures': escalation_errors,
            'escalation_error_rate': round(escalation_errors / total, 3) if total > 0 else 0,
            'top_confusion_pairs': [
                {'true': pair[0], 'predicted': pair[1], 'count': count}
                for pair, count in confusion_pairs.most_common(10)
            ],
            'failure_modes': failure_modes,
            'intent_failure_examples': intent_failures[:20],
            'escalation_failure_examples': escalation_failures[:20]
        }
        
        return analysis

    def _categorize_failure_modes(self, intent_failures: List[Dict],
                                   golden_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Categorize failures into named failure modes."""
        mode_counts = defaultdict(list)
        
        for failure in intent_failures:
            message = failure['customer_message'].lower()
            true_intent = failure['true_intent']
            pred_intent = failure['pred_intent']
            
            # Categorize based on message characteristics and Uber-specific intents
            if len(message.split()) <= 3:
                mode = 'short_noisy_message'
            elif true_intent == 'other' or pred_intent == 'other':
                mode = 'ambiguous_intent'
            elif true_intent in ['trip_fare_dispute', 'refund_request'] and \
                 pred_intent in ['trip_fare_dispute', 'refund_request']:
                mode = 'financial_intent_confusion'
            elif true_intent in ['account_access', 'app_functionality'] and \
                 pred_intent in ['account_access', 'app_functionality']:
                mode = 'access_vs_functionality_confusion'
            elif true_intent == 'ride_status' and pred_intent in ['trip_fare_dispute', 'app_functionality']:
                mode = 'ride_status_vs_other'
            elif any(word in message for word in ['but', 'however', 'also', 'and']):
                mode = 'multi_intent_message'
            else:
                mode = f'{true_intent}_confused_with_{pred_intent}'
            
            mode_counts[mode].append(failure)
        
        # Build failure mode summaries
        failure_modes = []
        for mode, examples in sorted(mode_counts.items(), key=lambda x: -len(x[1])):
            failure_modes.append({
                'name': mode,
                'count': len(examples),
                'examples': examples[:3],  # Top 3 examples
                'hypothesis': self._generate_hypothesis(mode),
                'proposed_fix': self._generate_fix(mode)
            })
        
        return failure_modes[:5]  # Top 5 failure modes

    def _generate_hypothesis(self, mode: str) -> str:
        """Generate hypothesis for why this failure mode occurs."""
        hypotheses = {
            'short_noisy_message': 'Very short messages lack sufficient context for accurate classification',
            'ambiguous_intent': 'Messages that don\'t clearly fit any intent category',
            'financial_intent_confusion': 'Fare disputes and refund requests share overlapping vocabulary (charges, money)',
            'access_vs_functionality_confusion': 'Account access and app functionality both involve "not working" language',
            'ride_status_vs_other': 'Ride issues can resemble general complaints or fare disputes without clear ride-status keywords',
            'multi_intent_message': 'Messages expressing multiple issues confuse the single-label classifier',
        }
        
        if mode in hypotheses:
            return hypotheses[mode]
        
        # Generic hypothesis for other modes
        parts = mode.split('_confused_with_')
        if len(parts) == 2:
            return f"The classifier struggles to distinguish {parts[0].replace('_', ' ')} from {parts[1].replace('_', ' ')}"
        
        return "Insufficient features or training data for this pattern"

    def _generate_fix(self, mode: str) -> str:
        """Generate proposed fix for this failure mode."""
        fixes = {
            'short_noisy_message': 'Add context from conversation thread or ask for clarification',
            'ambiguous_intent': 'Add an "ambiguous" escalation path or multi-label classification',
            'financial_intent_confusion': 'Add more specific fare/refund keywords or hierarchical classification',
            'access_vs_functionality_confusion': 'Add more specific account vs app keywords',
            'ride_status_vs_other': 'Better distinguish ride-specific issues from general complaints',
            'multi_intent_message': 'Implement multi-label classification or intent decomposition',
        }
        
        if mode in fixes:
            return fixes[mode]
        
        return "Collect more labeled data for this confusion pattern"

    def generate_failure_report(self, analysis: Dict[str, Any], output_path: str):
        """Generate a human-readable failure analysis report."""
        report_lines = [
            "# Failure Analysis Report\n",
            f"## Overview\n",
            f"- Total examples: {analysis.get('total_examples', 0)}",
            f"- Intent failures: {analysis.get('intent_failures', 0)} ({analysis.get('intent_error_rate', 0)*100:.1f}%)",
            f"- Escalation failures: {analysis.get('escalation_failures', 0)} ({analysis.get('escalation_error_rate', 0)*100:.1f}%)\n",
            "## Top Confusion Pairs\n",
            "| True Intent | Predicted Intent | Count |",
            "|-------------|-----------------|-------|"
        ]
        
        for pair in analysis.get('top_confusion_pairs', []):
            report_lines.append(f"| {pair['true']} | {pair['predicted']} | {pair['count']} |")
        
        report_lines.append("\n## Top 5 Failure Modes\n")
        
        for i, mode in enumerate(analysis.get('failure_modes', [])[:5], 1):
            report_lines.append(f"### {i}. {mode['name'].replace('_', ' ').title()}")
            report_lines.append(f"- **Count**: {mode['count']} examples")
            report_lines.append(f"- **Hypothesis**: {mode['hypothesis']}")
            report_lines.append(f"- **Proposed fix**: {mode['proposed_fix']}")
            
            if mode['examples']:
                example = mode['examples'][0]
                report_lines.append(f"- **Example**: \"{example.get('customer_message', '')[:80]}\"")
                report_lines.append(f"  - True: {example.get('true_intent')} -> Predicted: {example.get('pred_intent')}")
            report_lines.append("")
        
        # Write report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Failure report saved to {output_path}")


def main():
    """Main function to run failure analysis."""
    print("Running Failure Analysis...")
    print("=" * 50)
    
    analyzer = FailureAnalyzer()
    
    # Run analysis
    analysis = analyzer.analyze_failures(
        golden_path='evaluation/golden_set.csv',
        predictions_path='experiments/results.json'
    )
    
    if not analysis:
        print("No analysis results. Make sure evaluation has been run.")
        return
    
    # Print summary
    print(f"\nTotal examples: {analysis['total_examples']}")
    print(f"Intent failures: {analysis['intent_failures']} ({analysis['intent_error_rate']*100:.1f}%)")
    print(f"Escalation failures: {analysis['escalation_failures']} ({analysis['escalation_error_rate']*100:.1f}%)")
    
    print(f"\nTop confusion pairs:")
    for pair in analysis['top_confusion_pairs'][:5]:
        print(f"  {pair['true']} → {pair['predicted']}: {pair['count']}")
    
    print(f"\nTop failure modes:")
    for mode in analysis['failure_modes'][:5]:
        print(f"  {mode['name']}: {mode['count']} examples")
        print(f"    Hypothesis: {mode['hypothesis']}")
        print(f"    Fix: {mode['proposed_fix']}")
    
    # Generate report
    analyzer.generate_failure_report(analysis, 'evaluation/results/failure_analysis.md')
    print(f"\nDetailed report saved to evaluation/results/failure_analysis.md")
    
    # Save JSON
    with open('evaluation/results/failure_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print("JSON results saved to evaluation/results/failure_analysis.json")


if __name__ == "__main__":
    main()
