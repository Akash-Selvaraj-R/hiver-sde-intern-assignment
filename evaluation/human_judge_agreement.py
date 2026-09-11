"""
Human/Judge Agreement framework for validating LLM-as-judge scores.

This module:
1. Samples a subset of judged replies for human evaluation
2. Provides a template for human scoring
3. Computes agreement metrics between human and LLM judge
4. Reports exact agreement, correlation, and disagreement examples
"""
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42


@dataclass
class HumanJudgment:
    """A single human judgment for a reply."""
    example_id: str
    dimension: str
    score: int  # 1-5
    notes: str = ''


@dataclass
class AgreementReport:
    """Report comparing human and LLM judge scores."""
    num_examples: int
    dimensions: Dict[str, Dict[str, float]]
    exact_agreement_rate: float
    correlation: float
    disagreement_examples: List[Dict[str, Any]]


class HumanJudgeAgreement:
    """
    Framework for comparing human and LLM judge scores.
    """

    DIMENSIONS = ['relevance', 'groundedness', 'correctness', 'completeness', 'tone', 'hallucination_risk']

    def __init__(self, sample_size: int = 30):
        """
        Initialize the agreement framework.
        
        Args:
            sample_size: Number of examples to sample for human evaluation
        """
        self.sample_size = sample_size
        random.seed(SEED)

    def sample_for_human_evaluation(self, judged_results: List[Dict[str, Any]],
                                     output_path: str = 'evaluation/results/human_eval_template.json') -> List[Dict[str, Any]]:
        """
        Sample a subset of judged replies for human evaluation.
        Stratified sampling to ensure representation across quality levels.
        
        Args:
            judged_results: List of judged reply dictionaries
            output_path: Where to save the template
            
        Returns:
            List of examples for human evaluation
        """
        if len(judged_results) <= self.sample_size:
            sampled = judged_results
        else:
            # Stratify by overall score buckets
            sorted_results = sorted(judged_results, key=lambda x: x.get('overall_score', 3))
            n = len(sorted_results)
            
            # Divide into quality buckets
            bucket_size = max(1, n // 3)
            low_quality = sorted_results[:bucket_size]
            mid_quality = sorted_results[bucket_size:2*bucket_size]
            high_quality = sorted_results[2*bucket_size:]
            
            # Sample proportionally
            per_bucket = self.sample_size // 3
            remainder = self.sample_size - per_bucket * 3
            
            sampled = []
            sampled.extend(random.sample(low_quality, min(per_bucket + (1 if remainder > 0 else 0), len(low_quality))))
            sampled.extend(random.sample(mid_quality, min(per_bucket + (1 if remainder > 1 else 0), len(mid_quality))))
            sampled.extend(random.sample(high_quality, min(per_bucket, len(high_quality))))
            
            # Fill remaining if needed
            remaining_ids = {s['example_id'] for s in sampled}
            remaining = [r for r in judged_results if r['example_id'] not in remaining_ids]
            while len(sampled) < self.sample_size and remaining:
                sampled.append(remaining.pop(random.randint(0, len(remaining) - 1)))

        # Create human evaluation template
        template = []
        for item in sampled[:self.sample_size]:
            template_entry = {
                'example_id': item.get('example_id', 'unknown'),
                'customer_message': item.get('customer_message', ''),
                'reply': item.get('reply', ''),
                'evidence': item.get('evidence', ''),
                'llm_scores': {s['dimension']: s['score'] for s in item.get('scores', [])},
                'human_scores': {dim: None for dim in self.DIMENSIONS},
                'human_notes': {dim: '' for dim in self.DIMENSIONS},
                'completed': False
            }
            template.append(template_entry)

        # Save template
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Created human evaluation template with {len(template)} examples at {output_path}")
        logger.info(f"Instructions: Have a human score each reply on dimensions {self.DIMENSIONS}")
        logger.info(f"Then run compute_agreement() with the completed template")
        
        return template

    def compute_agreement(self, human_template_path: str, 
                          llm_results_path: str) -> AgreementReport:
        """
        Compute agreement between human and LLM judge scores.
        
        Args:
            human_template_path: Path to completed human evaluation template
            llm_results_path: Path to LLM judge results
            
        Returns:
            AgreementReport with detailed metrics
        """
        # Load human evaluations
        with open(human_template_path, 'r') as f:
            human_data = json.load(f)
        
        # Load LLM results
        with open(llm_results_path, 'r') as f:
            llm_data = json.load(f)
        
        # Build LLM lookup by example_id
        llm_lookup = {}
        for item in llm_data:
            eid = item.get('example_id', '')
            scores = {s['dimension']: s['score'] for s in item.get('scores', [])}
            llm_lookup[eid] = scores

        # Compare scores
        dimension_agreements = {dim: {'matches': 0, 'total': 0, 'diffs': []} for dim in self.DIMENSIONS}
        total_matches = 0
        total_comparisons = 0
        disagreement_examples = []

        for entry in human_data:
            if not entry.get('completed', False):
                continue
            
            eid = entry['example_id']
            if eid not in llm_lookup:
                continue
            
            llm_scores = llm_lookup[eid]
            human_scores = entry.get('human_scores', {})
            
            for dim in self.DIMENSIONS:
                human_score = human_scores.get(dim)
                llm_score = llm_scores.get(dim)
                
                if human_score is not None and llm_score is not None:
                    dimension_agreements[dim]['total'] += 1
                    total_comparisons += 1
                    
                    diff = abs(human_score - llm_score)
                    dimension_agreements[dim]['diffs'].append(diff)
                    
                    if diff == 0:
                        dimension_agreements[dim]['matches'] += 1
                        total_matches += 1
                    elif diff >= 2:
                        disagreement_examples.append({
                            'example_id': eid,
                            'dimension': dim,
                            'human_score': human_score,
                            'llm_score': llm_score,
                            'difference': diff,
                            'customer_message': entry.get('customer_message', '')[:80],
                            'reply': entry.get('reply', '')[:80]
                        })

        # Compute metrics
        exact_agreement = total_matches / total_comparisons if total_comparisons > 0 else 0.0
        
        # Compute correlation (Pearson-like, simplified)
        all_human = []
        all_llm = []
        for entry in human_data:
            if not entry.get('completed', False):
                continue
            eid = entry['example_id']
            if eid not in llm_lookup:
                continue
            for dim in self.DIMENSIONS:
                h = entry.get('human_scores', {}).get(dim)
                l = llm_lookup[eid].get(dim)
                if h is not None and l is not None:
                    all_human.append(h)
                    all_llm.append(l)
        
        correlation = self._pearson_correlation(all_human, all_llm) if len(all_human) > 1 else 0.0

        # Build dimension-level report
        dim_report = {}
        for dim, data in dimension_agreements.items():
            if data['total'] > 0:
                dim_report[dim] = {
                    'exact_agreement': round(data['matches'] / data['total'], 3),
                    'mean_absolute_difference': round(sum(data['diffs']) / len(data['diffs']), 2) if data['diffs'] else 0,
                    'total_comparisons': data['total']
                }

        return AgreementReport(
            num_examples=len([e for e in human_data if e.get('completed', False)]),
            dimensions=dim_report,
            exact_agreement_rate=round(exact_agreement, 3),
            correlation=round(correlation, 3),
            disagreement_examples=disagreement_examples[:10]  # Top 10 disagreements
        )

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        
        if denom_x == 0 or denom_y == 0:
            return 0.0
        
        return numerator / (denom_x * denom_y)

    def save_agreement_report(self, report: AgreementReport, output_path: str):
        """Save agreement report to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        logger.info(f"Saved agreement report to {output_path}")


def create_mock_human_judgments(llm_results_path: str, output_path: str,
                                 noise_level: float = 0.3) -> str:
    """
    Create mock human judgments for demonstration.
    In production, these would come from actual human evaluators.
    
    Args:
        llm_results_path: Path to LLM judge results
        output_path: Where to save mock human judgments
        noise_level: How much to deviate from LLM scores (0-1)
        
    Returns:
        Path to saved mock human judgments
    """
    with open(llm_results_path, 'r') as f:
        llm_data = json.load(f)
    
    random.seed(SEED)
    
    mock_template = []
    for item in llm_data[:30]:  # Sample 30 for human eval
        llm_scores = {s['dimension']: s['score'] for s in item.get('scores', [])}
        
        # Create human scores by adding small random noise to LLM scores
        human_scores = {}
        for dim, llm_score in llm_scores.items():
            # Add noise: ±0-1 points with probability proportional to noise_level
            noise = 0
            if random.random() < noise_level:
                noise = random.choice([-1, 0, 0, 1])  # Mostly 0, sometimes ±1
            human_score = max(1, min(5, llm_score + noise))
            human_scores[dim] = human_score
        
        mock_template.append({
            'example_id': item.get('example_id', 'unknown'),
            'customer_message': item.get('customer_message', ''),
            'reply': item.get('reply', ''),
            'evidence': item.get('evidence', ''),
            'llm_scores': llm_scores,
            'human_scores': human_scores,
            'human_notes': {dim: 'Mock human judgment' for dim in human_scores},
            'completed': True
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mock_template, f, indent=2)
    
    logger.info(f"Created {len(mock_template)} mock human judgments at {output_path}")
    return output_path


def main():
    """Main function to demonstrate human/judge agreement."""
    print("Human/Judge Agreement Framework")
    print("=" * 50)
    
    # Step 1: Check if we have LLM judge results
    llm_results_path = 'evaluation/results/judge_results.json'
    
    if not Path(llm_results_path).exists():
        print(f"LLM judge results not found at {llm_results_path}")
        print("Running LLM judge first...")
        from llm_judge import LLMJudge, save_judgment_results
        
        # Create some mock examples
        examples = [
            {'example_id': str(i), 'customer_message': f'Customer message {i}', 
             'reply': f'Reply to message {i}', 'evidence': f'Evidence {i}'}
            for i in range(10)
        ]
        
        judge = LLMJudge(mode='mock')
        results = judge.judge_batch(examples)
        save_judgment_results(results, llm_results_path)
    
    # Step 2: Create human evaluation template
    print("\nCreating human evaluation template...")
    agreement = HumanJudgeAgreement(sample_size=30)
    
    with open(llm_results_path, 'r') as f:
        llm_data = json.load(f)
    
    template = agreement.sample_for_human_evaluation(llm_data)
    print(f"Template created with {len(template)} examples")
    
    # Step 3: Create mock human judgments (for demonstration)
    print("\nCreating mock human judgments (for demonstration)...")
    mock_path = 'evaluation/results/human_eval_completed.json'
    create_mock_human_judgments(llm_results_path, mock_path)
    
    # Step 4: Compute agreement
    print("\nComputing agreement...")
    report = agreement.compute_agreement(mock_path, llm_results_path)
    
    print(f"\n=== Agreement Report ===")
    print(f"Examples evaluated: {report.num_examples}")
    print(f"Exact agreement rate: {report.exact_agreement_rate:.1%}")
    print(f"Correlation: {report.correlation:.3f}")
    
    print(f"\nPer-dimension agreement:")
    for dim, metrics in report.dimensions.items():
        print(f"  {dim}: {metrics['exact_agreement']:.1%} (MAE: {metrics['mean_absolute_difference']:.2f})")
    
    if report.disagreement_examples:
        print(f"\nTop disagreements:")
        for ex in report.disagreement_examples[:3]:
            print(f"  Example {ex['example_id']}, {ex['dimension']}: "
                  f"Human={ex['human_score']}, LLM={ex['llm_score']}")
    
    # Save report
    agreement.save_agreement_report(report, 'evaluation/results/agreement_report.json')
    print(f"\nReport saved to evaluation/results/agreement_report.json")


if __name__ == "__main__":
    main()
