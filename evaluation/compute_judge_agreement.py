"""
Compute agreement between human and LLM judge scores.

This script:
1. Loads completed human evaluation template (with human_scores filled in)
2. Loads LLM judge results
3. Computes Spearman correlation and weighted agreement
4. Reports per-dimension agreement metrics

Usage:
    python evaluation/compute_judge_agreement.py

    # With custom paths:
    python evaluation/compute_judge_agreement.py \
        --human-template evaluation/reply_quality_annotation/sample_template.json \
        --llm-results experiments/llm_judge_results.json
"""
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIMENSIONS = ['relevance', 'groundedness', 'correctness', 'completeness', 'tone', 'hallucination_risk']


def load_human_evaluations(path: str) -> List[Dict[str, Any]]:
    """Load completed human evaluation template."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Filter to completed entries
    completed = [entry for entry in data if entry.get('completed', False)]
    logger.info(f"Loaded {len(completed)} completed human evaluations (out of {len(data)} total)")
    return completed


def load_llm_results(path: str) -> Dict[str, Dict[str, int]]:
    """Load LLM judge results and build lookup by example_id."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    lookup = {}
    for item in data.get('per_example', []):
        eid = item.get('example_id', '')
        scores = item.get('scores', {})
        lookup[eid] = scores
    
    logger.info(f"Loaded LLM scores for {len(lookup)} examples")
    return lookup


def spearman_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    
    # Rank the data
    def rank(data):
        sorted_indices = sorted(range(n), key=lambda i: data[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_indices, 1):
            ranks[idx] = rank
        return ranks
    
    rank_x = rank(x)
    rank_y = rank(y)
    
    # Compute Pearson on ranks
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n
    
    numerator = sum((rx - mean_x) * (ry - mean_y) for rx, ry in zip(rank_x, rank_y))
    denom_x = sum((rx - mean_x) ** 2 for rx in rank_x) ** 0.5
    denom_y = sum((ry - mean_y) ** 2 for ry in rank_y) ** 0.5
    
    if denom_x == 0 or denom_y == 0:
        return 0.0
    
    return numerator / (denom_x * denom_y)


def compute_agreement(human_data: List[Dict[str, Any]], 
                      llm_lookup: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """Compute agreement metrics between human and LLM scores."""
    dimension_stats = {dim: {'human': [], 'llm': [], 'diffs': []} for dim in DIMENSIONS}
    total_comparisons = 0
    exact_matches = 0
    within_one = 0
    
    for entry in human_data:
        eid = entry.get('example_id', '')
        if eid not in llm_lookup:
            continue
        
        human_scores = entry.get('human_scores', {})
        llm_scores = llm_lookup[eid]
        
        for dim in DIMENSIONS:
            h_score = human_scores.get(dim)
            l_score = llm_scores.get(dim)
            
            if h_score is not None and l_score is not None:
                dimension_stats[dim]['human'].append(h_score)
                dimension_stats[dim]['llm'].append(l_score)
                dimension_stats[dim]['diffs'].append(abs(h_score - l_score))
                
                total_comparisons += 1
                if h_score == l_score:
                    exact_matches += 1
                if abs(h_score - l_score) <= 1:
                    within_one += 1
    
    # Compute per-dimension metrics
    dim_metrics = {}
    for dim in DIMENSIONS:
        stats = dimension_stats[dim]
        if len(stats['human']) > 0:
            spearman = spearman_correlation(stats['human'], stats['llm'])
            exact_agreement = sum(1 for d in stats['diffs'] if d == 0) / len(stats['diffs'])
            weighted_agreement = sum(1 for d in stats['diffs'] if d <= 1) / len(stats['diffs'])
            mean_diff = sum(stats['diffs']) / len(stats['diffs'])
            
            dim_metrics[dim] = {
                'spearman_correlation': round(spearman, 3),
                'exact_agreement': round(exact_agreement, 3),
                'weighted_agreement': round(weighted_agreement, 3),
                'mean_absolute_difference': round(mean_diff, 2),
                'num_comparisons': len(stats['human'])
            }
    
    # Overall metrics
    overall = {
        'exact_agreement_rate': round(exact_matches / total_comparisons, 3) if total_comparisons > 0 else 0.0,
        'weighted_agreement_rate': round(within_one / total_comparisons, 3) if total_comparisons > 0 else 0.0,
        'total_comparisons': total_comparisons,
        'num_examples': len(human_data)
    }
    
    # Identify disagreements
    disagreements = []
    for entry in human_data:
        eid = entry.get('example_id', '')
        if eid not in llm_lookup:
            continue
        
        human_scores = entry.get('human_scores', {})
        llm_scores = llm_lookup[eid]
        
        for dim in DIMENSIONS:
            h_score = human_scores.get(dim)
            l_score = llm_scores.get(dim)
            
            if h_score is not None and l_score is not None:
                diff = abs(h_score - l_score)
                if diff >= 2:
                    disagreements.append({
                        'example_id': eid,
                        'dimension': dim,
                        'human_score': h_score,
                        'llm_score': l_score,
                        'difference': diff,
                        'customer_message': entry.get('customer_message', '')[:100]
                    })
    
    return {
        'overall': overall,
        'per_dimension': dim_metrics,
        'disagreements': disagreements[:10]  # Top 10
    }


def main():
    parser = argparse.ArgumentParser(description="Compute judge-human agreement")
    parser.add_argument("--human-template", 
                       default="evaluation/reply_quality_annotation/sample_template.json",
                       help="Path to completed human evaluation template")
    parser.add_argument("--llm-results",
                       default="experiments/llm_judge_results.json",
                       help="Path to LLM judge results")
    parser.add_argument("--output",
                       default="experiments/judge_human_agreement.json",
                       help="Path to save agreement report")
    args = parser.parse_args()
    
    # Load data
    human_data = load_human_evaluations(args.human_template)
    llm_lookup = load_llm_results(args.llm_results)
    
    if not human_data:
        print("ERROR: No completed human evaluations found.")
        print("Please fill in human_scores in the sample_template.json file.")
        print("Then mark 'completed': true for each evaluated example.")
        sys.exit(1)
    
    # Compute agreement
    report = compute_agreement(human_data, llm_lookup)
    
    # Print summary
    print("\n" + "=" * 60)
    print("JUDGE-HUMAN AGREEMENT REPORT")
    print("=" * 60)
    print(f"Examples evaluated: {report['overall']['num_examples']}")
    print(f"Total comparisons: {report['overall']['total_comparisons']}")
    print(f"Exact agreement rate: {report['overall']['exact_agreement_rate']:.1%}")
    print(f"Weighted agreement rate (within 1 point): {report['overall']['weighted_agreement_rate']:.1%}")
    
    print("\nPer-dimension metrics:")
    for dim, metrics in report['per_dimension'].items():
        print(f"  {dim}:")
        print(f"    Spearman correlation: {metrics['spearman_correlation']:.3f}")
        print(f"    Exact agreement: {metrics['exact_agreement']:.1%}")
        print(f"    Weighted agreement: {metrics['weighted_agreement']:.1%}")
        print(f"    Mean absolute difference: {metrics['mean_absolute_difference']:.2f}")
    
    if report['disagreements']:
        print(f"\nTop disagreements ({len(report['disagreements'])} found):")
        for d in report['disagreements'][:5]:
            print(f"  Example {d['example_id']}, {d['dimension']}: "
                  f"Human={d['human_score']}, LLM={d['llm_score']}")
    
    # Save report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
