"""Run all evaluation components."""
import sys
sys.path.append('evaluation')
sys.path.append('src')

from llm_judge import LLMJudge, save_judgment_results
from human_judge_agreement import HumanJudgeAgreement, create_mock_human_judgments
from failure_analysis import FailureAnalyzer
import json

# 1. Run LLM Judge
print("=== Running LLM Judge ===")
with open('evaluation/results/examples_for_judge.json', 'r') as f:
    examples = json.load(f)

judge = LLMJudge(mode='mock')
results = judge.judge_batch(examples)
save_judgment_results(results, 'evaluation/results/judge_results.json')

aggregated = judge.aggregate_scores(results)
print(f"Judged {len(results)} examples")
print(f"Overall mean: {aggregated['overall_mean']}")
print(f"Hallucination rate: {aggregated['hallucination_rate']}")
for dim, m in aggregated.get('dimensions', {}).items():
    print(f"  {dim}: {m['mean']}")

# 2. Run Human/Judge Agreement
print("\n=== Running Human/Judge Agreement ===")
agreement = HumanJudgeAgreement(sample_size=30)
def _judged_reply_to_dict(jr):
    """Convert JudgedReply to plain dict."""
    return {
        'example_id': jr.example_id,
        'customer_message': jr.customer_message,
        'reply': jr.reply,
        'evidence': jr.evidence,
        'scores': [{'dimension': s.dimension, 'score': s.score, 'reasoning': s.reasoning} for s in jr.scores],
        'overall_score': jr.overall_score,
        'hallucination_flag': jr.hallucination_flag,
        'judge_mode': jr.judge_mode
    }

template = agreement.sample_for_human_evaluation(
    [_judged_reply_to_dict(r) for r in results]
)

# Create mock human judgments
mock_path = 'evaluation/results/human_eval_completed.json'
create_mock_human_judgments('evaluation/results/judge_results.json', mock_path)

# Compute agreement
report = agreement.compute_agreement(mock_path, 'evaluation/results/judge_results.json')
print(f"Examples: {report.num_examples}")
print(f"Exact agreement: {report.exact_agreement_rate:.1%}")
print(f"Correlation: {report.correlation:.3f}")
for dim, m in report.dimensions.items():
    print(f"  {dim}: {m['exact_agreement']:.1%}")
agreement.save_agreement_report(report, 'evaluation/results/agreement_report.json')

# 3. Run Failure Analysis
print("\n=== Running Failure Analysis ===")
analyzer = FailureAnalyzer()
analysis = analyzer.analyze_failures(
    golden_path='evaluation/golden_set.csv',
    predictions_path='experiments/results.json'
)
if analysis:
    print(f"Intent failures: {analysis['intent_failures']} ({analysis['intent_error_rate']*100:.1f}%)")
    print(f"Escalation failures: {analysis['escalation_failures']}")
    for mode in analysis['failure_modes'][:3]:
        print(f"  {mode['name']}: {mode['count']} examples")
    analyzer.generate_failure_report(analysis, 'evaluation/results/failure_analysis.md')
    with open('evaluation/results/failure_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

print("\n=== All evaluation components complete ===")
