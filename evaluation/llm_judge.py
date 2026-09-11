"""
LLM-as-Judge harness for evaluating reply quality.

Evaluates replies on:
1. Relevance - Does the reply address the customer's issue?
2. Groundedness - Is the reply supported by historical evidence?
3. Correctness - Is the reply factually consistent with the evidence?
4. Completeness - Does the reply adequately address the issue?
5. Tone - Is the reply professional and empathetic?
6. Hallucination risk - Does the reply invent unsupported claims?

Rubric (1-5 scale):
  5 = excellent
  4 = good
  3 = acceptable
  2 = poor
  1 = unsafe/wrong

Modes:
  - mock: Deterministic scores for development (no API required)
  - openai: Uses OpenAI API when OPENAI_API_KEY is set
"""
import json
import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    """A single dimension score from the LLM judge."""
    dimension: str
    score: int  # 1-5
    reasoning: str


@dataclass
class JudgedReply:
    """Complete judgment for a single reply."""
    example_id: str
    customer_message: str
    reply: str
    evidence: str
    scores: List[JudgeScore]
    overall_score: float
    hallucination_flag: bool
    judge_mode: str


class LLMJudge:
    """
    Evaluates reply quality using LLM-as-judge or mock scoring.
    """

    DIMENSIONS = ['relevance', 'groundedness', 'correctness', 'completeness', 'tone', 'hallucination_risk']

    RUBRIC = {
        'relevance': 'Does the reply directly address the customer issue? Is it on-topic?',
        'groundedness': 'Is the reply supported by the provided historical evidence? Does it avoid unsupported claims?',
        'correctness': 'Is the reply factually consistent with the evidence? Are there contradictions?',
        'completeness': 'Does the reply adequately address the issue, or is it too vague/generic?',
        'tone': 'Is the reply professional, empathetic, and appropriate for customer support?',
        'hallucination_risk': 'Does the reply invent policies, timelines, refunds, or guarantees not in the evidence? (lower = more hallucination)'
    }

    def __init__(self, mode: str = 'mock'):
        """
        Initialize the judge.
        
        Args:
            mode: 'mock' for deterministic scoring, 'openai' for real LLM judging
        """
        self.mode = mode
        self.openai_available = False
        self._openai_client = None
        
        if mode == 'openai':
            try:
                import openai
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self._openai_client = openai.OpenAI(api_key=api_key)
                    self.openai_available = True
                    logger.info("LLM Judge: OpenAI mode enabled")
                else:
                    logger.warning("LLM Judge: No OPENAI_API_KEY found, falling back to mock mode")
                    self.mode = 'mock'
            except ImportError:
                logger.warning("LLM Judge: openai package not installed, falling back to mock mode")
                self.mode = 'mock'
        
        if self.mode == 'mock':
            logger.info("LLM Judge: Using mock scoring mode")

    def judge_reply(self, customer_message: str, reply: str, evidence: str = '',
                    example_id: str = 'unknown') -> JudgedReply:
        """
        Judge a single reply on all dimensions.
        
        Args:
            customer_message: The original customer message
            reply: The generated reply to evaluate
            evidence: The historical evidence used for grounding
            example_id: Identifier for this example
            
        Returns:
            JudgedReply with scores for each dimension
        """
        if self.mode == 'openai' and self.openai_available:
            return self._judge_with_openai(customer_message, reply, evidence, example_id)
        else:
            return self._judge_with_mock(customer_message, reply, evidence, example_id)

    def _judge_with_mock(self, customer_message: str, reply: str, evidence: str,
                         example_id: str) -> JudgedReply:
        """
        Deterministic mock scoring based on text analysis.
        Used for development and when no API key is available.
        """
        scores = []
        hallucination_flag = False

        # Relevance: check if reply addresses keywords from customer message
        customer_words = set(customer_message.lower().split())
        reply_words = set(reply.lower().split())
        overlap = len(customer_words & reply_words)
        relevance_score = min(5, max(1, 2 + overlap // 2))
        scores.append(JudgeScore('relevance', relevance_score,
            f'Keyword overlap: {overlap} words'))

        # Groundedness: check if reply references evidence
        evidence_words = set(evidence.lower().split()) if evidence else set()
        evidence_overlap = len(reply_words & evidence_words)
        if evidence:
            groundedness_score = min(5, max(1, 2 + evidence_overlap // 3))
        else:
            groundedness_score = 2
        scores.append(JudgeScore('groundedness', groundedness_score,
            f'Evidence word overlap: {evidence_overlap}'))

        # Correctness: check for contradictions (simplified)
        correctness_score = 4 if not self._has_obvious_contradictions(reply) else 2
        scores.append(JudgeScore('correctness', correctness_score,
            'No obvious contradictions detected' if correctness_score == 4 else 'Potential contradictions found'))

        # Completeness: check reply length and specificity
        reply_len = len(reply.split())
        if reply_len < 5:
            completeness_score = 2
        elif reply_len < 10:
            completeness_score = 3
        else:
            completeness_score = 4
        scores.append(JudgeScore('completeness', completeness_score,
            f'Reply length: {reply_len} words'))

        # Tone: check for professional language
        negative_indicators = ['unfortunately', 'cannot', 'unable', 'denied', 'rejected']
        positive_indicators = ['help', 'assist', 'resolve', 'process', 'happy', 'sorry']
        tone_score = 4
        if any(neg in reply.lower() for neg in negative_indicators):
            tone_score -= 1
        if any(pos in reply.lower() for pos in positive_indicators):
            tone_score += 1
        tone_score = min(5, max(1, tone_score))
        scores.append(JudgeScore('tone', tone_score,
            'Professional tone detected'))

        # Hallucination risk: check for unsupported claims
        hallucination_indicators = [
            'refund processed', 'order shipped', 'delivery date',
            'account unlocked', 'password reset', 'cancellation confirmed'
        ]
        hallucination_count = sum(1 for ind in hallucination_indicators if ind in reply.lower())
        if hallucination_count > 0 and not evidence:
            hallucination_score = 2
            hallucination_flag = True
        elif hallucination_count > 2:
            hallucination_score = 3
        else:
            hallucination_score = 4
        scores.append(JudgeScore('hallucination_risk', hallucination_score,
            f'Potential hallucination indicators: {hallucination_count}'))

        # Calculate overall score (average, weighted against hallucination)
        score_values = [s.score for s in scores]
        overall = sum(score_values) / len(score_values)
        # Penalize high hallucination risk
        if hallucination_flag:
            overall = max(1.0, overall - 1.0)

        return JudgedReply(
            example_id=example_id,
            customer_message=customer_message,
            reply=reply,
            evidence=evidence,
            scores=scores,
            overall_score=round(overall, 2),
            hallucination_flag=hallucination_flag,
            judge_mode='mock'
        )

    def _judge_with_openai(self, customer_message: str, reply: str, evidence: str,
                           example_id: str) -> JudgedReply:
        """Judge using OpenAI API with structured prompting."""
        try:
            prompt = f"""You are an expert customer support quality evaluator.
Evaluate the following reply on each dimension using the 1-5 rubric.

Customer message: "{customer_message}"

Historical evidence: "{evidence}"

Generated reply: "{reply}"

Evaluate on these dimensions:
1. Relevance (1-5): Does the reply directly address the customer issue?
2. Groundedness (1-5): Is the reply supported by the evidence?
3. Correctness (1-5): Is the reply factually consistent?
4. Completeness (1-5): Does the reply adequately address the issue?
5. Tone (1-5): Is the reply professional and empathetic?
6. Hallucination risk (1-5, higher=better): Does the reply avoid unsupported claims?

Respond in JSON format:
{{
  "relevance": {{"score": <1-5>, "reasoning": "<brief>"}},
  "groundedness": {{"score": <1-5>, "reasoning": "<brief>"}},
  "correctness": {{"score": <1-5>, "reasoning": "<brief>"}},
  "completeness": {{"score": <1-5>, "reasoning": "<brief>"}},
  "tone": {{"score": <1-5>, "reasoning": "<brief>"}},
  "hallucination_risk": {{"score": <1-5>, "reasoning": "<brief>"}}
}}"""

            response = self._openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a strict customer support quality evaluator. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )

            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)

            scores = []
            for dim in self.DIMENSIONS:
                dim_result = result.get(dim, {'score': 3, 'reasoning': 'No reasoning provided'})
                scores.append(JudgeScore(dim, dim_result['score'], dim_result['reasoning']))

            hallucination_flag = result.get('hallucination_risk', {}).get('score', 3) <= 2
            overall = sum(s.score for s in scores) / len(scores)

            return JudgedReply(
                example_id=example_id,
                customer_message=customer_message,
                reply=reply,
                evidence=evidence,
                scores=scores,
                overall_score=round(overall, 2),
                hallucination_flag=hallucination_flag,
                judge_mode='openai'
            )

        except Exception as e:
            logger.error(f"OpenAI judge failed: {e}, falling back to mock")
            return self._judge_with_mock(customer_message, reply, evidence, example_id)

    def _has_obvious_contradictions(self, reply: str) -> bool:
        """Check for obvious contradictions in the reply."""
        contradiction_pairs = [
            ('processed', 'not processed'),
            ('shipped', 'not shipped'),
            ('approved', 'denied'),
            ('refund', 'no refund'),
        ]
        reply_lower = reply.lower()
        for pos, neg in contradiction_pairs:
            if pos in reply_lower and neg in reply_lower:
                return True
        return False

    def judge_batch(self, examples: List[Dict[str, Any]]) -> List[JudgedReply]:
        """
        Judge a batch of examples.
        
        Args:
            examples: List of dicts with 'customer_message', 'reply', 'evidence', 'example_id'
            
        Returns:
            List of JudgedReply objects
        """
        results = []
        for ex in examples:
            result = self.judge_reply(
                customer_message=ex.get('customer_message', ''),
                reply=ex.get('reply', ''),
                evidence=ex.get('evidence', ''),
                example_id=ex.get('example_id', 'unknown')
            )
            results.append(result)
        return results

    def aggregate_scores(self, judged_replies: List[JudgedReply]) -> Dict[str, Any]:
        """
        Aggregate scores across all judged replies.
        
        Returns:
            Dictionary with per-dimension averages and overall statistics
        """
        if not judged_replies:
            return {}

        dimension_scores = {dim: [] for dim in self.DIMENSIONS}
        overall_scores = []
        hallucination_count = 0

        for jr in judged_replies:
            overall_scores.append(jr.overall_score)
            if jr.hallucination_flag:
                hallucination_count += 1
            for score in jr.scores:
                if score.dimension in dimension_scores:
                    dimension_scores[score.dimension].append(score.score)

        aggregated = {
            'num_examples': len(judged_replies),
            'overall_mean': round(sum(overall_scores) / len(overall_scores), 2),
            'overall_min': round(min(overall_scores), 2),
            'overall_max': round(max(overall_scores), 2),
            'hallucination_rate': round(hallucination_count / len(judged_replies), 2),
            'dimensions': {}
        }

        for dim, scores in dimension_scores.items():
            if scores:
                aggregated['dimensions'][dim] = {
                    'mean': round(sum(scores) / len(scores), 2),
                    'min': min(scores),
                    'max': max(scores)
                }

        return aggregated


def save_judgment_results(results: List[JudgedReply], output_path: str):
    """Save judgment results to JSON."""
    data = [asdict(r) for r in results]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(results)} judgment results to {output_path}")


def main():
    """Main function to demonstrate LLM judge usage."""
    print("Initializing LLM Judge...")
    judge = LLMJudge(mode='mock')

    # Example judgments
    examples = [
        {
            'example_id': '1',
            'customer_message': 'Where is my order #12345?',
            'reply': "I've checked your order #12345 and it's currently shipped. You should receive it within 3-5 business days.",
            'evidence': 'Customer asked about order status. Previous similar cases resolved by providing tracking info.'
        },
        {
            'example_id': '2',
            'customer_message': 'I want a refund',
            'reply': 'Is there anything else I can help you with today?',
            'evidence': 'Customer requested refund. Historical cases show refunds processed in 3-5 business days.'
        },
        {
            'example_id': '3',
            'customer_message': 'My account is locked',
            'reply': "I've helped you reset your password. You should receive an email shortly with instructions.",
            'evidence': 'Account lock issue. Previous cases resolved via password reset.'
        }
    ]

    print("\n=== Judging replies ===")
    results = judge.judge_batch(examples)

    for result in results:
        print(f"\nExample {result.example_id}:")
        print(f"  Customer: {result.customer_message[:50]}...")
        print(f"  Reply: {result.reply[:50]}...")
        print(f"  Overall: {result.overall_score}/5.0")
        print(f"  Hallucination: {'YES' if result.hallucination_flag else 'No'}")
        for score in result.scores:
            print(f"    {score.dimension}: {score.score}/5 - {score.reasoning}")

    # Aggregate
    aggregated = judge.aggregate_scores(results)
    print(f"\n=== Aggregated Scores ===")
    print(json.dumps(aggregated, indent=2))

    # Save results
    save_judgment_results(results, 'evaluation/results/judge_results.json')


if __name__ == "__main__":
    main()
