"""
Run LLM-as-Judge evaluation on the human-labelled golden set.

This script:
1. Loads the 200 human-labelled examples
2. Runs the full pipeline to get predicted intent, generated reply, and retrieved evidence
3. Feeds each example to the LLM judge (customer_message, predicted_intent, reply, evidence)
4. Does NOT pass human_intent to the judge (no ground truth leak)
5. Caches results by example_id to avoid re-running API calls
6. Saves results to experiments/llm_judge_results.json

Usage:
    set OPENAI_API_KEY=your_key_here
    python evaluation/run_llm_judge.py

    # Or with mock mode (no API key needed, for development):
    python evaluation/run_llm_judge.py --mode mock
"""
import pandas as pd
import numpy as np
import json
import sys
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_judge import LLMJudge, save_judgment_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANNOTATION_PATH = Path(__file__).parent / "human_annotation" / "annotation.csv"
CACHE_PATH = Path("experiments") / "llm_judge_cache.json"
OUTPUT_PATH = Path("experiments") / "llm_judge_results.json"


def load_pipeline():
    """Load and fit the full pipeline on training data."""
    from pipeline import SupportAgentPipeline
    from retrieval import load_historical_cases
    
    logger.info("Loading training data...")
    train_df = pd.read_csv("data/processed/train.csv")
    train_texts = train_df["customer_message"].tolist()
    
    if "intent" in train_df.columns:
        train_labels = train_df["intent"].tolist()
    else:
        from classifier import _pseudo_label_batch
        train_labels = _pseudo_label_batch(train_texts)
    
    logger.info("Loading historical cases...")
    historical_cases = load_historical_cases()
    
    logger.info("Fitting pipeline...")
    pipeline = SupportAgentPipeline()
    pipeline.fit(
        training_data=(train_texts, train_labels),
        historical_cases=historical_cases
    )
    
    return pipeline


def load_cache():
    """Load existing cache of judged examples."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Save cache of judged examples."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)


def example_id_from_message(message):
    """Generate a stable example_id from customer message."""
    return hashlib.md5(message.encode('utf-8')).hexdigest()[:12]


def run_judge_evaluation(mode='openai', batch_size=10):
    """
    Run LLM judge evaluation on all golden set examples.
    
    Args:
        mode: 'openai' for real LLM, 'mock' for deterministic scoring
        batch_size: Number of examples to process before saving cache
    """
    # Load golden set
    logger.info("Loading golden set...")
    golden_df = pd.read_csv(ANNOTATION_PATH)
    logger.info(f"Loaded {len(golden_df)} examples")
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Initialize judge
    judge = LLMJudge(mode=mode)
    logger.info(f"Judge mode: {judge.mode}")
    
    # Load cache
    cache = load_cache()
    logger.info(f"Cache has {len(cache)} previously judged examples")
    
    # Process examples
    results = []
    new_judgments = 0
    
    for idx, row in golden_df.iterrows():
        example_id = str(row['example_id'])
        customer_message = row['customer_message']
        
        # Check cache first
        if example_id in cache:
            cached = cache[example_id]
            # Reconstruct JudgedReply from cache
            from llm_judge import JudgedReply, JudgeScore
            scores = [JudgeScore(**s) for s in cached['scores']]
            result = JudgedReply(
                example_id=example_id,
                customer_message=cached['customer_message'],
                reply=cached['reply'],
                evidence=cached['evidence'],
                scores=scores,
                overall_score=cached['overall_score'],
                hallucination_flag=cached['hallucination_flag'],
                judge_mode=cached['judge_mode']
            )
            results.append(result)
            continue
        
        # Run pipeline to get predicted intent, reply, and evidence
        logger.info(f"Processing example {example_id} ({idx+1}/{len(golden_df)})")
        pipeline_result = pipeline.predict(customer_message)
        
        pred_intent = pipeline_result['intent']['name']
        reply = pipeline_result['reply']
        evidence = pipeline_result.get('evidence', '')
        
        # If evidence is a list of cases, join them
        if isinstance(evidence, list):
            evidence_texts = []
            for case in evidence[:3]:
                if isinstance(case, dict):
                    evidence_texts.append(
                        f"Customer: {case.get('customer_message', '')[:100]}\n"
                        f"Brand: {case.get('brand_response', '')[:100]}\n"
                        f"Resolution: {case.get('resolution', '')}"
                    )
                else:
                    evidence_texts.append(str(case))
            evidence = "\n\n".join(evidence_texts)
        
        # Judge the reply (does NOT receive human_intent)
        result = judge.judge_reply(
            customer_message=customer_message,
            reply=reply,
            evidence=evidence,
            example_id=example_id
        )
        
        results.append(result)
        new_judgments += 1
        
        # Cache the result
        cache[example_id] = {
            'example_id': example_id,
            'customer_message': customer_message,
            'predicted_intent': pred_intent,
            'reply': reply,
            'evidence': evidence,
            'scores': [{'dimension': s.dimension, 'score': s.score, 'reasoning': s.reasoning} 
                       for s in result.scores],
            'overall_score': result.overall_score,
            'hallucination_flag': result.hallucination_flag,
            'judge_mode': result.judge_mode
        }
        
        # Save cache periodically
        if new_judgments % batch_size == 0:
            save_cache(cache)
            logger.info(f"Saved cache after {new_judgments} new judgments")
    
    # Final cache save
    save_cache(cache)
    logger.info(f"Total new judgments: {new_judgments}")
    
    # Aggregate scores
    aggregated = judge.aggregate_scores(results)
    
    # Build output
    output = {
        'timestamp': datetime.now().isoformat(),
        'num_examples': len(results),
        'judge_mode': judge.mode,
        'aggregated_scores': aggregated,
        'per_example': []
    }
    
    for result in results:
        output['per_example'].append({
            'example_id': result.example_id,
            'customer_message': result.customer_message[:200],
            'reply': result.reply[:200],
            'overall_score': result.overall_score,
            'hallucination_flag': result.hallucination_flag,
            'scores': {s.dimension: s.score for s in result.scores}
        })
    
    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {OUTPUT_PATH}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("LLM JUDGE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Examples evaluated: {len(results)}")
    print(f"Judge mode: {judge.mode}")
    print(f"Overall mean score: {aggregated.get('overall_mean', 'N/A')}")
    print(f"Hallucination rate: {aggregated.get('hallucination_rate', 'N/A')}")
    print("\nPer-dimension scores:")
    for dim, metrics in aggregated.get('dimensions', {}).items():
        print(f"  {dim}: mean={metrics['mean']}, min={metrics['min']}, max={metrics['max']}")
    
    if judge.mode == 'mock':
        print("\nNOTE: These are MOCK scores (no API key). Set OPENAI_API_KEY for real evaluation.")
        print("Command: set OPENAI_API_KEY=your_key && python evaluation/run_llm_judge.py")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge evaluation")
    parser.add_argument("--mode", choices=["openai", "mock"], default="openai",
                       help="Judge mode: 'openai' for real LLM, 'mock' for deterministic")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Cache save interval")
    args = parser.parse_args()
    
    run_judge_evaluation(mode=args.mode, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
