"""
Prepare golden set annotation template from real data.

This script:
1. Loads the golden split from real data
2. Samples examples for manual annotation
3. Creates an annotation template CSV
4. Creates ANNOTATION_GUIDE.md with clear labeling instructions

The script may SAMPLE the examples.
The script must NOT claim to have HAND-LABELLED them.

Labels must be manually reviewed by a human.

Usage:
    python evaluation/prepare_golden_annotation.py
    python evaluation/prepare_golden_annotation.py --golden-path data/processed/real_golden.csv --output-size 200
"""
import pandas as pd
import numpy as np
import json
import yaml
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_golden_split(path: str = "data/processed/real_golden.csv") -> pd.DataFrame:
    """Load the golden split from processed data."""
    golden_path = Path(path)
    if not golden_path.exists():
        raise FileNotFoundError(
            f"Golden split not found at {golden_path}. "
            f"Run: python scripts/load_real_data.py first."
        )

    df = pd.read_csv(golden_path)
    logger.info(f"Loaded {len(df)} conversations from golden split")
    return df


def load_taxonomy(path: str = "configs/intents_real.yaml") -> List[Dict]:
    """Load the intent taxonomy."""
    tax_path = Path(path)
    if not tax_path.exists():
        # Fallback to synthetic taxonomy
        tax_path = Path("configs/intents.yaml")
        if not tax_path.exists():
            logger.warning("No taxonomy found, using default intents")
            return []

    with open(tax_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('intents', [])


def sample_for_annotation(
    golden_df: pd.DataFrame,
    target_size: int = 200,
    taxonomy: List[Dict] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Sample examples for human annotation.

    Sampling strategy:
    - Stratified by pseudo-intent (if available)
    - Include short/noisy messages
    - Include ambiguous cases
    - Include potential escalation candidates
    """
    random.seed(seed)
    np.random.seed(seed)

    n_available = len(golden_df)
    target_size = min(target_size, n_available)

    logger.info(f"Sampling {target_size} examples from {n_available} available")

    if target_size >= n_available:
        logger.info("Using all available examples")
        sampled = golden_df.copy()
    else:
        # Try to stratify if we have pseudo-labels
        if 'pseudo_intent' in golden_df.columns:
            intent_groups = golden_df.groupby('pseudo_intent')
            sampled_indices = []

            for intent_name, group in intent_groups:
                # Proportional sampling
                proportion = len(group) / n_available
                n_samples = max(1, int(target_size * proportion))
                n_samples = min(n_samples, len(group))
                sampled = group.sample(n=n_samples, random_state=seed)
                sampled_indices.extend(sampled.index.tolist())

            # Trim to target size
            if len(sampled_indices) > target_size:
                sampled_indices = sampled_indices[:target_size]

            sampled = golden_df.loc[sampled_indices].copy()
        else:
            sampled = golden_df.sample(n=target_size, random_state=seed).copy()

    # Reset index
    sampled = sampled.reset_index(drop=True)

    # Add annotation fields
    sampled['example_id'] = range(1, len(sampled) + 1)
    sampled['intent'] = ''  # Empty - annotator fills this
    sampled['should_escalate'] = ''  # Empty - annotator fills this
    sampled['escalation_reason'] = ''  # Empty - annotator fills this
    sampled['annotator_notes'] = ''
    sampled['human_labelled'] = False
    sampled['annotation_timestamp'] = ''

    logger.info(f"Sampled {len(sampled)} examples for annotation")
    return sampled


def create_annotation_csv(sampled_df: pd.DataFrame, output_path: str):
    """Save the annotation template as CSV."""
    # Select relevant columns for annotation
    annotation_cols = [
        'example_id', 'conversation_id', 'customer_message',
        'intent', 'should_escalate', 'escalation_reason',
        'annotator_notes', 'human_labelled', 'annotation_timestamp'
    ]

    # Add pseudo_intent if available (as hint for annotator)
    if 'pseudo_intent' in sampled_df.columns:
        annotation_cols.insert(3, 'pseudo_intent')

    # Only include columns that exist
    annotation_cols = [col for col in annotation_cols if col in sampled_df.columns]

    output_df = sampled_df[annotation_cols].copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Saved annotation template to {output_path} ({len(output_df)} examples)")


def create_annotation_json(sampled_df: pd.DataFrame, output_path: str):
    """Save the annotation template as JSON (more flexible format)."""
    examples = []
    for _, row in sampled_df.iterrows():
        example = {
            'example_id': int(row.get('example_id', 0)),
            'conversation_id': str(row.get('conversation_id', '')),
            'customer_message': str(row.get('customer_message', '')),
            'pseudo_intent': str(row.get('pseudo_intent', '')),  # Hint for annotator
            'intent': '',  # Empty - annotator fills this
            'should_escalate': '',  # Empty - annotator fills this
            'escalation_reason': '',
            'annotator_notes': '',
            'human_labelled': False,
            'annotation_timestamp': '',
        }
        examples.append(example)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    logger.info(f"Saved annotation template to {output_path} ({len(examples)} examples)")


def main():
    """Main function to prepare golden set annotation."""
    parser = argparse.ArgumentParser(description="Prepare golden set annotation template")
    parser.add_argument("--golden-path", type=str, default="data/processed/real_golden.csv",
                        help="Path to golden split CSV")
    parser.add_argument("--output-size", type=int, default=200,
                        help="Target number of examples for annotation")
    parser.add_argument("--output-dir", type=str, default="evaluation",
                        help="Output directory for annotation files")
    args = parser.parse_args()

    print("=" * 60)
    print("GOLDEN SET ANNOTATION PREPARATION")
    print("=" * 60)

    # Load golden split
    try:
        golden_df = load_golden_split(args.golden_path)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run: python scripts/load_real_data.py first")
        return

    # Load taxonomy
    taxonomy = load_taxonomy()
    print(f"Loaded taxonomy with {len(taxonomy)} intents")

    # Sample for annotation
    sampled = sample_for_annotation(golden_df, target_size=args.output_size, taxonomy=taxonomy)

    # Create output files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV format
    csv_path = output_dir / "golden_set_annotation.csv"
    create_annotation_csv(sampled, str(csv_path))

    # JSON format
    json_path = output_dir / "golden_set_annotation.json"
    create_annotation_json(sampled, str(json_path))

    # Print summary
    print(f"\n=== Annotation Template Created ===")
    print(f"Examples: {len(sampled)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")

    # Print intent distribution (if available)
    if 'pseudo_intent' in sampled.columns:
        intent_dist = sampled['pseudo_intent'].value_counts()
        print(f"\nIntent distribution (pseudo-labels for reference):")
        for intent, count in intent_dist.items():
            print(f"  {intent}: {count} ({count/len(sampled)*100:.1f}%)")

    # Print status
    print(f"\n=== STATUS: HAND LABELING PENDING ===")
    print(f"The script has sampled {len(sampled)} examples.")
    print(f"Labels are NOT hand-labelled yet.")
    print(f"Please annotate the golden set before reporting final metrics.")
    print(f"\nSee evaluation/ANNOTATION_GUIDE.md for labeling instructions.")


if __name__ == "__main__":
    main()
