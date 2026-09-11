"""
Phase 13 Demo — Hiver Customer Support Agent
=============================================
Run this script to see the full pipeline flow for 3 curated examples.

Usage:
    python demo.py              # Run all 3 demos
    python demo.py --live       # Call the live API (server must be running)
    python demo.py --local      # Run directly via pipeline (no server needed)
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Demo configuration ──────────────────────────────────────────────

DEMOS = [
    {
        "id": 1,
        "label": "Normal issue -> Confidently auto-handle",
        "tag": "ride_status",
        "input": "Where is my driver? I have been waiting for 20 minutes.",
        "note": "Clean classification. High confidence. Good retrieval. No escalation.",
    },
    {
        "id": 2,
        "label": "Historical resolution -> Show grounding",
        "tag": "account_access",
        "input": "My account was deactivated and I need help getting it back.",
        "note": "System adapts a real historical brand response. Shows evidence grounding.",
    },
    {
        "id": 3,
        "label": "Sensitive issue -> Escalation attempt",
        "tag": "account_access",
        "input": "Someone accessed my account without authorization and made charges.",
        "note": (
            "Highest-scoring escalation candidate (score 0.271, threshold 0.30). "
            "Policy did NOT escalate — documented as a known gap. "
            "A stronger policy would escalate this due to high_risk_content + ambiguous_message."
        ),
    },
]

# ── Formatting helpers ──────────────────────────────────────────────

DIVIDER = "=" * 64
THIN    = "-" * 64

def print_flow(demo, result, elapsed_ms):
    """Print the structured pipeline flow for a single demo."""
    intent     = result["intent"]
    cases      = result["retrieved_cases"]
    reply      = result["reply"]
    escalate   = result["should_escalate"]
    reason     = result.get("escalation_reason")
    meta       = result.get("metadata", {})
    grounding  = meta.get("grounding_confidence", "N/A")
    esc_score  = meta.get("escalation_score", "N/A")
    factors    = meta.get("factor_scores", {})

    print(DIVIDER)
    print(f"  DEMO #{demo['id']}  —  {demo['label']}")
    print(DIVIDER)
    print()
    print(f"  Customer message:")
    print(f"    \"{demo['input']}\"")
    print()
    print(f"  Analyzing...")
    print()
    print(f"  Intent detected:      {intent['name']}  (confidence: {intent['confidence']:.4f})")
    print(f"  Historical cases:     {len(cases)} found")
    if cases:
        top = cases[0]
        print(f"    Top match:          similarity {top['similarity']:.3f}  |  intent: {top['intent']}")
        print(f"    Customer said:      \"{top['customer_message'][:80]}\"")
        if top.get("brand_response"):
            print(f"    Brand responded:    \"{top['brand_response'][:80]}\"")
    print()
    print(f"  Resolution pattern:   Grounding confidence {grounding:.4f}" if isinstance(grounding, float) else f"  Resolution pattern:   Grounding confidence {grounding}")
    print(f"    Evidence IDs:       {meta.get('evidence_ids', [])[:3]}")
    print()
    print(f"  Draft reply:")
    print(f"    \"{reply}\"")
    print()
    print(f"  Escalate?             {'YES' if escalate else 'NO'}")
    if escalate and reason:
        print(f"  Reason:               {reason}")
    if not escalate:
        top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3]
        factor_str = ", ".join(f"{k}={v:.3f}" for k, v in top_factors)
        print(f"  Escalation score:     {esc_score:.4f}  (threshold: 0.3000)")
        print(f"  Top factor scores:    {factor_str}")
    print()
    print(f"  Known issue:          {demo['note']}")
    print(f"  Time:                 {elapsed_ms:.0f}ms")
    print()


# ── Runner ──────────────────────────────────────────────────────────

def run_live():
    """Run demos against the live API server."""
    import requests

    print("\n" + DIVIDER)
    print("  HIVER CUSTOMER SUPPORT AGENT — LIVE API DEMO")
    print(DIVIDER)
    print()

    for demo in DEMOS:
        t0 = time.time()
        resp = requests.post(
            "http://localhost:8000/predict",
            json={"message": demo["input"]},
            timeout=120,
        )
        elapsed = (time.time() - t0) * 1000
        result = resp.json()
        print_flow(demo, result, elapsed)

    print(THIN)
    print("  End of live demo.")
    print(THIN)


def run_local():
    """Run demos directly via the pipeline (no server needed)."""
    from pipeline import SupportAgentPipeline
    from classifier import load_training_data
    from retrieval import load_historical_cases

    print("\n" + DIVIDER)
    print("  HIVER CUSTOMER SUPPORT AGENT — LOCAL DEMO")
    print(DIVIDER)
    print()

    print("  Initializing pipeline (this may take ~30s)...")
    t0 = time.time()
    pipeline = SupportAgentPipeline()
    train_texts, train_labels = load_training_data()
    historical_cases = load_historical_cases()
    pipeline.fit(
        training_data=(train_texts, train_labels),
        historical_cases=historical_cases,
    )
    init_time = (time.time() - t0) * 1000
    print(f"  Pipeline ready in {init_time:.0f}ms.\n")

    for demo in DEMOS:
        t0 = time.time()
        result = pipeline.predict(demo["input"])
        elapsed = (time.time() - t0) * 1000
        print_flow(demo, result, elapsed)

    print(THIN)
    print("  End of local demo.")
    print(THIN)


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hiver Customer Support Agent Demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--live", action="store_true", help="Run against live API server")
    group.add_argument("--local", action="store_true", help="Run directly via pipeline")
    args = parser.parse_args()

    if args.live:
        run_live()
    elif args.local:
        run_local()
