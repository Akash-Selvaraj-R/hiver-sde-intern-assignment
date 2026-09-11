"""
Fast grounding audit: trace 10 representative examples using TF-IDF baseline
for intent + a small retriever subset for evidence.

This avoids building the full 35K FAISS index.
"""
import json
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ANNOTATION_PATH = Path(__file__).parent / "human_annotation" / "annotation.csv"


def run_fast_grounding_audit():
    """Trace 10 examples using the TF-IDF baseline + small retriever."""
    golden_df = pd.read_csv(ANNOTATION_PATH)
    
    # Load training data
    train_df = pd.read_csv("data/processed/train.csv")
    train_texts = train_df["customer_message"].tolist()
    if "intent" in train_df.columns:
        train_labels = train_df["intent"].tolist()
    else:
        from classifier import _pseudo_label_batch
        train_labels = _pseudo_label_batch(train_texts)
    
    # Use TF-IDF baseline for intent
    from baselines import TfIdfBaseline
    baseline = TfIdfBaseline()
    baseline.fit(train_texts, train_labels, train_texts)
    
    # Build a small retriever from a subset
    from retrieval import HistoricalRetriever, load_historical_cases
    
    # Use only 500 cases for fast retrieval
    all_cases = load_historical_cases()
    subset_cases = all_cases[:500]
    
    retriever = HistoricalRetriever()
    retriever.fit(subset_cases)
    
    # Use the generator
    from generator import ResponseGenerator
    generator = ResponseGenerator()
    
    # Select 10 representative examples
    representative_ids = [1, 5, 8, 12, 18, 22, 35, 48, 60, 75]
    
    audit_results = []
    
    for example_id in representative_ids:
        row = golden_df[golden_df['example_id'] == example_id].iloc[0]
        message = row['customer_message']
        human_intent = row['human_intent']
        
        # Predict intent
        pred_intent = baseline.predict_intent([message])[0]
        
        # Retrieve similar cases
        retrieved = retriever.retrieve(message, k=5)
        
        # Extract resolution actions
        resolution_actions = []
        for case in retrieved:
            brand_resp = case.get('brand_response', '')
            if brand_resp and len(brand_resp) > 10:
                # Check if it's a real response (not empty)
                resolution_actions.append(brand_resp[:100])
        
        # Generate reply
        gen_result = generator.generate(
            query=message,
            intent=pred_intent,
            confidence=0.8,
            retrieved_cases=retrieved
        )
        
        reply = gen_result.reply
        grounding_confidence = gen_result.grounding_confidence
        
        # Check if reply is template-based
        is_template = '{resolution_details}' in reply or reply.startswith('Thanks for')
        
        # Check if reply uses brand_response
        uses_brand_response = False
        brand_response_text = ''
        for case in retrieved[:3]:
            br = case.get('brand_response', '')
            if br and len(br) > 10:
                br_words = set(br.lower().split())
                reply_words = set(reply.lower().split())
                overlap = len(br_words & reply_words)
                if overlap > 3:
                    uses_brand_response = True
                    brand_response_text = br[:200]
                    break
        
        audit_entry = {
            'example_id': example_id,
            'customer_message': message[:200],
            'human_intent': human_intent,
            'predicted_intent': pred_intent,
            'reply': reply[:200],
            'is_template_based': is_template,
            'uses_brand_response': uses_brand_response,
            'brand_response_text': brand_response_text,
            'evidence_count': len(retrieved),
            'grounding_confidence': grounding_confidence,
            'retrieved_brand_responses': [c.get('brand_response', '')[:80] for c in retrieved[:3] if c.get('brand_response')]
        }
        
        audit_results.append(audit_entry)
    
    # Analysis
    template_count = sum(1 for r in audit_results if r['is_template_based'])
    brand_response_count = sum(1 for r in audit_results if r['uses_brand_response'])
    
    print("\n" + "=" * 60)
    print("GROUNDING AUDIT (Fast Mode)")
    print("=" * 60)
    print(f"Examples traced: {len(audit_results)}")
    print(f"Template-based replies: {template_count}/{len(audit_results)}")
    print(f"Uses brand_response from evidence: {brand_response_count}/{len(audit_results)}")
    print(f"Average grounding confidence: {sum(r['grounding_confidence'] for r in audit_results)/len(audit_results):.3f}")
    
    print("\nDetailed Trace:")
    for entry in audit_results:
        print(f"\n--- Example {entry['example_id']} ---")
        print(f"  Message: {entry['customer_message'][:80]}...")
        print(f"  Human intent: {entry['human_intent']}")
        print(f"  Predicted intent: {entry['predicted_intent']}")
        print(f"  Reply: {entry['reply'][:100]}...")
        print(f"  Template-based: {entry['is_template_based']}")
        print(f"  Uses brand_response: {entry['uses_brand_response']}")
        if entry['brand_response_text']:
            print(f"    Brand response used: {entry['brand_response_text'][:80]}...")
        print(f"  Evidence cases: {entry['evidence_count']}")
        print(f"  Grounding confidence: {entry['grounding_confidence']:.3f}")
        if entry['retrieved_brand_responses']:
            print(f"  Retrieved brand responses:")
            for i, br in enumerate(entry['retrieved_brand_responses'][:2]):
                print(f"    {i+1}. {br[:60]}...")
    
    # Honest assessment
    print("\n" + "=" * 60)
    print("HONEST ASSESSMENT")
    print("=" * 60)
    
    if brand_response_count >= len(audit_results) * 0.5:
        print("FINDING: Generation PARTIALLY uses historical brand responses.")
        print("The generator extracts resolution patterns from real historical brand responses.")
    elif template_count > len(audit_results) * 0.7:
        print("FINDING: Generation is STILL PRIMARILY template-based.")
        print("While brand responses are now loaded, the generator often falls back to templates.")
        print("This is because many retrieved cases have short/generic brand responses.")
    else:
        print("FINDING: Generation has MIXED grounding.")
    
    # Save audit
    output_path = Path("experiments/grounding_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'mode': 'fast',
            'examples_traced': len(audit_results),
            'template_count': template_count,
            'brand_response_count': brand_response_count,
            'avg_grounding_confidence': sum(r['grounding_confidence'] for r in audit_results)/len(audit_results),
            'results': audit_results
        }, f, indent=2)
    
    print(f"\nAudit saved to {output_path}")


if __name__ == "__main__":
    run_fast_grounding_audit()
