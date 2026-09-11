# Decision Log

## 1. Brand Selection: brand_account_5

**Decision:** Use brand_account_5 as the focus brand.

**Why:** This brand has the most messages in the synthetic dataset (72 brand messages), providing the most data for retrieval and response generation. It was the default selection from previous work.

**Alternative considered:** Selecting a brand with more diverse intents or more balanced inbound/outbound ratio.

**Why rejected:** With only 72 brand messages total, no brand has truly sufficient data. brand_account_5 is the least bad option in a synthetic dataset.

---

## 2. Synthetic Dataset Usage

**Decision:** Use the synthetic dataset and clearly label all results as synthetic.

**Why:** The real Kaggle "Customer Support on Twitter" dataset requires API credentials that are not available. The synthetic dataset mimics the structure but lacks real-world complexity.

**Alternative considered:** Downloading the real dataset via alternative means or using a different public dataset.

**Why rejected:** No alternative dataset was available without credentials. The synthetic data is the only option for completing the assignment.

---

## 3. Intent Taxonomy: 10 Intents from Keyword Analysis

**Decision:** Use 10 intent categories derived from keyword frequency analysis of customer messages.

**Why:** The taxonomy covers the main support scenarios: order status, refunds, payments, account access, technical issues, cancellations, billing, information requests, complaints, and other.

**Alternative considered:** Fewer intents (5-6) for simpler classification.

**Why rejected:** Fewer intents would collapse distinct customer needs, making the system less useful for routing and response generation.

---

## 4. Keyword Pseudo-Labeling

**Decision:** Use keyword-based pseudo-labeling for training data instead of manual annotation.

**Why:** Manual annotation of 572+ examples was not feasible within the time constraint. Keyword matching provides consistent, reproducible labels that correlate with actual customer intent.

**Alternative considered:** Using a pre-trained zero-shot classifier for auto-labeling.

**Why rejected:** Zero-shot classifiers add complexity and non-determinism without guaranteed quality improvement over simple keywords for this limited vocabulary.

---

## 5. TF-IDF + Logistic Regression as Final Classifier

**Decision:** Use TF-IDF vectorization with Logistic Regression for the final intent classifier.

**Why:** This is a well-understood, fast, interpretable approach that works well for text classification with limited data. It serves as both a strong baseline and a reasonable final system.

**Alternative considered:** Sentence embeddings with nearest-neighbor classification.

**Why rejected:** Would require additional model downloads and FAISS indexing, adding complexity without guaranteed improvement on this small synthetic dataset.

---

## 6. Top-K Retrieval with K=5

**Decision:** Retrieve top-5 most similar historical cases for response grounding.

**Why:** 5 cases provide enough context for the generator to craft a grounded response while avoiding information overload. This balances specificity with coverage.

**Alternative considered:** K=3 (less context) or K=10 (more context).

**Why rejected:** K=3 may miss relevant examples; K=10 introduces noise from less relevant cases.

---

## 7. Conservative Escalation Threshold (0.3)

**Decision:** Set escalation threshold at 0.3, meaning cases with weighted escalation score >= 0.3 are escalated.

**Why:** For customer support, false negatives (failing to escalate when needed) are more costly than false positives (unnecessary escalation). A lower threshold catches more edge cases.

**Alternative considered:** Higher threshold (0.5) to reduce escalation volume.

**Why rejected:** Would miss ambiguous or high-risk cases that need human judgment.

---

## 8. Golden Set Size: 200 Examples

**Decision:** Create a golden evaluation set of 200 examples.

**Why:** 200 provides enough examples for statistically meaningful metrics while being small enough for manual review. It falls within the 150-250 range specified in the assignment.

**Alternative considered:** 150 examples (minimum) or 250 examples (maximum).

**Why rejected:** 150 may be too small for per-intent analysis; 250 adds labeling burden without proportional benefit.

---

## 9. Stratified Golden Set Sampling

**Decision:** Stratify golden set sampling by intent to ensure minimum representation.

**Why:** With only 872 customer messages and 7 intents, random sampling might miss rare intents entirely. Stratification ensures every intent has enough examples for evaluation.

**Alternative considered:** Pure random sampling.

**Why rejected:** Risked having zero examples for some intents, making per-intent F1 undefined.

---

## 10. Mock LLM Judge for Development

**Decision:** Implement a deterministic mock judge for development, with real OpenAI judge as optional.

**Why:** No OpenAI API key is available. The mock judge provides consistent, reproducible scores for pipeline development. The real judge can be enabled when a key is provided.

**Alternative considered:** Skip LLM judge entirely and use only heuristic scoring.

**Why rejected:** The assignment explicitly requires LLM-as-judge evaluation. Having the harness ready demonstrates the capability even if mock mode is used.

---

## 11. No Autonomous Actions

**Decision:** The system drafts replies but does not execute any autonomous actions (refunds, cancellations, account changes).

**Why:** Autonomous financial or account actions require production-grade security, authorization, and audit trails that cannot be safely implemented in a take-home assignment.

**Alternative considered:** Simulating autonomous actions with mock APIs.

**Why rejected:** Would create misleading evidence of capabilities that don't exist in production.

---

## 12. Data Leakage Prevention via Train/Golden Split

**Decision:** Split data into train (572), golden (200), and test (100) before any analysis.

**Why:** Prevents information leakage where the golden set examples influence training or retrieval. The golden set is held out before the classifier is trained or the retrieval index is built.

**Alternative considered:** Using cross-validation on the full dataset.

**Why rejected:** Cross-validation would leak test information into training, violating the holdout principle required for honest evaluation.

---

## 13. Template-Based Response Generation

**Decision:** Use template-based response generation instead of LLM generation for the main system.

**Why:** No OpenAI API key is available. Templates grounded in historical evidence provide deterministic, auditable responses without API costs or non-determinism.

**Alternative considered:** Local LLM generation with a small model.

**Why rejected:** Local LLMs add complexity and require significant compute resources not guaranteed to be available.

---

## 14. Evaluation Metric Selection: Macro-F1

**Decision:** Use Macro-F1 as the primary intent classification metric.

**Why:** Macro-F1 treats all classes equally regardless of frequency, which is appropriate for imbalanced datasets where rare intents are as important as common ones.

**Alternative considered:** Weighted-F1 or accuracy.

**Why rejected:** Weighted-F1 and accuracy are dominated by the majority class (order_status at 42%), masking poor performance on rare intents.

---

## 15. Escalation Evaluation: Precision/Recall Trade-off

**Decision:** Report both escalation precision and recall, with recall prioritized.

**Why:** In customer support, missing an escalation (low recall) is worse than unnecessary escalation (low precision). The system should err on the side of human review.

**Alternative considered:** Using F1 alone as the escalation metric.

**Why rejected:** F1 balances precision and recall equally, which doesn't reflect the asymmetric cost of escalation errors.
