# Data Directory

This directory contains datasets used for the customer support agent.

## Structure

- `sample/` — Synthetic test fixture (development only)
- `brand_specific/` — Brand-filtered data
- `processed/` — Train/validation/golden splits
- `raw/` — Real Kaggle dataset (blocked until authentication)
- `embeddings_cache/` — Cached sentence transformer embeddings

## Real Dataset

The real Kaggle "Customer Support on Twitter" dataset is not yet available because Kaggle API credentials are not configured.

**To obtain the real dataset:**
1. Get API credentials from https://www.kaggle.com/settings
2. Set `KAGGLE_API_TOKEN` environment variable, OR place `kaggle.json` in `~/.kaggle/`
3. Run: `kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw/`
4. Run: `unzip data/raw/*.zip -d data/raw/`
5. Run: `python scripts/load_real_data.py`

See `data/raw/README.md` for detailed instructions.

## Dataset Source

The dataset used is "Customer Support on Twitter" from Kaggle:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

This dataset contains approximately 3 million tweets from customer support conversations on Twitter.

## Preparation

### Synthetic Data (current)
Run `python scripts/prepare_data.py` to:
1. Load the synthetic dataset
2. Create train/test/golden splits
3. Apply keyword pseudo-labels

### Real Data (when available)
Run `python scripts/load_real_data.py` to:
1. Load the real Kaggle dataset
2. Select the best brand based on data-driven criteria
3. Build conversation-level cases from tweet threads
4. Create conversation-aware splits
5. Derive the intent taxonomy from real data
6. Save everything for training and evaluation

## Important Notes

- The golden evaluation set is created manually and stored in `evaluation/golden_set.csv`
- Golden set examples must NOT appear in training or retrieval index
- Conversation-aware splitting ensures no leakage
- Labels must be manually verified by a human annotator
