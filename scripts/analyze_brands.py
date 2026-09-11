"""
Analyze brand accounts in the synthetic dataset to select a brand for focus.
"""
import pandas as pd
from collections import Counter

def analyze_brands():
    # Load the synthetic dataset
    df = pd.read_csv("data/sample/twitter_support_synthetic.csv")

    # Get brand messages (inbound == 0)
    brand_messages = df[df['inbound'] == 0]

    # Count messages per brand account
    brand_counts = brand_messages['author_id'].value_counts()

    print("Brand account message counts:")
    print(brand_counts.head(10))
    print()

    # Get unique brand accounts
    unique_brands = brand_messages['author_id'].unique()
    print(f"Unique brand accounts: {len(unique_brands)}")
    print("Brand accounts:", unique_brands)
    print()

    # Analyze a sample of messages from each brand to see if we can differentiate
    print("Sample messages from each brand:")
    for brand in unique_brands[:5]:  # Show first 5 brands
        brand_msgs = brand_messages[brand_messages['author_id'] == brand]['text'].head(3)
        print(f"\n{brand}:")
        for msg in brand_msgs:
            print(f"  - {msg}")

if __name__ == "__main__":
    analyze_brands()