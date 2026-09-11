# Data Exploration Report

## Dataset Overview

- **Total rows**: 10,000
- **Columns**: tweet_id, author_id, inbound, text, created_at, text_length
- **Date range**: 2023-01-01 00:37:00 to 2023-12-31 23:55:00

## Message Composition

- **Customer messages**: 7,125 (71.2%)
- **Brand messages**: 2,875 (28.7%)

## Conversation Structure

- **Number of conversations**: 9,842
- **Average conversation length**: 1.0 messages
- **Median conversation length**: 1.0 messages
- **Longest conversation**: 3 messages
- **Shortest conversation**: 1 messages
- **Estimated resolved conversations**: 2,825 (28.7%)

## Text Analysis

- **Average message length**: 34.6 characters
- **Median message length**: 32.0 characters
- **Standard deviation**: 11.5 characters
- **Empty messages**: 0
- **Very short messages (<5 chars)**: 0

## Common Words in Customer Messages

| Rank | Word | Frequency |
|------|------|-----------|
| 1 | order | 2,256 |
| 2 | item | 1,674 |
| 3 | purchase | 1,175 |
| 4 | not | 1,105 |
| 5 | working | 1,105 |
| 6 | need | 728 |
| 7 | received | 714 |
| 8 | when | 703 |
| 9 | from | 703 |
| 10 | refund | 701 |
| 11 | charged | 681 |
| 12 | subscription | 434 |
| 13 | how | 395 |
| 14 | return | 395 |
| 15 | missing | 384 |
| 16 | get | 382 |
| 17 | tracking | 380 |
| 18 | number | 380 |
| 19 | never | 373 |
| 20 | confirmation | 373 |

## Data Quality Issues

- **Duplicate tweets**: To be analyzed
- **Missing/noisy data**: See missing values above
- **Empty messages**: Counted above
- **Potential spam/bots**: Requires further analysis

## Next Steps

1. Select a specific brand for focused analysis
2. Derive intent taxonomy from historical conversations
3. Create golden evaluation set through manual annotation
4. Implement and evaluate baselines
