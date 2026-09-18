# Synthetic Ground Truth

This file documents the intentional patterns planted in the synthetic data.

## Intended Demo Story

A credit-card dispute-rate alert fires in August 2026.

The spike is partly caused by a replayed dispute-platform file, but a real business movement remains after deduplication.

The remaining real movement is concentrated in travel merchant activity, mobile channel transactions, customers with FICO <=660, and student/young-professional segments.

Complaint narrative themes that rise from July to August: travel-related charge, dispute status or resolution delay, unclear merchant descriptor, duplicate-looking charge, mobile app dispute friction, failed autopay, fraud or security concern, and foreign transaction fee.

Every probed narrative theme rises from July to August, but the increases are far from equal: rank them by the change column below before calling anything a driver of the spike.

Narratives are generated from each row's merchant_category, channel, transaction_type, merchant_name and issue, so representative evidence rows agree with their structured columns: only travel rows use travel-specific purchase wording, mobile-app dispute friction is concentrated on mobile rows, and foreign-transaction-fee wording appears only on an actual FOREIGN TRANSACTION FEE row.

## Planted Data Quality Issues

- Duplicate source transactions from replayed batch: 165
- Missing merchant categories from card-processor batch: 1,384

## Monthly Dispute Rate Check

| month | purchases | disputed | unique_source_events | raw_dispute_rate | deduped_dispute_rate |
| --- | --- | --- | --- | --- | --- |
| 2026-01 | 105267 | 1440 | 105267 | 0.0137 | 0.0137 |
| 2026-02 | 106086 | 1468 | 106086 | 0.0138 | 0.0138 |
| 2026-03 | 105487 | 1396 | 105487 | 0.0132 | 0.0132 |
| 2026-04 | 105578 | 1411 | 105578 | 0.0134 | 0.0134 |
| 2026-05 | 106167 | 1418 | 106167 | 0.0134 | 0.0134 |
| 2026-06 | 105847 | 1487 | 105847 | 0.0140 | 0.0140 |
| 2026-07 | 105611 | 1406 | 105611 | 0.0133 | 0.0133 |
| 2026-08 | 105834 | 1871 | 105669 | 0.0177 | 0.0161 |

## Top August Dispute Drivers After Deduplication

| merchant_category | channel | fico_band | disputed_purchase_count |
| --- | --- | --- | --- |
| travel | mobile | >660 | 169 |
| travel | mobile | <=660 | 111 |
| subscription | recurring | >660 | 83 |
| online_retail | card_present | >660 | 81 |
| restaurant | card_present | >660 | 74 |
| travel | card_present | >660 | 74 |
| grocery | card_present | >660 | 73 |
| travel | web | >660 | 63 |

## August Complaint Issue Mix

| issue | complaint_count |
| --- | --- |
| Problem with a purchase shown on your statement | 524 |
| Fees or interest | 175 |
| Problem when making payments | 113 |
| Problem with fraud alerts or security | 10 |

## Complaint Narrative Theme Movement (keyword probe, July vs August)

Counts are complaints whose narrative matches a simple keyword probe. A narrative can match more than one probe, so these columns are not a partition and do not sum to the monthly complaint count.

These probes are not the same measurement as the embedding theme classifier in src/text_theme_analysis.py, which assigns every complaint to exactly one theme. A probe can rise while the corresponding single-label theme falls: autopay wording rises here, but the classifier's failed_mobile_autopay theme also absorbs non-autopay payment failures, and payment complaints fall overall in August. Use the probes to confirm what the generator planted in the text, not as a substitute for the classifier output.

| narrative_theme | july_complaints | august_complaints | change |
| --- | --- | --- | --- |
| travel-related charge | 63 | 187 | 124 |
| dispute status or resolution delay | 62 | 124 | 62 |
| unclear merchant descriptor | 117 | 171 | 54 |
| duplicate-looking charge | 83 | 134 | 51 |
| mobile app dispute friction | 82 | 116 | 34 |
| failed autopay | 46 | 57 | 11 |
| fraud or security concern | 0 | 4 | 4 |
| foreign transaction fee | 40 | 42 | 2 |
