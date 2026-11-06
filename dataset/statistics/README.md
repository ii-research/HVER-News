# Evidence statistics by query type

Counts are query–sentence pairs, not unique sentences. Non-supporting counts cover only judged query–article pairs; unjudged pairs are excluded.

Computed from the released dataset files. Positive labels are checked against sentence qrels; source-article labels are checked against the document-level view.

The `document-level` rows count source-article evidence, `cross-document-only` rows count other-article evidence, and `corpus-level` rows combine both. Cross-document-only is an evidence subset, not a third ranking task.

## Train

| Scope | Query | Queries with judgments | Direct | Partial | Non-supporting |
|---|---|---:|---:|---:|---:|
| document-level | Q1 | 16,355 | 31,045 | 54,034 | 704,081 |
| document-level | Q2 | 16,355 | 31,045 | 54,034 | 704,081 |
| document-level | Q3 | 16,355 | 31,045 | 54,034 | 704,081 |
| document-level | Q4 | 16,355 | 31,045 | 54,034 | 704,081 |
| cross-document-only | Q1 | 6,766 | 34,640 | 36,646 | 859,239 |
| cross-document-only | Q2 | 6,762 | 10,258 | 11,322 | 908,432 |
| cross-document-only | Q3 | 6,769 | 11,325 | 12,389 | 906,830 |
| cross-document-only | Q4 | 6,759 | 11,754 | 12,664 | 904,876 |
| corpus-level | Q1 | 16,355 | 65,685 | 90,680 | 1,563,320 |
| corpus-level | Q2 | 16,355 | 41,303 | 65,356 | 1,612,513 |
| corpus-level | Q3 | 16,355 | 42,370 | 66,423 | 1,610,911 |
| corpus-level | Q4 | 16,355 | 42,799 | 66,698 | 1,608,957 |

## Dev

| Scope | Query | Queries with judgments | Direct | Partial | Non-supporting |
|---|---|---:|---:|---:|---:|
| document-level | Q1 | 2,506 | 4,696 | 8,097 | 99,274 |
| document-level | Q2 | 2,506 | 4,696 | 8,097 | 99,274 |
| document-level | Q3 | 2,506 | 4,696 | 8,097 | 99,274 |
| document-level | Q4 | 2,506 | 4,696 | 8,097 | 99,274 |
| cross-document-only | Q1 | 544 | 1,547 | 1,626 | 35,402 |
| cross-document-only | Q2 | 546 | 343 | 404 | 37,980 |
| cross-document-only | Q3 | 545 | 381 | 396 | 37,886 |
| cross-document-only | Q4 | 546 | 446 | 532 | 37,673 |
| corpus-level | Q1 | 2,506 | 6,243 | 9,723 | 134,676 |
| corpus-level | Q2 | 2,506 | 5,039 | 8,501 | 137,254 |
| corpus-level | Q3 | 2,506 | 5,077 | 8,493 | 137,160 |
| corpus-level | Q4 | 2,506 | 5,142 | 8,629 | 136,947 |

## Test

| Scope | Query | Queries with judgments | Direct | Partial | Non-supporting |
|---|---|---:|---:|---:|---:|
| document-level | Q1 | 5,539 | 11,789 | 19,106 | 210,504 |
| document-level | Q2 | 5,539 | 11,789 | 19,106 | 210,504 |
| document-level | Q3 | 5,539 | 11,789 | 19,106 | 210,504 |
| document-level | Q4 | 5,539 | 11,789 | 19,106 | 210,504 |
| cross-document-only | Q1 | 2,032 | 11,506 | 13,212 | 198,313 |
| cross-document-only | Q2 | 2,032 | 3,544 | 4,200 | 215,554 |
| cross-document-only | Q3 | 2,035 | 3,881 | 4,533 | 214,980 |
| cross-document-only | Q4 | 2,034 | 3,852 | 4,577 | 214,894 |
| corpus-level | Q1 | 5,539 | 23,295 | 32,318 | 408,817 |
| corpus-level | Q2 | 5,539 | 15,333 | 23,306 | 426,058 |
| corpus-level | Q3 | 5,539 | 15,670 | 23,639 | 425,484 |
| corpus-level | Q4 | 5,539 | 15,641 | 23,683 | 425,398 |

## Counting rules

- Document-level includes only each query’s source article. Cross-document-only includes other judged articles in the same split. Corpus-level is their union.
- Each sentence in a valid judged article pair receives its Direct or Partial label; remaining sentences count as Non-supporting under the final annotation contract.
- Unknown article pairs are excluded from all three sentence-label counts. Their number is recorded in the CSV; no unjudged article is assigned a negative label.
- Query counts refer to queries with at least one judged article in that scope. They need not match across scopes.
- Q1–Q4 share source-article evidence labels; cross-document labels are query-specific.
