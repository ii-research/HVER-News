# Supplementary Corpus-Level Article Ranking Results

These results evaluate article ranking in the corpus-level evidence-ranking pipeline, including the source article and other articles in the same split. They are distinct from document-level sentence ranking and from the cross-document-only diagnostic subset.

All scores are percentages. Fine-tuned main results report mean ± standard deviation over seeds 13, 42 and 87; ablations use seed 42. Bold marks column maxima within each table, including ties. ZS denotes zero-shot and FT denotes fine-tuning. SI and DI denote sentence interaction and document interaction.

## Main results

| Method | nDCG@1 | nDCG@5 | nDCG@10 | MRR@10 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 52.12 | 60.50 | 62.61 | 60.36 | 68.35 | 74.48 |
| E5-base-v2 | 64.21 | 73.25 | 75.29 | 72.95 | 81.41 | 87.13 |
| BGE-M3 | 59.82 | 69.82 | 71.97 | 69.16 | 78.52 | 84.57 |
| Qwen3-Emb.-0.6B | 66.10 | 74.63 | 76.54 | 74.36 | 82.34 | 87.68 |
| BGE-Reranker-v2-M3 | 62.98 | 72.99 | 75.05 | 72.34 | 81.79 | 87.51 |
| Provence (ZS) | 43.87 | 60.79 | 64.06 | 58.47 | 75.60 | 84.97 |
| E5 (FT) | 66.13 ± 1.37 | 76.59 ± 0.89 | 78.74 ± 0.83 | 75.89 ± 1.06 | 85.98 ± 0.55 | **91.99 ± 0.39** |
| Qwen3 (FT) | 65.11 ± 0.41 | 75.83 ± 0.09 | 78.02 ± 0.07 | 75.04 ± 0.20 | 85.51 ± 0.07 | 91.61 ± 0.25 |
| Provence (FT) | 55.50 ± 0.24 | 70.54 ± 0.11 | 72.49 ± 0.11 | 68.26 ± 0.17 | 83.39 ± 0.03 | 88.64 ± 0.02 |
| HVER-News | **81.36 ± 0.23** | **84.95 ± 0.08** | **85.34 ± 0.08** | **85.68 ± 0.13** | **88.39 ± 0.04** | 89.43 ± 0.03 |

## Ablation study (seed 42)

| Method | nDCG@1 | nDCG@5 | nDCG@10 | MRR@10 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| HVER-News | **81.61** | **85.03** | **85.43** | **85.82** | 88.35 | **89.40** |
| w/o SI | 80.93 | 84.77 | 85.16 | 85.45 | 88.33 | 89.37 |
| w/o DI | 81.24 | 84.90 | 85.27 | 85.62 | 88.35 | 89.36 |
| w/o SI & DI | 80.97 | 84.79 | 85.18 | 85.45 | **88.37** | **89.40** |

## Interpretation and protocol

HVER-News has the highest main-result nDCG and MRR scores and Recall@5, while E5 (FT) has the highest Recall@10. The seed-42 article-ranking differences among interaction variants are small; these ablations should not be interpreted as establishing statistical significance.

Article-ranking results use the corpus-level evaluator's `document` output, not `cross_only_document`. Candidate depths differ by baseline; see the [experimental setup](../code/EXPERIMENTAL_SETUP.md) and [execution commands](../code/README.md). These are the supplied experiment results, not a new run.

Download the [numeric CSV](tables/corpus_level_article_ranking.csv). Blank SD fields mean that no across-seed SD is reported; they do not indicate zero variance. Blank training-seed fields denote methods without fine-tuning in this table.
