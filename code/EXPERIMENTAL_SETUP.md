# Experimental Setup

This document summarizes the released experiment protocol. Executable commands are in [README.md](README.md); numeric settings are in [experiment_settings.json](configs/experiment_settings.json). Changing task names does not change the model or evaluation protocol.

## Tasks and data

- **Document-level evidence ranking:** rank all sentences in the query's source article using `dataset/document-level`. Use its `qrels.tsv` and document-level loader.
- **Corpus-level evidence ranking:** rank evidence across articles using `dataset/corpus-level`. Use `qrels.sentences.tsv` for sentence evaluation and `qrels.documents.tsv` for document evaluation. Dense/sparse first-stage ranking and candidate-based reranking are distinct configurations.

| Split | Articles | Sentences | Relations | Queries |
|---|---:|---:|---:|---:|
| Train | 1,469 | 59,424 | 16,355 | 65,420 |
| Development | 236 | 9,565 | 2,506 | 10,024 |
| Test | 530 | 19,906 | 5,539 | 22,156 |

Training and checkpoint selection use one balanced query formulation per relation, fixed by query-view seed 42. Test evaluates all four formulations, with aggregate and Q1–Q4 results. Article collections remain split-specific. The test set is not used for checkpoint selection. Structured actor/value/direction labels and original annotation rationales are not additional model inputs.

## Systems and candidate settings

| System | Released implementation / setting |
|---|---|
| BM25 | Sparse baseline |
| E5 | `intfloat/e5-base-v2`; zero-shot and Generic FT |
| BGE-M3 | `BAAI/bge-m3`; zero-shot, and first-stage candidates for hierarchical ranking |
| Qwen3 embedding | `Qwen/Qwen3-Embedding-0.6B`; zero-shot and Generic FT |
| BGE reranker | `BAAI/bge-reranker-v2-m3`; all source sentences for document-level; candidate depth 100 for corpus-level |
| Provence | Official zero-shot / Generic FT and adapted sentence-head variants; checkpoint supplied separately |
| HVER-News | Sentence interaction (SI); corpus-level adds document interaction (DI) |
| Commercial API baselines | GPT-6 Luna and DeepSeek Flash; document-level top-ten sentence IDs, [exact prompts](prompts/README.md) |

Corpus-level Provence zero-shot and hierarchical HVER-News use candidate depth 20. Candidate depth is an inference setting and is not the E5 actor-description top-ten threshold used during dataset construction. Follow the [commands](README.md#parameters-and-evaluation-protocol) to generate first-stage files before reranking.

## Optimization and supervision

Shared fine-tuning settings: 3 epochs, AdamW, weight decay 0.01, linear warmup/decay with warmup ratio 0.06, and gradient clipping at 1.0. Main fine-tuned results use seeds 13, 42 and 87. Corpus-level component ablations use seed 42.

| System | Learning rates | Batch and objective |
|---|---|---|
| E5 / Qwen3 Generic FT | Encoder `2e-5` | Global batch 16; contrastive temperature 0.05; in-batch contrastive plus pairwise logistic loss, weight 1 |
| Document-level Provence Generic FT | Encoder `3e-6` | Effective batch 16; official binary token head; sentence-averaged, class-balanced token CE |
| Document-level HVER-News / w/o SI | Encoder `3e-6`, head `3e-5` | Effective batch 16; class-balanced sentence CE; up to 3 sampled ordinary negatives |
| Corpus-level Provence Generic FT | Encoder `3e-6`, head `3e-5` | Batch 16; adapted sentence head; one eligible positive article per query per epoch |
| Corpus-level HVER-News | Encoder `3e-6`, head `3e-5`, interactions `3e-6` | Effective query-set batch 4 in the recorded four-GPU setup; one positive and up to 3 explicitly negative articles; sentence CE + 0.25 document BCE |

Direct sentences supply positive supervision. Partial evidence stays in context but is excluded from dense positive/negative sampling and masked from adapted sentence-head supervision. The adapted sentence head uses class targets 2 and 0. Document-level ordinary negatives also exclude evidence for controlled actor/value/direction contrasts. Corpus-level negative articles must have explicit negative judgments. No nonzero auxiliary ranking loss is used in the reported Provence variants.

Dense token limits are 128 for queries and 256 for passages. Provence uses query-conditioned token windows and pools sentence tokens, including oversized-sentence fragments. HVER-News projects representations to width 256. SI has two Transformer encoder layers; DI has one. Both use four attention heads, feed-forward width 1,024 and dropout 0.1. No sentence positional encoding is added. Corpus-level residual gates start at zero. Sentence scores weight three-way probabilities by `[0, 1/3, 1]`.

Checkpoint selection uses development sentence nDCG@10, except corpus-level Provence Generic FT, which uses development document nDCG@10.

## Ablations

| Task | Paper configuration | Implementation identifier | Seeds |
|---|---|---|---|
| Document-level | HVER-News | `without_pe` | 13, 42, 87 |
| Document-level | w/o SI | `adapted_provence` | 13, 42, 87 |
| Corpus-level | HVER-News | `hierarchical` | 13, 42, 87 |
| Corpus-level | w/o DI | `sentence_only` | 42 |
| Corpus-level | w/o SI | `document_only` | 42 |
| Corpus-level | w/o SI & DI | `no_interaction` | 42 |

Train and infer with the same variant. Compare seed-42 ablations with the seed-42 full model. Existing implementation identifiers remain unchanged for checkpoint compatibility.

## Evaluation and aggregation

Report nDCG and Recall at 1, 3, 5 and 10, and MRR@10. nDCG uses graded gains `2^relevance - 1`, with Direct=2, Partial=1 and Non-supporting=0. Recall and reciprocal rank count Direct evidence only. The document-level dense evaluator stores MRR@10 under the key `mrr`; the corpus evaluator uses `mrr@10`.

Metrics are macro-averaged over evaluated queries. The corpus evaluator includes queries with at least one Direct qrel and keeps returned items at their original ranks, assigning zero gain when no qrel exists. Cross-only evaluation removes source-article results and qrels, then applies the same eligibility rule. Check exported query counts: missing predictions are not a license to silently reduce evaluation coverage.

For three-seed results, report the arithmetic mean of seed-level metrics and, if needed, sample standard deviation (`ddof=1`). Do not assign a standard deviation to a single run or replace missing runs with zero. API baselines return only top-ten IDs, not full sentence-score matrices.

## Commercial API execution

The released GPT/DeepSeek runners send one query and ordered source-article candidate sentences per request. They share the original prompt and JSON schema, request no explanations, set reasoning effort to `none`, and default to 1,024 output tokens. The OpenAI request manifest leaves temperature unspecified. OpenAI uses batch submission; DeepSeek uses bounded asynchronous calls (the documented command uses concurrency 32) with persistence and resume support. API execution costs money; offline preparation and prompt inspection do not submit requests. See [API commands](README.md#document-level-api-baselines) and [prompt details](prompts/README.md).

## Environment and reproducibility boundary

Use Python 3.11+ and [requirements.txt](requirements.txt); the API runners additionally require `openai`. BF16 needs compatible hardware. Model checkpoints and credentials are not bundled. Preserve generated configurations, dataset fingerprints, query-view manifests and seed-level metrics with each run.

The repository specifies dependency lower bounds, not an exact lockfile for every experiment. It does not establish one uniform GPU model, CUDA version, API model snapshot or wall-clock runtime for all experiments. Report such details from the original run logs rather than inferring them from default code. No training or paid API calls are required to read this setup.

Implementation references: [document runner](avr_retriever_experiment/run_document_level.py), [document metrics](avr_retriever_experiment/evaluation.py), [corpus metrics](corpus_evidence_ranking/metrics.py), [Provence configuration generator](tools/make_provence_configs.py), and [hierarchical model](corpus_evidence_ranking/provence_hierarchical_model.py).
