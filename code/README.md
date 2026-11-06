# Code and reproduction

See [Experimental Setup](EXPERIMENTAL_SETUP.md) for the consolidated protocol, model settings, evaluation definitions and reproducibility limitations.

Run all commands from this `code/` directory. Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/provence_bundle/scripts:$PWD/provence_bundle/scripts/provence_final_suite:${PYTHONPATH:-}"
```

Install a CUDA-compatible PyTorch build for your hardware before training. The dependency file gives compatibility requirements, not a frozen reproduction environment; this repository does not include environment snapshots. Optional API baselines require their own SDKs (`openai`) and credentials supplied through environment variables. They are not run by any setup command.

## Document-level baselines

```bash
python -m avr_retriever_experiment.run_document_level \
  --setting bm25 --data-dir ../dataset/document-level --device cpu
python -m avr_retriever_experiment.run_document_level \
  --setting e5_generic_ft --train-seed 42 --data-dir ../dataset/document-level --device cuda
```

Other settings: e5_zero_shot, bge_m3_zero_shot, qwen3_zero_shot, qwen3_generic_ft. Repeat training for seeds 13 and 87 where required. Model identifiers and sampling recipes are defined in `avr_retriever_experiment/settings.py`. Results and checkpoints go to generated results/models directories.

## Provence initialization and document-level HVER-News

Download the official initialization separately (this command uses the model host, not a paid inference API):

```bash
hf download naver/provence-reranker-debertav3-v1 \
  --revision ef49e233e3c6e50efc476c68f1390f8a63add4d4 \
  --local-dir checkpoints/provence_official
python tools/prepare_document_level.py --checkpoint checkpoints/provence_official
```

Preparation generates the balanced train/dev query views, all-four-formulation Test manifest, fixed negative samples and executable seed-specific configs. The pinned checkpoint must include its tokenizer, config, custom model code and weights. The loader uses the upstream checkpoint's model code.

Run smoke validation before formal training, using the same GPU count for both:

```bash
python provence_bundle/scripts/provence_final_suite/run.py \
  --config provence_bundle/configs/without_pe_42_smoke.json --smoke
python provence_bundle/scripts/provence_final_suite/run.py \
  --config provence_bundle/configs/without_pe_42.json
```

`without_pe` is the published document-level HVER-News. Replace it with `adapted_provence` for w/o SI. Repeat for seeds 13 and 87. For multiple GPUs, run both commands through `torchrun --standalone --nproc_per_node=N`.

The separate official document-level Provence Generic FT baseline uses the original binary pruning head, **not** the adapted sentence head:

```bash
for stage in zero-reference prepare preflight smoke train eval-full; do
  python tools/run_provence_official.py "$stage" \
    --bundle-root "$PWD/provence_bundle" --project-root "$PWD" \
    --checkpoint "$PWD/checkpoints/provence_official" \
    --results-root "$PWD/results/document-level" --seed 42
done
```

Use `zero-full` as the stage to evaluate official Provence zero-shot on all four formulations.

## Corpus-level retrieval

First-stage BGE-M3 rankings for reranking:

```bash
for split in dev test; do
  python -m corpus_evidence_ranking.evaluate --setting dense --model BAAI/bge-m3 \
    --data-dir ../dataset/corpus-level --split "$split" \
    --results-dir "results/candidates/$split" --device cuda
done
```

The resulting `predictions.jsonl` files can be supplied directly as candidate files; rerankers select the first 20 articles. Other dense baselines use the same entry point with their model ID. BM25 uses `--setting bm25` and no model argument.

Dense Generic FT:

```bash
python -m corpus_evidence_ranking.train --model intfloat/e5-base-v2 \
  --data-dir ../dataset/corpus-level --output-dir models/corpus-level/e5/seed_42 --seed 42
```

Corpus-level Provence Generic FT:

```bash
python -m corpus_evidence_ranking.provence_train \
  --data-dir ../dataset/corpus-level --output-dir models/corpus-level/provence/seed_42 \
  --bundle provence_bundle --checkpoint checkpoints/provence_official \
  --variant adapted_provence --seed 42 \
  --dev-candidates results/candidates/dev/predictions.jsonl \
  --test-candidates results/candidates/test/predictions.jsonl
```

Corpus-level hierarchical HVER-News (four GPUs, effective query-set batch 4):

```bash
torchrun --standalone --nproc_per_node=4 -m corpus_evidence_ranking.provence_hierarchical_train \
  --data-dir ../dataset/corpus-level --output-dir models/corpus-level/hierarchical/seed_42 \
  --bundle provence_bundle --checkpoint checkpoints/provence_official \
  --variant hierarchical --seed 42 --query-view-seed 42 \
  --dev-candidates results/candidates/dev/predictions.jsonl \
  --query-set-batch-size 1 --block-batch-size 4 --negative-documents 3 \
  --document-loss-weight 0.25
python -m corpus_evidence_ranking.provence_hierarchical_infer \
  --data-dir ../dataset/corpus-level --split test --output-dir results/corpus-level/hierarchical/seed_42 \
  --bundle provence_bundle --checkpoint checkpoints/provence_official \
  --trained-checkpoint models/corpus-level/hierarchical/seed_42/best_checkpoint/model.safetensors \
  --training-run-dir models/corpus-level/hierarchical/seed_42 \
  --variant hierarchical --seed 42 --candidates results/candidates/test/predictions.jsonl
```

For one GPU use plain `python -m` and `--query-set-batch-size 4`. Component ablations use `sentence_only` (w/o DI), `document_only` (w/o SI), or `no_interaction` (w/o both); give each a different output directory. Use matching seed 42 for component comparisons. `--help` lists optional smoke and sharding arguments.

See the protocol below and `configs/experiment_settings.json` for training targets, hyperparameters and metric conventions. Model checkpoints and prediction outputs are generated by the commands above.

## Parameters and evaluation protocol

### Training settings

Use seeds 13, 42 and 87 for the three-seed main experiments, and seed 42 for corpus-level component ablations. Compare ablations against the full model at the same seed. Hyperparameters are also provided in `configs/experiment_settings.json`.

Common: 3 epochs, weight decay 0.01, warmup ratio 0.06; training/dev select one balanced query formulation per relation with view seed 42. Test evaluates all 22,156 queries. Checkpoint selection uses dev sentence nDCG@10 except corpus-level Provence Generic FT, which uses dev document nDCG@10.

| System | Learning rate | Batch / supervision |
|---|---|---|
| E5 / Qwen3 Generic FT | 2e-5 | Global batch 16; temperature .05; in-batch contrastive + pairwise logistic (weight 1) |
| Document-level Provence Generic FT | 3e-6 | Effective batch 16; official binary pruning head; token CE averaged within sentence, class-balanced within query |
| Document-level HVER-News / w/o SI | encoder 3e-6; head 3e-5 | Effective batch 16; width 256; class-balanced sentence CE; fixed ordinary negatives (up to 3), sampling seed 42 |
| Corpus-level Provence Generic FT | encoder 3e-6; head 3e-5 | Batch 16; adapted sentence head; class-balanced sentence CE; one eligible positive article/query/epoch |
| Corpus-level hierarchical HVER-News | encoder 3e-6; head 3e-5; interactions 3e-6 | Effective query-set batch 4 in the recorded four-GPU run; one positive article and up to 3 explicit negative articles; sentence CE + .25 document BCE |

Dense Generic FT uses Direct positives, excludes Partial from positive/negative sampling, and uses legal Non-supporting negatives. Document-level official Provence maps sampled Direct sentences to positive binary token targets and ordinary negatives to negative targets; Partial provides context but no loss. The adapted sentence-head variants use targets 2 and 0 and mask target 1. No nonzero auxiliary ranking loss is applied in the reported Provence-based variants.

Corpus-level hierarchical training selects Direct-containing positive articles and explicitly judged negative articles, not arbitrary unjudged articles. Corpus-level ranking uses a BGE-M3 first-stage candidate depth of 20. The hierarchical model uses its final sentence scores; it is not the independent document-level score.

Ablation mapping: hierarchical=SI+DI; sentence_only=w/o DI; document_only=w/o SI; no_interaction=w/o SI & DI. Document-level `without_pe` is the published full method (SI, no position encoding), and `adapted_provence` is its w/o SI control; these are the implementation identifiers used in the executable configurations.

### Metrics

nDCG uses grades 2/1/0 and gain 2^rel-1. Recall and MRR count Direct only. Corpus evaluator retains unjudged items at their retrieved ranks and assigns zero gain, rather than deleting them. It averages only queries with at least one Direct qrel. Cross-only removes source-article results/qrels and applies that eligibility rule: 3,305 Test queries, versus 4,049 with any positive non-source evidence (Direct or Partial).

The source files are authoritative: `corpus_evidence_ranking/metrics.py`, `avr_retriever_experiment/evaluation.py`, `provence_bundle/scripts/provence_finetune.py`, `provence_bundle/scripts/provence_final_suite/model.py`, and `corpus_evidence_ranking/provence_hierarchical_model.py`.


## Additional published baselines

The BGE cross-encoder baseline ranks all source-article sentences document-level:

```bash
python -m avr_retriever_experiment.run_cross_encoder_reranker \
  --project-root "$PWD" --test-dir ../dataset/document-level/test \
  --output-dir results/document-level/bge_reranker --model-name BAAI/bge-reranker-v2-m3 \
  --device cuda --batch-size 128 --max-length 256
```

For corpus-level BGE cross-encoder ranking, the reported candidate depth is 100:

```bash
python -m corpus_evidence_ranking.rerank \
  --model BAAI/bge-reranker-v2-m3 --data-dir ../dataset/corpus-level \
  --first-stage results/candidates/test/predictions.jsonl \
  --results-dir results/corpus-level/bge_reranker --candidate-depth 100 --batch-size 128
```

Corpus-level official Provence zero-shot uses candidate depth 20:

```bash
python -m corpus_evidence_ranking.provence_rerank \
  --data-dir ../dataset/corpus-level --split test \
  --bundle provence_bundle --checkpoint checkpoints/provence_official \
  --candidates results/candidates/test/predictions.jsonl \
  --results-dir results/corpus-level/provence_zero_shot --variant official \
  --candidate-depth 20 --batch-size 8
```

For corpus-level Qwen3 Generic FT, replace `intfloat/e5-base-v2` with
`Qwen/Qwen3-Embedding-0.6B` in the dense training command and give it a separate output directory.
Use the same three seeds. For zero-shot dense results, use `corpus_evidence_ranking.evaluate`
with the E5, BGE-M3 or Qwen3 model identifier and `--split test`.

## Document-level API baselines

The current official commercial-model prompts are published in [`prompts/README.md`](prompts/README.md):
[`system.txt`](prompts/document_level_ranking/system.txt),
[`user_template.txt`](prompts/document_level_ranking/user_template.txt), and
[`output_schema.json`](prompts/document_level_ranking/output_schema.json).
GPT and DeepSeek read the same files through `avr_retriever_experiment/api_prompts.py`.
Each request supplies one query and ordered source-article sentences and returns at most ten sentence IDs; no gold labels or original rationales are supplied. These are ranking-experiment prompts, not dataset annotation prompts. The prompt README includes an offline preview example.

These optional commands send requests to external providers and incur API charges.
Install `openai` and set `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` in your environment;
never put a key in this repository. Provider model names may no longer be available.
The published runs used GPT-6 Luna and DeepSeek Flash. These baselines return top-ten
rankings rather than complete sentence score matrices.

```bash
python -m pip install openai
python -m avr_retriever_experiment.run_openai_llm_baseline prepare \
  --project-root "$PWD" --output results/document-level/gpt6_luna --model gpt-6-luna
python -m avr_retriever_experiment.run_openai_llm_baseline submit \
  --project-root "$PWD" --output results/document-level/gpt6_luna --max-new-jobs 12
python -m avr_retriever_experiment.run_openai_llm_baseline status \
  --project-root "$PWD" --output results/document-level/gpt6_luna
# Run download after the submitted batches have finished, then evaluate.
python -m avr_retriever_experiment.run_openai_llm_baseline download \
  --project-root "$PWD" --output results/document-level/gpt6_luna
python -m avr_retriever_experiment.run_openai_llm_baseline evaluate \
  --project-root "$PWD" --output results/document-level/gpt6_luna

python -m avr_retriever_experiment.run_deepseek_llm_baseline prepare \
  --project-root "$PWD" --output results/document-level/deepseek_flash --model deepseek-flash
python -m avr_retriever_experiment.run_deepseek_llm_baseline run \
  --project-root "$PWD" --output results/document-level/deepseek_flash --model deepseek-flash \
  --concurrency 32 --max-output-tokens 1024
python -m avr_retriever_experiment.run_deepseek_llm_baseline evaluate \
  --project-root "$PWD" --output results/document-level/deepseek_flash
```

## Outputs, seeds and component comparisons

Training commands save their run configuration and development history together with
selected weights. Test commands save `predictions.jsonl` and `metrics.json` (the official
document-level Provence runner uses `formal_metrics.json`). Dense document-level runs are under
`results/document-level/G3/<model>/seed_<seed>` by default. Keep separate directories
for every seed and model; evaluation scripts may refuse a non-empty output directory.

For the corpus-level hierarchical ablations, repeat **both** training and inference with the same
variant and a distinct path:

| Paper name | `--variant` | Comparison seed |
|---|---|---|
| HVER-News | `hierarchical` | 13, 42, 87 |
| w/o DI | `sentence_only` | 42 |
| w/o SI | `document_only` | 42 |
| w/o SI & DI | `no_interaction` | 42 |

Document-level HVER-News and w/o SI use the generated `without_pe_<seed>.json` and
`adapted_provence_<seed>.json` configurations, respectively, for seeds 13, 42 and 87.
The strings are implementation identifiers for the two published configurations;
no positional-encoding or old D-series model variants are exposed.

Use the arithmetic mean of seed-level metrics for three-seed main results. If reporting
variability, use sample standard deviation (ddof=1). Single-run baselines have no seed
standard deviation. Never replace a failed or missing run with zero.

The document-level negative sampler retains the published eligibility rule: its ordinary pool
excludes evidence for controlled actor/value/direction contrasts as well as the target's
Direct/Partial evidence. This filtering is part of the Generic FT baseline, not an extra
trainable model. The dense training loop preserves an additional encoder pass so that
dropout random-number consumption matches the reported implementation.

### Architecture and optimization details

HVER-News projects sentence representations to 256 dimensions. SI uses two
Transformer encoder layers with four attention heads, feed-forward width 1,024
and dropout 0.1. The corpus-level DI module uses one layer with the same width,
head count and dropout. The corpus-level residual interaction gates start at zero.
No sentence positional encoding is added. Sentence ranking scores are the
three-way output probabilities weighted by `[0, 1/3, 1]`.

Training uses AdamW and a linear warmup/decay schedule; gradient norms are clipped
to 1.0. Dense encoders use query/passage token limits 128/256. Provence packs
query-conditioned token windows and pools all sentence tokens, including fragments
of oversized sentences. Partial evidence stays in the input context even when
masked from supervision. BF16 requires compatible GPUs; exact floating-point
results can vary across hardware/software versions.

## Task paths

Use `dataset/document-level` and `dataset/corpus-level`. Experiment outputs use `results/document-level`, `models/document-level`, `results/corpus-level` and `models/corpus-level`.

The document-level entry points are `avr_retriever_experiment.run_document_level` and `tools/prepare_document_level.py`. The corpus-level implementation is in `corpus_evidence_ranking`.

### API output format

GPT and DeepSeek automatically load the shared [ranking prompt](prompts/document_level_ranking/README.md). Model responses use `ranked_sen_ids` with local sentence IDs such as `s3`; parsing restores full IDs before saving and scoring. Prepare a new output directory rather than resuming requests prepared with a different prompt.
