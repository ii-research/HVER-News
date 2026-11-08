<div align="center">

# HVER-News: Benchmarking Sentence-Level Evidence Ranking for Actor-Specific Human Value Judgments

### Human-value evidence ranking in news

Rank sentence evidence for **who acts**, **which value is involved**, and **whether the action aligns with or contradicts that value**.

**2,235 articles · 88,895 sentences · 24,400 relations · 97,600 queries**

[Explore the data](dataset/README.md) · [Run experiments](code/README.md) · [Try annotation](annotation/README.md) · [View statistics](dataset/statistics/README.md)

</div>

---

## What is HVER-News?

HVER-News is a benchmark for ranking sentence evidence for actor–value–direction relations in news. Each relation has four query formulations, varying how explicitly the actor and value are expressed. Sentence judgments distinguish **Direct**, **Partial/contextual**, and **Non-supporting** evidence.

The benchmark provides two tasks:

| Task | Ranking scope | Dataset |
|---|---|---|
| **Document-level (DER)** | Rank evidence sentences within the query’s source article. | [document-level/](dataset/document-level) |
| **Corpus-level (CER)** | Rank evidence sentences across the split’s article collection, including the source article and other articles. | [corpus-level/](dataset/corpus-level) |

Cross-document judgments extend the source-article evidence; they are part of the corpus-level view, rather than a separate third task.

See the [dataset overview and real examples](dataset/README.md) for query formulations, article contexts, and sentence evidence labels.

## Start here

| I want to… | Go to |
|---|---|
| **Download and inspect the benchmark** | [Quick start](#quick-start) → [data format](dataset/README.md) |
| **Compare evidence distributions** | [Counts by query type, split, and document scope](dataset/statistics/README.md) |
| **Train or evaluate a model** | [Experiment guide](code/README.md) → [experimental setup](code/EXPERIMENTAL_SETUP.md) |
| **Inspect model settings or API prompts** | [Training configuration](code/configs/experiment_settings.json) · [ranking prompts](code/prompts/README.md) |
| **Use the annotation and assessment interface** | [Annotation guide](annotation/README.md) |

## Quick start

### 1. Download the data

The dataset uses **Git LFS**. Install Git LFS before cloning so the JSONL/TSV files download as data rather than pointer files.

```bash
git lfs install
git clone https://github.com/ii-research/HVER-News.git
cd HVER-News
git lfs pull
```

### 2. Inspect a query

No machine-learning dependencies are needed to read the dataset:

```python
import json
from pathlib import Path

split = Path("dataset/corpus-level/test")
with (split / "queries.jsonl").open(encoding="utf-8") as stream:
    query = json.loads(next(stream))

print(query["query_type"], query["query"])
print("Source article:", query["source_article_id"])
```

### 3. Choose your workflow

- **Experiments:** follow the [code setup and baseline commands](code/README.md). Python 3.11+ is recommended; model checkpoints are downloaded separately.
- **Annotation demo:** run the command below, then open **http://127.0.0.1:8015/**.

```bash
cd annotation
python server_crossdoc_relations.py --port 8015
```

The annotation demo includes **small example inputs only**. Its eight modules cover evidence annotation and quality assessment. Accounts and submitted annotations are excluded. Starting the demo does not call a commercial model API.

## Queries and evidence labels

### Four ways to express a relation

| Type | Actor wording | Value wording |
|---|---|---|
| **Q1** | Explicit | Explicit value label |
| **Q2** | Indirect | Explicit value label |
| **Q3** | Explicit | Natural-language expression |
| **Q4** | Indirect | Natural-language expression |

Implicit wording preserves the intended relation; it does not automatically broaden the query to other actors. Q1–Q4 share source-article evidence labels, while cross-document judgments are query-specific.

### Three evidence grades

| Grade | Label | Interpretation |
|---|---|---|
| **2** | Direct | Supports the specified actor–value–direction relation. |
| **1** | Partial/contextual | Provides specific supporting context without independently establishing the relation. |
| **0** | Non-supporting | Does not support the specified relation. |

Value direction and evidence relevance are distinct: a sentence can directly support a relation whose direction is **contradictory**. An absent cross-document judgment is **unjudged**, not an explicit negative; see the [labeling and evaluation conventions](dataset/README.md).

## Benchmark at a glance

| Split | Articles | Sentences | Relations | Queries |
|---|---:|---:|---:|---:|
| Train | 1,469 | 59,424 | 16,355 | 65,420 |
| Dev | 236 | 9,565 | 2,506 | 10,024 |
| Test | 530 | 19,906 | 5,539 | 22,156 |
| **Total** | **2,235** | **88,895** | **24,400** | **97,600** |

For Direct, Partial, and Non-supporting **query–sentence counts**, see the [detailed tables](dataset/statistics/README.md) or [download the CSV](dataset/statistics/evidence_by_query_type.csv). These distinguish document-level, cross-document-only, and corpus-level evidence.

## Supplementary Results

The [appendix](appendix/README.md) provides corpus-level article-ranking results, seed-42 interaction ablations, and downloadable result tables. Full [training settings](code/EXPERIMENTAL_SETUP.md#training-settings-for-hver-news) cover both evidence-ranking tasks.

## Repository map

```text
HVER-News/
├── dataset/
│   ├── document-level/    # Train / dev / test: source-article evidence
│   ├── corpus-level/      # Train / dev / test: integrated evidence
│   └── statistics/        # Evidence counts and summary tables
├── code/                  # Training, ranking, evaluation, and API prompts
├── appendix/              # Supplementary results and numeric tables
└── annotation/            # English annotation and quality-assessment interface
    ├── index.html         # Unified home page
    └── example_data/      # Small demonstration inputs
```

<details>
<summary><strong>Which dataset files should I use?</strong></summary>

<br>

| File | Purpose |
|---|---|
| `queries.jsonl` | Query text, type, relation, and source article |
| `articles.jsonl` | Articles with their sentence lists |
| `corpus.jsonl` | Individual sentence units |
| `instances.jsonl` | Query–article records and evidence sentence IDs |
| `qrels.tsv` | Document-level sentence relevance judgments |
| `qrels.sentences.tsv` | Corpus-level sentence relevance judgments |
| `qrels.documents.tsv` | Corpus-level query–article relevance judgments |

Use the file names and schema for the task you are running. The document-level and corpus-level loaders use different instance formats; the [dataset guide](dataset/README.md) explains their relationship.

</details>

<details>
<summary><strong>What is included for reproducibility?</strong></summary>

<br>

- [Experiment commands and evaluation protocol](code/README.md)
- [Consolidated experimental setup](code/EXPERIMENTAL_SETUP.md)
- [Model-specific training settings](code/configs/experiment_settings.json)
- [Shared GPT/DeepSeek ranking prompts](code/prompts/README.md)
- [Annotation and quality-assessment interface](annotation/README.md)

Model weights, environment snapshots, private credentials, and full prediction outputs are not bundled. Optional API baselines read credentials from environment variables.

</details>

## Attribution and licensing

HVER-News derives from **NEVU** and retains news article text and identifiers. News content and model checkpoints remain subject to their respective rights and upstream terms. The upstream Provence license is included in [UPSTREAM_LICENSE](code/provence_bundle/UPSTREAM_LICENSE).
