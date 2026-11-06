# Annotation and quality assessment

This directory contains an English-localized copy of the existing HVER-News annotation system. It reuses the original pages, routes, evidence editor, grouped relation views, storage implementation and administrator dashboard.

## Task scope

Document-level modules assess or annotate evidence in a query's source article. Cross-document modules assess or annotate **cross-document-only** query–article pairs: the candidate article differs from the source article. Those judgments supplement source-article judgments to form the **corpus-level** dataset; cross-document-only is not a separate benchmark task.

`server_crossdoc_relations.py` serves the unified dashboard, including both document-level and cross-document-only modules. The `crossdoc_*` filenames and routes identify annotation components, not the full corpus-level experimental pipeline. Experiments use `code/document_level_evidence_ranking/` and `code/corpus_level_evidence_ranking/`.

## Run locally

Python 3.11 or later is recommended. The server uses the Python standard library.

```bash
cd annotation
python server_crossdoc_relations.py --port 8015
```

Open http://127.0.0.1:8015/ and create a local annotation account. Accounts from the production system are not included. Starting this server does not invoke any commercial model API.

For the administrator dashboard, in another terminal:

```bash
cd annotation
# Set a private token in your shell; do not commit it.
export NEVU_ADMIN_TOKEN='replace-with-your-private-token'
python admin_server_crossdoc.py --host 127.0.0.1 --port 8016
```

Open http://127.0.0.1:8016/ and provide the configured token. Both servers bind to localhost by default.

## Modules

| Module | Purpose | Route |
|---|---|---|
| Original annotation | Original Stage 1 assessment and Stage 2 evidence annotation | `/original/` |
| Post-annotation quality | Document-level quality reassessment | `/quality-review/` |
| Remaining annotation | Document-level Stage 2 annotation, example Group 41 | `/remaining/` |
| Cross-document-only pre-assessment | Quality assessment before evidence annotation | `/crossdoc-quality/` |
| Cross-document-only Direct | Evidence annotation for pairs with Direct evidence | `/crossdoc-rel-direct/` |
| Cross-document-only Partial | Evidence annotation for pairs with Partial evidence only | `/crossdoc-rel-partial/` |
| Cross-document-only no evidence | Optional review of pairs without positive evidence | `/crossdoc-rel-empty/` |
| Cross-document-only post-assessment | Quality reassessment of the same sampled pairs | `/crossdoc-quality-after/` |

Use the dashboard links to open a group. Cross-document annotation groups combine queries and candidate articles under their source relation. Direct and Partial selections remain query-specific. Existing confirmation, invalid-item handling and navigation checks are retained.

## Example-only data

This copy contains only a small demonstration sample, not the full annotation collection. Each document-level workflow retains two relations; each grouped cross-document category retains two relations (4 Direct, 2 Partial-only and 8 no-evidence query–article pairs in total). Quality-assessment pages retain two pairs each. Initial evidence selections in these examples demonstrate the editor; they are not submitted annotations. See `EXAMPLE_MANIFEST.json` for counts.

All complete input collections, submitted annotations, accounts and saved drafts have been removed from this copy.

## Included and excluded data

Included: task input pages, articles, queries, initial evidence selections, group indexes and registries needed by the existing application. Pre- and post-assessment modules retain their respective task inputs.

Excluded: production accounts, submitted annotation histories, annotator identities, API request logs, credentials and private deployment files. Optional Fable suggestion payloads are empty; the existing interface can display suggestions when supplied later. Initial evidence selections are task inputs, not new human submissions.

New accounts and annotations are written locally to runtime stores. The existing storage grouping is retained, including shards of 20 storage groups for cross-document records. A displayed relation-page group and an underlying storage group are not necessarily identical. Runtime stores are excluded by `.gitignore`.

## Layout

- `server_crossdoc_relations.py`: annotation server entry point.
- `admin_server_crossdoc.py`: administrator entry point.
- `crossdoc_addon/`: cross-document views, stores and administrator extension.
- `example_data/`: original grouped task inputs and supporting indexes.
- Root HTML/JavaScript files: shared dashboard, authentication, reference and administrator interfaces.

The original local annotation system and remote deployment are separate from this copy. Editing or running this directory does not update either of them.

## Directory guide

`index.html` is the annotation and quality-assessment home page, served by the annotation server (open the HTTP address rather than the file directly).

```text
annotation/
├── index.html                       # Unified task dashboard
├── dashboard_template.html          # Reusable dashboard template
├── example_data/                    # Small examples only
│   ├── pages/                       # Original annotation examples
│   ├── document_annotation/         # Document-level evidence annotation
│   ├── document_quality/            # Document-level quality assessment
│   ├── cross_document_review/       # Pre/post quality and pair review
│   ├── cross_document_evidence/     # Evidence categories and save registries
│   └── cross_document_relations/    # Grouped relation annotation examples
└── crossdoc_addon/                  # Cross-document UI and storage code
```

Directory names describe the purpose of each module.
