# Document-level ranking prompt

The official shared prompt for GPT and DeepSeek document-level evidence ranking. It includes an analyst role, actor/value/direction-aware ranking rules, and an ID-only JSON response. Both runners load these files automatically.

## Files and parameters

| File or parameter | Meaning |
|---|---|
| `system.txt` | Analyst role, input definitions, ranking rules, and output contract |
| `user_template.txt` | Per-request query and candidate sentences |
| `output_schema.json` | Response schema with the `ranked_sen_ids` field |
| `{query}` | Natural-language query; no separate gold actor/value/direction fields |
| `{candidate_sentences}` | Original-order lines: `[000] ID="s1" TEXT=...` |
| `ranked_sen_ids` | Ordered sentence IDs; at most 10 |

The JSON braces in the user template are doubled for Python `str.format`; the rendered request contains ordinary braces. The example sentence IDs illustrate output structure only and must be replaced by actual candidate IDs. The number of returned IDs is determined by the candidate count, not by the illustrative three-item array.

Requests use article-local IDs such as `s1`; parsers restore complete sentence IDs using the request candidate manifest before evaluation. Saved predictions use `ranked_sen_ids` with complete IDs to prevent cross-article collisions. Ambiguous or unknown short IDs are rejected. The prompt does not request generated rationales or introduce annotation metadata as inference inputs.

## Offline preview

From the repository root:

```python
from pathlib import Path

folder = Path("code/prompts/document_level_ranking")
print((folder / "system.txt").read_text())
print((folder / "user_template.txt").read_text().format(
    query="Which sentences show the fire department helping residents?",
    candidate_sentences=(
        '[000] ID="s0" TEXT=Fire crews rescued residents from the building.\n'
        '[001] ID="s1" TEXT=The building opened in 1985.'
    ),
))
```

This is a synthetic formatting example. It makes no API calls.

For both GPT and DeepSeek, prepare a new output directory when changing prompts. GPT executes its frozen request files; DeepSeek verifies the prepared prompt fingerprint before running.
