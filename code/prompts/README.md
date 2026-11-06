# Commercial API baseline prompts

The document-level GPT and DeepSeek ranking baselines share the same prompt and output schema:

- [`document_level_ranking/system.txt`](document_level_ranking/system.txt): ranking instructions, sent as a developer message.
- [`document_level_ranking/user_template.txt`](document_level_ranking/user_template.txt): query and ordered candidate sentences.
- [`document_level_ranking/output_schema.json`](document_level_ranking/output_schema.json): structured response schema.

`avr_retriever_experiment/api_prompts.py` loads this single official prompt for both providers. No variant selection is required.

## Input and output

Each request contains one query and all its source-article candidate sentences, in original article order. Candidate lines have the form `[000] ID="s1" TEXT=...`. Newlines inside sentence text are replaced with spaces. Structured actor/value/direction labels, gold relevance labels, rationales and label-origin metadata are not inserted into the prompt.

The model returns `{"ranked_sen_ids": ["s1", "s2"]}` with at most ten candidate IDs. These IDs identify **sentences**, not articles. Parsers map them back to complete IDs before saving and evaluation. No generated evidence text or explanations are requested. The prompt requests all candidates when fewer than ten exist.

The runners validate and normalize model outputs separately from prompting. In particular, the DeepSeek parser maps valid numeric candidate indices to IDs and fills short lists from remaining candidates in original order; consult its `parse_response` for this evaluation behavior.

These files describe the document-level ranking experiments, not Gemini/Claude dataset-construction or annotation prompts. They do not imply a corpus-level API baseline.

## Inspect without API calls

From `code/`:

```bash
python - <<'PY'
from avr_retriever_experiment.api_prompts import SYSTEM_PROMPT, format_ranking_prompt
print(SYSTEM_PROMPT)
print(format_ranking_prompt(
    "Which statements show the company protecting its employees?",
    ["example::s1", "example::s2"],
    {"example::s1": {"text": "The company installed protective equipment."},
     "example::s2": {"text": "Its headquarters opened in 1998."}},
))
PY
```

The example is synthetic. Printing prompts makes no network calls. Provider execution commands are in [`code/README.md`](../README.md#document-level-api-baselines). Changing the prompt defines a new experimental setting; use a new output directory rather than resuming existing requests.
