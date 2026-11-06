"""Shared API prompts and request-local sentence ID normalization."""
import hashlib
import json
import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts/document_level_ranking"
SYSTEM_PROMPT = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")
USER_TEMPLATE = (PROMPT_DIR / "user_template.txt").read_text(encoding="utf-8")
OUTPUT_SCHEMA = json.loads((PROMPT_DIR / "output_schema.json").read_text(encoding="utf-8"))
PROMPT_FINGERPRINT = hashlib.sha256((SYSTEM_PROMPT + USER_TEMPLATE + json.dumps(OUTPUT_SCHEMA, sort_keys=True)).encode()).hexdigest()


def sentence_id_map(candidates):
    """Map article-local suffixes to canonical IDs; reject ambiguous requests."""
    aliases = {}
    for full_id in candidates:
        short = full_id.rsplit("::", 1)[-1]
        if not re.fullmatch(r"s\d+", short):
            raise ValueError("Candidate sentence ID must end in s<number>")
        if short in aliases:
            raise ValueError(f"Duplicate or ambiguous sentence ID: {short}")
        aliases[short] = full_id
    return aliases


def ranked_sentence_ids(payload):
    """Read sentence rankings and reject conflicting fields."""
    if "ranked_sen_ids" in payload:
        values = payload["ranked_sen_ids"]
        if "ranked_doc_ids" in payload and payload["ranked_doc_ids"] != values:
            raise ValueError("Conflicting ranking fields")
    else:
        values = payload["ranked_doc_ids"]
    if not isinstance(values, list) or not values or not all(isinstance(x, str) for x in values):
        raise ValueError("Ranking must be a nonempty list of sentence ID strings")
    return values


def resolve_sentence_ids(values, candidates, allow_indices=False):
    """Restore canonical IDs before validation, persistence, or evaluation."""
    if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
        raise ValueError("Sentence IDs must be a list of strings")
    allowed = set(candidates)
    aliases = sentence_id_map(candidates)
    result = []
    for value in values:
        if value in allowed:
            result.append(value)
        elif value in aliases:
            result.append(aliases[value])
        elif allow_indices and value.isdigit() and 0 <= int(value) < len(candidates):
            result.append(candidates[int(value)])
        else:
            raise ValueError(f"Unknown candidate sentence ID: {value}")
    return result


def ranking_for_evaluation(payload, candidates):
    ranked = resolve_sentence_ids(ranked_sentence_ids(payload), candidates)
    if len(ranked) != len(set(ranked)):
        raise ValueError("Duplicate sentence IDs in prediction")
    return ranked


def format_ranking_prompt(query, candidates, corpus):
    sentence_id_map(candidates)
    rows = []
    for index, full_id in enumerate(candidates):
        sentence_id = full_id.rsplit("::", 1)[-1]
        text = str(corpus[full_id].get("text", "")).replace("\n", " ").strip()
        rows.append(f"[{index:03d}] ID={json.dumps(sentence_id, ensure_ascii=False)} TEXT={text}")
    return USER_TEMPLATE.format(query=query, candidate_sentences="\n".join(rows))
