"""Prepare frozen query views and executable configs; no model inference."""

import argparse, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avr_retriever_experiment.data import load_split
from avr_retriever_experiment.sampling import (
    select_experiment_query_ids,
    DEFAULT_QUERY_TYPES,
)

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
a = p.parse_args()
checkpoint = a.checkpoint.resolve()
if not checkpoint.is_dir():
    raise ValueError("Download the pinned official checkpoint first")
bundle = ROOT / "provence_bundle"
data = ROOT.parent / "dataset/document-level"
for split in ["train", "dev"]:
    select_experiment_query_ids(
        load_split(data / split, split),
        bundle / "query_views",
        "one_per_relation_balanced",
        42,
        True,
        DEFAULT_QUERY_TYPES,
        None,
    )
views = bundle / "query_views/_query_views/one_per_relation_balanced_seed42"
subprocess.run(
    [
        sys.executable,
        str(ROOT / "tools/build_provence_bundle.py"),
        "--bundle-root",
        str(bundle),
        "--project-root",
        str(ROOT),
        "--data-root",
        str(data),
        "--query-view-root",
        str(views),
        "--checkpoint",
        str(checkpoint),
    ],
    check=True,
)
for seed in [13, 42, 87]:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/make_provence_configs.py"),
            "--bundle-root",
            str(bundle),
            "--project-root",
            str(ROOT),
            "--checkpoint",
            str(checkpoint),
            "--results-root",
            str(ROOT / "results/document-level"),
            "--seed",
            str(seed),
        ],
        check=True,
    )
