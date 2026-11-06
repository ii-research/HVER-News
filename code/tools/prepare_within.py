"""Compatibility entry point; prefer prepare_document_level.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("prepare_document_level.py")), run_name="__main__")
