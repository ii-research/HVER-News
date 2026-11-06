#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

try:
    import fcntl  # Linux/macOS; optional extra process-level lock.
except Exception:  # pragma: no cover
    fcntl = None

ANNOTATOR_RE = re.compile(r"^[A-Za-z0-9._-]{1,60}$")
GROUP_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
MAX_BODY_BYTES = 20 * 1024 * 1024


class AnnotationStore:
    """Append-only JSONL annotation store shared by all annotators."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.Lock()
        if not self.path.exists():
            self.path.touch()

    @staticmethod
    def _validate_identity(annotator: str, group: str) -> tuple[str, str]:
        annotator = str(annotator or "").strip()
        group = str(group or "").strip()
        if not ANNOTATOR_RE.fullmatch(annotator):
            raise ValueError(
                "annotator must match [A-Za-z0-9._-] and be 1-60 characters"
            )
        if not GROUP_RE.fullmatch(group):
            raise ValueError(
                "group must match [A-Za-z0-9._-] and be 1-80 characters"
            )
        return annotator, group

    def _append_records(self, records: list[dict]) -> None:
        if not records:
            return
        lines = "".join(
            json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n"
            for x in records
        )
        with self._thread_lock:
            with self.path.open("a", encoding="utf-8") as f:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(lines)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    if fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def save_one(
        self,
        *,
        annotator: str,
        group: str,
        relation_id: str,
        state: dict,
        protocol: str = "",
        client_updated_at: str = "",
    ) -> dict:
        annotator, group = self._validate_identity(annotator, group)
        relation_id = str(relation_id or "").strip()
        if not relation_id:
            raise ValueError("relation_id is required")
        if not isinstance(state, dict):
            raise ValueError("state must be a JSON object")

        record = {
            "record_type": "annotation_state",
            "annotator": annotator,
            "group": group,
            "relation_id": relation_id,
            "client_updated_at": str(client_updated_at or state.get("_client_updated_at", "")),
            "server_saved_at": datetime.now(timezone.utc).isoformat(),
            "protocol": str(protocol or ""),
            "state": state,
        }
        self._append_records([record])
        return record

    def save_batch(
        self,
        *,
        annotator: str,
        group: str,
        states: dict,
        protocol: str = "",
    ) -> int:
        annotator, group = self._validate_identity(annotator, group)
        if not isinstance(states, dict):
            raise ValueError("states must be a JSON object keyed by relation_id")

        now = datetime.now(timezone.utc).isoformat()
        records = []
        for relation_id, state in states.items():
            if not isinstance(state, dict):
                continue
            rid = str(relation_id or "").strip()
            if not rid:
                continue
            records.append(
                {
                    "record_type": "annotation_state",
                    "annotator": annotator,
                    "group": group,
                    "relation_id": rid,
                    "client_updated_at": str(state.get("_client_updated_at", "")),
                    "server_saved_at": now,
                    "protocol": str(protocol or ""),
                    "state": state,
                }
            )
        self._append_records(records)
        return len(records)

    def load_latest(self, *, annotator: str, group: str) -> dict:
        annotator, group = self._validate_identity(annotator, group)
        latest: dict[str, dict] = {}
        if not self.path.exists():
            return latest

        # Append order is the authoritative version order. If the final line is
        # truncated by an interrupted write, skip that line and keep older data.
        with self._thread_lock:
            with self.path.open("r", encoding="utf-8") as f:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    for raw in f:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            rec = json.loads(raw)
                        except Exception:
                            continue
                        if rec.get("record_type") != "annotation_state":
                            continue
                        if str(rec.get("annotator", "")) != annotator:
                            continue
                        if str(rec.get("group", "")) != group:
                            continue
                        rid = str(rec.get("relation_id", ""))
                        if rid:
                            latest[rid] = rec
                finally:
                    if fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return latest


class AnnotationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, *, pages_dir: Path, store: AnnotationStore):
        super().__init__(server_address, RequestHandlerClass)
        self.pages_dir = pages_dir.resolve()
        self.store = store
        self.quality_review_dir = Path(__file__).resolve().parent / "example_data" / "document_quality"
        self.quality_review_store = None
        self.quality_review_lock = threading.Lock()
        work = Path(__file__).resolve().parent / "example_data"
        self.annotation_areas = {
            "/original/": (work / "pages", work / "server_data/annotations.jsonl"),
            "/remaining/": (work / "document_annotation/pages", work / "document_annotation/server_data"),
            "/quality-review/": (work / "document_quality/pages", work / "document_quality/server_data/quality_annotations.jsonl"),
        }
        self.area_stores = {store.path: store}


class Handler(BaseHTTPRequestHandler):
    server_version = "NEVUAnnotationServer/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} - {fmt % args}")

    def _send_json(self, obj, status=HTTPStatus.OK):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except Exception:
            n = 0
        if n <= 0:
            raise ValueError("empty request body")
        if n > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        raw = self.rfile.read(n)
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("JSON body must be an object")
        return obj

    def _route_annotation_area(self, method):
        """Route each area to its own pages and store without mutating shared server state."""
        if getattr(self, "_area_routed", False):
            return False
        prefix = next((p for p in self.server.annotation_areas if self.path.startswith(p)), None)
        if prefix is None:
            return False
        original_server, original_path = self.server, self.path
        pages_dir, data_path = original_server.annotation_areas[prefix]
        if not (pages_dir / "index.html").is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Generate this area's pages first")
            return True
        data_path = data_path.resolve()
        with original_server.quality_review_lock:
            if data_path not in original_server.area_stores:
                if prefix == "/remaining/":
                    from latest_annotation_store import ShardedAnnotationStore
                    original_server.area_stores[data_path] = ShardedAnnotationStore(data_path)
                elif prefix == "/quality-review/":
                    from quality_annotation_store import QualityAnnotationStore
                    original_server.area_stores[data_path] = QualityAnnotationStore(data_path)
                else:
                    original_server.area_stores[data_path] = AnnotationStore(data_path)
            store = original_server.area_stores[data_path]
        self.server = SimpleNamespace(pages_dir=pages_dir.resolve(), store=store)
        self.path = "/" + original_path[len(prefix):]
        self._area_routed = True
        self._area_prefix = prefix
        try:
            if method == "GET":
                self.do_GET()
            else:
                self.do_POST()
        finally:
            self.server, self.path = original_server, original_path
            self._area_routed = False
            self._area_prefix = ""
        return True

    def do_POST(self):
        if self._route_annotation_area("POST"):
            return
        parsed = urlparse(self.path)
        try:
            body = self._read_json_body()
            if getattr(self, "_area_prefix", "") == "/quality-review/" and parsed.path in {"/api/save", "/api/save-batch"}:
                from quality_dataset_editor import QualityDatasetEdits, RevisionConflict
                states = body.get("states") if parsed.path.endswith("save-batch") else {body.get("relation_id"): body.get("state")}
                try:
                    if not isinstance(states, dict): raise ValueError("Invalid states")
                    QualityDatasetEdits(self.server.pages_dir.parent).validate_submission(body.get("group"), states)
                except RevisionConflict as exc:
                    self._send_json({"ok":False,"error":str(exc)}, HTTPStatus.CONFLICT)
                    return
            if getattr(self.server.store, "manual_only", False) and body.get("manual_save") is not True:
                self._send_json({"ok": False, "error": "This batch uses manual saves，Refresh the page and click Save。"}, HTTPStatus.CONFLICT)
                return
            if parsed.path == "/api/skip" and hasattr(self.server.store, "delete_one"):
                self.server.store.delete_one(annotator=body.get("annotator"), group=body.get("group"), relation_id=body.get("relation_id"))
                self._send_json({"ok": True})
                return
            if parsed.path == "/api/save":
                rec = self.server.store.save_one(
                    annotator=body.get("annotator"),
                    group=body.get("group"),
                    relation_id=body.get("relation_id"),
                    state=body.get("state"),
                    protocol=body.get("protocol", ""),
                    client_updated_at=body.get("client_updated_at", ""),
                )
                self._send_json(
                    {
                        "ok": True,
                        "relation_id": rec["relation_id"],
                        "server_saved_at": rec["server_saved_at"],
                    }
                )
                return

            if parsed.path == "/api/save-batch":
                count = self.server.store.save_batch(
                    annotator=body.get("annotator"),
                    group=body.get("group"),
                    states=body.get("states"),
                    protocol=body.get("protocol", ""),
                )
                self._send_json(
                    {
                        "ok": True,
                        "saved_relations": count,
                        "server_saved_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                return

            self._send_json({"ok": False, "error": "unknown API endpoint"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"server error: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self):
        if self._route_annotation_area("GET"):
            return
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "data_file": str(self.server.store.path),
                    "storage_layout": getattr(self.server.store, "storage_layout", "single_file"),
                }
            )
            return

        if parsed.path == "/api/progress" and hasattr(self.server.store, "progress"):
            qs = parse_qs(parsed.query)
            try:
                groups = self.server.store.progress((qs.get("annotator") or [""])[0])
                self._send_json({"ok": True, "groups": groups})
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception:
                self._send_json({"ok": False, "error": "Cannot read annotation progress"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if parsed.path == "/api/load":
            qs = parse_qs(parsed.query)
            annotator = (qs.get("annotator") or [""])[0]
            group = (qs.get("group") or [""])[0]
            try:
                latest = self.server.store.load_latest(annotator=annotator, group=group)
                self._send_json(
                    {
                        "ok": True,
                        "annotator": annotator,
                        "group": group,
                        "states": {rid: rec.get("state", {}) for rid, rec in latest.items()},
                        "meta": {
                            rid: {
                                "client_updated_at": rec.get("client_updated_at", ""),
                                "server_saved_at": rec.get("server_saved_at", ""),
                            }
                            for rid, rec in latest.items()
                        },
                    }
                )
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        self._serve_static(parsed.path)

    def _serve_static(self, url_path: str):
        rel = unquote(url_path).lstrip("/")
        if not rel:
            rel = "index.html"

        if rel == "values.html":
            candidate = Path(__file__).resolve().parent / "example_data" / "values.html"
        else:
            candidate = (self.server.pages_dir / rel).resolve()
            try:
                candidate.relative_to(self.server.pages_dir)
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN)
                return

        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.exists() or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        ctype, _ = mimetypes.guess_type(str(candidate))
        ctype = ctype or "application/octet-stream"
        data = candidate.read_bytes()
        if getattr(self, "_area_prefix", "") == "/quality-review/" and candidate.name.startswith("group_") and candidate.suffix == ".html":
            from quality_dataset_editor import QualityDatasetEdits
            data = QualityDatasetEdits(self.server.pages_dir.parent).render_page(data.decode("utf-8")).encode("utf-8")
        if candidate.suffix.lower() in {".html", ".htm"}:
            prefix = getattr(self, "_area_prefix", "")
            if prefix in ("/original/", "/remaining/"):
                # Old HTML stays byte-for-byte unchanged on disk. Its absolute
                # API URLs are scoped only in the served response.
                text = data.decode("utf-8")
                for quote in ('"', "'", "`"):
                    text = text.replace(quote + "/api/", quote + prefix + "api/")
                data = text.encode("utf-8")
        if candidate.suffix.lower() in {".html", ".htm"} and candidate.name.startswith("group_"):
            from shared_annotation_auth import enhance_group_auth
            data = enhance_group_auth(data.decode("utf-8")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(data)))
        # Annotation HTML embeds the current sample; avoid stale cached pages.
        if candidate.suffix.lower() in {".html", ".htm"}:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Serve NEVU human-annotation pages and autosave every annotator's "
            "state into one append-only annotations.jsonl file."
        )
    )
    ap.add_argument(
        "--pages-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing index.html/group_XX.html. Default: "
            "<script_dir>/example_data/pages"
        ),
    )
    ap.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help=(
            "Shared append-only JSONL file. Default: "
            "<script_dir>/example_data/server_data/annotations.jsonl"
        ),
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    pages_dir = (
        args.pages_dir.resolve()
        if args.pages_dir
        else (script_dir / "example_data" / "pages").resolve()
    )
    data_file = (
        args.data_file.resolve()
        if args.data_file
        else (script_dir / "example_data" / "server_data" / "annotations.jsonl").resolve()
    )

    if not pages_dir.exists():
        raise FileNotFoundError(
            f"Pages directory does not exist: {pages_dir}\n"
            "Run 03_build_human_annotation_pages.py first."
        )

    store = AnnotationStore(data_file)
    server = AnnotationServer((args.host, args.port), Handler, pages_dir=pages_dir, store=store)

    print("=== NEVU Annotation Server ===")
    print("Pages     :", pages_dir)
    print("Data file :", data_file)
    print("URL       :", f"http://{args.host}:{args.port}/")
    print("All annotators are appended to the SAME JSONL file.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
