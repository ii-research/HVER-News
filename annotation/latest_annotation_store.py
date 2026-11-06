"""Latest-only storage for the remaining Stage-2 batch; legacy stores stay append-only."""
import json
import os
import tempfile
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from server import AnnotationStore, fcntl


class LatestAnnotationStore(AnnotationStore):
    manual_only = True

    @contextmanager
    def _write_lock(self):
        deadline = time.monotonic() + 5
        if not self._thread_lock.acquire(timeout=5):
            raise TimeoutError("Save in progress，Try again later")
        try:
            with self.path.with_suffix('.lock').open('a') as lock:
                if fcntl is not None:
                    while True:
                        try:
                            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Another process is saving this file，Try again later")
                            time.sleep(0.05)
                try:
                    yield
                finally:
                    if fcntl is not None:fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            self._thread_lock.release()

    def _append_records(self, records):
        self._rewrite(records)

    def compact(self):
        return self._rewrite([], backup=True)

    def _rewrite(self, records, backup=False, remove_key=None):
        # Lock a separate inode so atomic replacement also works across processes.
        with self._write_lock():
            latest = {}
            count = 0
            try:
                existing_text = self.path.read_text(encoding='utf-8')
            except FileNotFoundError:
                existing_text = ''  # A cached store may outlive a removed data file.
            for line in existing_text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)  # Never silently drop malformed history.
                key = (rec['annotator'], str(rec['group']), rec['relation_id'])
                latest[key] = rec
                count += 1
            if remove_key is not None:
                if remove_key not in latest:return {"removed":False}
                del latest[remove_key]
            saved_backup = None
            if backup:
                saved_backup = self.path.with_name(self.path.name + datetime.now().strftime('.before_compaction_%Y%m%d_%H%M%S_%f.bak'))
                shutil.copy2(self.path, saved_backup)
            for rec in records:
                key = (rec['annotator'], str(rec['group']), rec['relation_id'])
                latest[key] = rec
            fd, temporary = tempfile.mkstemp(prefix=self.path.name+'.', dir=self.path.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as out:
                    for rec in latest.values():
                        out.write(json.dumps(rec, ensure_ascii=False, separators=(',', ':'))+'\n')
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):os.unlink(temporary)
            return {'before':count,'after':len(latest),'backup':str(saved_backup) if saved_backup else None}


class ShardedAnnotationStore:
    """Route each group to one 20-group file, with independent shard locks."""
    manual_only = True
    storage_layout = 'document_annotation_groups_of_20_v1'

    def __init__(self, directory):
        import threading
        self.path = directory.resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self._stores = {}
        self._stores_lock = threading.Lock()

    def file_for_group(self, group):
        group = str(group)
        if not group.isdigit() or int(group) < 41:
            raise ValueError('Remaining annotation group must be an integer >= 41')
        start = 41 + ((int(group)-41)//20)*20
        return self.path / f'annotations_group_{start}_{start+19}_plus.jsonl'

    def _store(self, group):
        path = self.file_for_group(group)
        with self._stores_lock:
            if path not in self._stores:
                self._stores[path] = LatestAnnotationStore(path)
            return self._stores[path]

    def save_one(self, **kwargs):
        kwargs['group'] = str(int(kwargs['group']))
        return self._store(kwargs['group']).save_one(**kwargs)

    def save_batch(self, **kwargs):
        kwargs['group'] = str(int(kwargs['group']))
        return self._store(kwargs['group']).save_batch(**kwargs)

    def load_latest(self, *, annotator, group):
        store = self._store(group)
        return store.load_latest(annotator=annotator, group=str(int(group)))

    def progress(self, annotator):
        AnnotationStore._validate_identity(annotator, '41')
        groups = {}
        # Each file is replaced atomically: readers see a complete old or new snapshot.
        for path in self.path.glob('annotations_group_*_*_plus.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                rec = json.loads(line)
                if rec.get('annotator') != annotator:continue
                if rec.get('state', {}).get('human_confirmed') is not True:continue
                groups.setdefault(str(rec['group']), set()).add(rec['relation_id'])
        return {group:len(ids) for group,ids in groups.items()}

    def delete_one(self, *, annotator, group, relation_id):
        annotator, group = AnnotationStore._validate_identity(annotator, str(group))
        path = self.file_for_group(group)
        if not relation_id:raise ValueError('relation_id is required')
        if not path.exists():return
        self._store(group)._rewrite([], remove_key=(annotator,str(int(group)),str(relation_id)))
