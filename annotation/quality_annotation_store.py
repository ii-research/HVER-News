"""Latest quality score per account/relation, with atomic writes and migration backup."""
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from latest_annotation_store import LatestAnnotationStore


def content(state):
    result = dict(state)
    result.pop('_client_updated_at', None)
    result['_dataset_revision'] = str(result.get('_dataset_revision', '0'))
    return result


def timestamp(record):
    value = record.get('client_updated_at') or record.get('state', {}).get('_client_updated_at')
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


class QualityAnnotationStore(LatestAnnotationStore):
    manual_only = False  # Existing quality pages may still autosave drafts.
    storage_layout = 'quality_latest_per_account_relation_v1'

    @staticmethod
    def _key(record):
        return (str(record['annotator']), str(record['relation_id']))

    def _read_records(self):
        rows = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            record = json.loads(line)
            if record.get('record_type') != 'annotation_state' or not isinstance(record.get('state'), dict):
                raise ValueError('Unexpected record in quality annotations; refusing to rewrite')
            self._key(record)
            rows.append(record)
        return rows

    def load_all_latest(self):
        # Atomic replacement means each read sees a complete file snapshot.
        latest = {self._key(r): r for r in self._read_records()}
        return {(r['annotator'], str(r['group']), r['relation_id']): r for r in latest.values()}

    def save_one(self, *, annotator, group, relation_id, state, protocol='', client_updated_at='', source='', admin_reason=''):
        annotator, group = self._validate_identity(annotator, group)
        rid = str(relation_id or '').strip()
        if not rid or not isinstance(state, dict): raise ValueError('relation_id and state are required')
        record = dict(record_type='annotation_state', annotator=annotator, group=group, relation_id=rid,
                      client_updated_at=str(client_updated_at or state.get('_client_updated_at', '')),
                      server_saved_at=datetime.now(timezone.utc).isoformat(), protocol=str(protocol), state=state)
        if source: record['source'] = source
        if admin_reason: record['admin_reason'] = admin_reason
        self._rewrite([record])
        return record

    def _rewrite(self, records, backup=False, remove_key=None):
        if remove_key is not None: raise ValueError('Quality record deletion is not supported')
        with self._write_lock():
            rows = self._read_records()
            latest = {self._key(r): r for r in rows}
            changed = len(latest) != len(rows)
            accepted = 0
            for rec in records:
                key = self._key(rec)
                previous = latest.get(key)
                if previous:
                    if content(previous['state']) == content(rec['state']):
                        rec.clear();rec.update(previous)
                        continue
                    old_content, new_content = content(previous['state']), content(rec['state'])
                    for field in ('_manual_confirmed', '_manual_confirmed_at'):
                        old_content.pop(field, None);new_content.pop(field, None)
                    if old_content == new_content and previous['state'].get('_manual_confirmed') and not rec['state'].get('_manual_confirmed'):
                        rec.clear();rec.update(previous)
                        continue
                    old, new = timestamp(previous), timestamp(rec)
                    if old is not None and new is not None and new < old:
                        raise ValueError('The server has newer ratings，Refresh before editing again；This stale request did not overwrite the data。')
                latest[key] = rec
                changed = True
                accepted += 1
            saved_backup = None
            # Always preserve the original bytes before collapsing historical duplicates.
            if backup or len(rows) != len({self._key(r) for r in rows}):
                saved_backup = self.path.with_name(self.path.name + datetime.now().strftime('.before_latest_%Y%m%d_%H%M%S_%f.bak'))
                shutil.copy2(self.path, saved_backup)
            if changed:
                fd, temporary = tempfile.mkstemp(prefix=self.path.name+'.',dir=self.path.parent)
                try:
                    os.fchmod(fd,self.path.stat().st_mode & 0o777)
                    with os.fdopen(fd,'w',encoding='utf-8') as out:
                        for rec in latest.values():out.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n')
                        out.flush();os.fsync(out.fileno())
                    os.replace(temporary,self.path)
                finally:
                    if os.path.exists(temporary):os.unlink(temporary)
            return {'before':len(rows),'after':len(latest),'accepted':accepted,'backup':str(saved_backup) if saved_backup else None}
