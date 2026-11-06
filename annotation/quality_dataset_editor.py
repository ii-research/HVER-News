"""Versioned corrections to the Stage 1 review snapshot, shared by both services."""
from __future__ import annotations
import copy
import fcntl
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

TEXT_FIELDS = ('unit_text', 'scope_title', 'nevu_explanation', 'actor_name', 'value_name', 'direction')

class RevisionConflict(ValueError):
    pass

def embedded_relations(text):
    match = re.search(r'const RELATIONS\s*=\s*', text)
    if not match:
        raise ValueError('Missing embedded relations')
    rows, length = json.JSONDecoder().raw_decode(text[match.end():])
    return rows, match.end(), match.end() + length

def script_json(value):
    return json.dumps(value, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')

def revision(relation):
    return str(relation.get('_dataset_revision', '0'))

def current_score(relation, state):
    return str(state.get('_dataset_revision', '0')) == revision(relation)

class QualityDatasetEdits:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'dataset_edits' / 'revisions.json'

    def read(self):
        if not self.path.exists():
            return {'relations': {}}
        return json.loads(self.path.read_text(encoding='utf-8'))

    @contextmanager
    def lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)

    def apply(self, relation, document=None):
        record = (document if document is not None else self.read())['relations'].get(relation['relation_id'])
        return copy.deepcopy(record['current'] if record else relation)

    def base_relation(self, group, rid):
        if not str(group).isdigit() or not 1 <= int(group) <= 40:
            raise ValueError('Invalid quality review group')
        path = self.root / 'pages' / f'group_{int(group):02d}.html'
        rows, _, _ = embedded_relations(path.read_text(encoding='utf-8'))
        for row in rows:
            if row['original']['relation_id'] == rid:
                return {**row['original'], 'group': str(int(group))}
        raise ValueError('Relation does not belong to this quality review group')

    def save(self, *, group, relation_id, expected_revision, changes, reason):
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
            raise ValueError('Enter a reason for the change（At most 4000 characters）')
        if not isinstance(changes, dict) or set(changes) - set(TEXT_FIELDS) - {'queries','direct_doc_ids','partial_doc_ids'}:
            raise ValueError('Unsupported dataset fields')
        with self.lock():
            document = self.read()
            base = self.base_relation(group, relation_id)
            before = self.apply(base, document)
            if str(expected_revision) != revision(before):
                raise RevisionConflict('Another administrator modified this item，Reopen before saving。')
            after = copy.deepcopy(before)
            for field in TEXT_FIELDS:
                if field in changes:
                    value = changes[field]
                    if not isinstance(value, str) or len(value) > 100000:
                        raise ValueError('Invalid text field: ' + field)
                    if field in ('actor_name','value_name','direction','unit_text') and not value.strip():
                        raise ValueError(field + ' Cannot be empty')
                    if value != before.get(field, ""):
                        after[field] = value
            if 'queries' in changes:
                qs = changes['queries']
                ids = {q['query_id'] for q in before['queries']}
                if not isinstance(qs, dict) or set(qs) != ids:
                    raise ValueError('Query IDs must remain unchanged')
                for q in after['queries']:
                    value = qs[q['query_id']]
                    if not isinstance(value, str) or not value.strip() or len(value) > 100000:
                        raise ValueError('Query Cannot be empty or too long')
                    q['query'] = value
            allowed = {str(s['doc_id']) for s in before['article_sentences']}
            for field in ('direct_doc_ids','partial_doc_ids'):
                if field in changes:
                    values = changes[field]
                    if not isinstance(values, list) or any(not isinstance(x,str) for x in values) or not set(values) <= allowed or len(values) != len(set(values)):
                        raise ValueError('Evidence IDs must refer to unique sentences in this article')
                    if set(values) != set(before[field]):
                        after[field] = values
            if set(after['direct_doc_ids']) & set(after['partial_doc_ids']):
                raise ValueError('A sentence cannot be labeled both Direct and Partial')
            for kind in ('direct','partial'):
                if after[kind+'_doc_ids'] == before[kind+'_doc_ids']: continue
                by_id = {str(x['doc_id']):x for x in after['article_sentences']}
                after[kind+'_evidence'] = [copy.deepcopy(by_id[x]) for x in after[kind+'_doc_ids']]
            if after == before:
                return {'changed': False, 'relation': after}
            after['_dataset_revision'] = uuid.uuid4().hex
            timestamp = datetime.now(timezone.utc).isoformat()
            record = document['relations'].setdefault(relation_id, {'history': []})
            record['history'].append({'edited_at':timestamp,'reason':reason.strip(),'source':'admin_dashboard','before':before,'after':after})
            record['current'] = after
            fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix='.revisions-')
            try:
                with os.fdopen(fd,'w',encoding='utf-8') as out:
                    json.dump(document,out,ensure_ascii=False);out.flush();os.fsync(out.fileno())
                os.replace(temporary,self.path)
            finally:
                if os.path.exists(temporary):os.unlink(temporary)
            return {'changed':True,'relation':after,'edited_at':timestamp}

    def validate_submission(self, group, states):
        document = self.read()
        for rid, state in states.items():
            if not isinstance(state,dict):raise ValueError('Invalid annotation state')
            record = document['relations'].get(rid)
            if record and (str(group) != str(record['current']['group']) or not current_score(record['current'],state)):
                raise RevisionConflict('Reassessment data updated，Refresh the page，Rate the new version and save。Previous ratings retained。')

    def render_page(self, text):
        rows,start,end = embedded_relations(text)
        document = self.read()
        for row in rows:
            row['original'] = self.apply(row['original'], document)
        text = text[:start] + script_json(rows) + text[end:]
        text = text.replace('markSaved("Saved locally; SERVER SAVE FAILED",true)', 'markSaved("SERVER SAVE FAILED: "+e.message,true)')
        patch = (Path(__file__).parent / 'quality_dataset_revision_ui.js').read_text(encoding='utf-8')
        return text.replace('\nbootAuth();', '\n'+patch+'\nbootAuth();')
