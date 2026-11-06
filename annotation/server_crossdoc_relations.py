"""Grouped relation UI backed by the existing evidence-module stores."""
import json,argparse
from functools import lru_cache
from urllib.parse import urlparse,parse_qs
from pathlib import Path
import server_crossdoc_evidence as old
from crossdoc_addon.build_relation_groups import OUT
from crossdoc_addon.store import CrossdocStore
ASSETS=Path(__file__).resolve().parent/'crossdoc_addon/relation_ui'
@lru_cache(maxsize=3)
def relation_page_requirements(bucket):
 return {str(x['group']):[set(t['required_pair_ids']) for t in x['tasks']] for x in (json.loads(f.read_text()) for f in sorted((OUT/bucket).glob('group_*.json')))}
def completed_page_counts(requirements,completed):
 return {g:sum(bool(ids) and ids<=completed for ids in pages) for g,pages in requirements.items()}
class RelationStore(CrossdocStore):
 def checked(self,group,rid,state):
  if not isinstance(state,dict):raise ValueError('Invalid annotation state')
  normalized=dict(state)
  if 'invalid' in state and type(state['invalid']) is not bool:raise ValueError('Invalid invalid flag')
  if state.get('invalid') is True:
   # Invalid is an exclusion decision, never a relevance-0 judgment.
   clean={**state,'human_confirmed':False,'document_relevance':None,'actor_match':'unknown','scope_match':'unknown','direct_sentence_ids':[],'partial_sentence_ids':[],'non_sentence_ids':[]}
   out=super().checked(group,rid,clean)
   return {**out,'invalid':True,'annotation_status':'invalid','human_confirmed':state.get('human_confirmed') is True,'human_verified':False,'invalid_confirmed':state.get('human_confirmed') is True,'sentence_annotation_complete':False,'non_label_policy':None,'source':'human'}
  if state.get('annotation_schema') in ('evidence_only_v1','evidence_complete_v2'):
   if type(state.get('review_uncertain',False)) is not bool:raise ValueError('Invalid uncertain flag')
   confirmed=state.get('human_confirmed') is True;uncertain=state.get('review_uncertain',False)
   clean={**state,'human_confirmed':False,'actor_match':'unknown','scope_match':'unknown'}
   out=super().checked(group,rid,clean)
   d=out.get('direct_sentence_ids',[]);p=out.get('partial_sentence_ids',[]);n=out.get('non_sentence_ids',[])
   spec=self.registry[str(int(group))][rid]
   if not isinstance(n,list) or any(not isinstance(x,str) or x not in spec['sentence_ids'] for x in n) or len(n)!=len(set(n)) or set(n)&(set(d)|set(p)):raise ValueError('Invalid Non evidence')
   if uncertain and (d or p):raise ValueError('Uncertain judgment cannot retain accepted evidence')
   if uncertain and confirmed and not str(state.get('note','')).strip():raise ValueError('Explain uncertainty in the notes')
   if state.get('annotation_schema')=='evidence_complete_v2':
    complete=confirmed and not uncertain
    out['non_sentence_ids']=sorted(set(spec['sentence_ids'])-set(d)-set(p)) if complete else []
    out['sentence_annotation_complete']=complete
    out['non_label_policy']='complement_of_confirmed_evidence' if complete else None
   grade='unknown' if uncertain or not confirmed else '2' if d else '1' if p else '0'
   return {**out,'document_relevance':grade,'relevance_source':'derived_from_human_evidence' if confirmed and not uncertain else 'unjudged','actor_match':None,'scope_match':None,'actor_scope_human_verified':False,'match_review_required':False,'human_confirmed':confirmed,'human_verified':confirmed and not uncertain,'annotation_status':'needs_review' if uncertain else 'valid' if confirmed else 'draft','invalid':False,'invalid_confirmed':False}
  for key in ('actor_match','scope_match'):
   value=normalized.get(key)
   if value in ('yes','no'):value={'yes':'2','no':'0'}[value]
   elif type(value) is int:value=str(value)
   if value not in ('2','1','0','unknown',None,''):raise ValueError('Invalid actor/scope score')
   normalized[key]=value
  # Reuse legacy structural checks; partial compatibility is stored distinctly.
  legacy={**normalized}
  for key in ('actor_match','scope_match'):
   legacy[key]={'2':'yes','1':'yes','0':'no'}.get(normalized[key],normalized[key])
  out=super().checked(group,rid,legacy)
  for key in ('actor_match','scope_match'):out[key]=normalized[key]
  out['invalid']=False;out['annotation_status']='valid' if out.get('human_confirmed') else 'draft';out['invalid_confirmed']=False
  out['match_scale']='2_full_1_partial_0_no_unknown_v2'
  out['match_review_required']=out.get('document_relevance') in ('1','2') and any(out[k]=='1' for k in ('actor_match','scope_match'))
  ids=out.get('non_sentence_ids',[]);spec=self.registry[str(int(group))][rid]
  if not isinstance(ids,list) or any(not isinstance(s,str) or s not in spec['sentence_ids'] for s in ids) or len(ids)!=len(set(ids)):raise ValueError('Invalid Non sentence IDs')
  if set(ids)&(set(out.get('direct_sentence_ids',[]))|set(out.get('partial_sentence_ids',[]))):raise ValueError('Non overlaps accepted evidence')
  return out
class RelationHandler(old.EvidenceHandler):
 def do_GET(self):
  parsed=urlparse(self.path);path=parsed.path
  if path in ('/crossdoc-assets/quality_grouped.js','/crossdoc-assets/quality_after_grouped.js'):
   self.send_content((ASSETS.parent/path.rsplit('/',1)[-1]).read_bytes(),'text/javascript; charset=utf-8');return
  if path.startswith('/relation-assets/'):
   name=path.rsplit('/',1)[-1]
   if name not in ('app.js','style.css','claude.json'):self.send_error(404);return
   self.send_content((ASSETS/name).read_bytes(),{'app.js':'text/javascript','style.css':'text/css','claude.json':'application/json'}[name]);return
  for bucket,_,_,_ in old.SPECS:
   prefix='/crossdoc-rel-'+bucket+'/'
   if not path.startswith(prefix):continue
   leaf=path[len(prefix):]
   if leaf=='api/progress':
    from server import AnnotationStore
    user=parse_qs(parsed.query).get('annotator',[''])[0]
    try:AnnotationStore._validate_identity(user,'1')
    except ValueError:self.send_error(400);return
    completed=set()
    for p in (old.OUT/bucket/'server_data').glob('*.jsonl'):
     for line in p.read_text().splitlines():
      if not line.strip():continue
      r=json.loads(line)
      if r['annotator']==user and r['state'].get('human_confirmed') is True:completed.add(r['relation_id'])
    groups={g:len(completed&set(ids)) for g,ids in json.loads((OUT/bucket/'progress_map.json').read_text()).items()};self._send_json({'ok':True,'groups':groups,'page_groups':completed_page_counts(relation_page_requirements(bucket),completed)});return
   file=(OUT/bucket/leaf).resolve()
   try:file.relative_to(OUT/bucket)
   except ValueError:self.send_error(403);return
   if not file.is_file():self.send_error(404);return
   if file.suffix=='.json' and leaf.startswith('group_'):
    data=json.loads(file.read_text());data['total_pages']=len(json.loads((OUT/bucket/'index.json').read_text()));self._send_json(data);return
   self.send_content(file.read_bytes(),'text/html; charset=utf-8' if file.suffix=='.html' else 'application/json');return
  return super().do_GET()
 def send_content(self,data,ctype):
  if ctype.startswith('text/html') and (getattr(self,'_area_prefix','') == '/crossdoc-quality/' or urlparse(self.path).path.startswith('/crossdoc-quality/')):
   data=data.replace(b'/crossdoc-assets/ui.js',b'/crossdoc-assets/quality_grouped.js')
  if ctype.startswith('text/html') and urlparse(self.path).path in ('/','/index.html'):
   text=data.decode();entries={};config={}
   for bucket,key,title,_ in old.SPECS:
    entries[key]=json.loads((OUT/bucket/'index.json').read_text());config[key]={'title':title,'desc':'Up to 25 relation pages per group. A page is complete when all its query–article pairs are confirmed.','note':'Only query–article pairs in this category are shown. Submissions retain their storage groups, with 20 storage groups per file. Model suggestions are separate from human confirmations.','tag':'Review by relation','prefix':'/crossdoc-rel-'+bucket+'/'}
   # The outer evidence wrapper injects its old entries before this block;
   # override them immediately before rendering.
   injection='Object.assign(DATA,'+json.dumps(entries,ensure_ascii=False)+');Object.assign(CONFIG,'+json.dumps(config,ensure_ascii=False)+');\n'
   text=text.replace('function choose(area)',injection+'function choose(area)',1)
   # Bypass the intermediate wrapper's later override; manually apply its root processing first.
   data=text.encode()
  return super().send_content(data,ctype)
def make_server(host,port,legacy_root='remaining'):
 server=old.make_server(host,port,legacy_root);server.RequestHandlerClass=RelationHandler
 for bucket,_,_,_ in old.SPECS:
  directory=(old.OUT/bucket/'server_data').resolve();s=server.area_stores[directory];server.area_stores[directory]=RelationStore(directory,s.registry,'annotation')
 return server
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8001);ap.add_argument('--legacy-root',choices=['original','remaining'],default='remaining');ap.add_argument('--public-prefix',default='');a=ap.parse_args();old.old.PUBLIC_PREFIX=a.public_prefix.rstrip('/');s=make_server('127.0.0.1',a.port,a.legacy_root)
 print(f'Relation review http://127.0.0.1:{a.port}',flush=True)
 try:s.serve_forever()
 except KeyboardInterrupt:pass
 finally:s.server_close()
