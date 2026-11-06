import json
from pathlib import Path
from latest_annotation_store import ShardedAnnotationStore
class CrossdocStore(ShardedAnnotationStore):
 storage_layout='crossdoc_pages_of_20_v1'
 def __init__(self,directory,registry,mode):
  super().__init__(directory);self.registry=registry;self.mode=mode
 def file_for_group(self,group):
  if not str(group).isdigit() or str(int(group)) not in self.registry:raise ValueError('Unknown cross-document page')
  start=1+((int(group)-1)//20)*20
  return self.path/f'annotations_group_{start}_{start+19}.jsonl'
 def checked(self,group,rid,state):
  group=str(int(group));self.file_for_group(group)
  if rid not in self.registry[group]:raise ValueError('Pair does not belong to this page')
  if not isinstance(state,dict):raise ValueError('Invalid annotation state')
  spec=self.registry[group][rid]
  if state.get('query_id')!=spec['query_id'] or state.get('doc_id')!=spec['doc_id']:raise ValueError('Query/document mismatch')
  if state.get('module')!=self.mode:raise ValueError('Wrong module')
  complete=state.get('human_confirmed') is True
  if self.mode=='quality':
   if state.get('quality_schema') in ('quality_query_save_v3','quality_query_save_v4'):
    if type(state.get('invalid',False)) is not bool:raise ValueError('Invalid exclusion flag')
    if state.get('invalid') is True:
     return {**state,'pair_id':rid,'source':'human','module':'quality','annotation_status':'invalid','human_verified':False,'invalid_confirmed':complete}
    for k in ('actor_match_quality','scope_match_quality'):
     if state.get(k) not in ('0','1','2','unknown',None,'') or (complete and state.get(k) not in ('0','1','2','unknown')):raise ValueError('Complete actor and event ratings')
   if state.get('quality_schema')=='quality_eight_dimensions_v2':
    for k in ('document_relevance_quality','actor_match_quality','scope_match_quality'):
     if state.get(k) not in ('0','1','2','unknown',None,'') or (complete and state.get(k) not in ('0','1','2','unknown')):raise ValueError('Incomplete document/actor/scope quality scores')
   applicable=['evidence_set_sufficiency','query_evidence_consistency']+([] if state.get('quality_schema')=='quality_query_save_v4' else ['query_quality'])+(['direct_evidence_validity'] if spec['has_direct'] else [])+(['partial_evidence_validity'] if spec['has_partial'] else [])
   if complete and any(state.get(k) not in ('0','1','2') for k in applicable):raise ValueError('Incomplete Stage1 scores')
   for key,present in [('direct_evidence_validity',spec['has_direct']),('partial_evidence_validity',spec['has_partial'])]:
    if not present and state.get(key) not in (None,''):raise ValueError('Absent evidence must be N/A')
  else:
   d=state.get('direct_sentence_ids',[]);p=state.get('partial_sentence_ids',[])
   for a in [d,p]:
    if not isinstance(a,list) or any(not isinstance(s,str) or s not in spec['sentence_ids'] for s in a) or len(a)!=len(set(a)):raise ValueError('Invalid sentence ID')
   if set(d)&set(p):raise ValueError('Direct/Partial overlap')
   if complete:
    g=state.get('document_relevance')
    if g not in ('0','1','2','unknown'):raise ValueError('Choose document relevance')
    if state.get('actor_match') not in ('yes','no','unknown') or state.get('scope_match') not in ('yes','no','unknown'):raise ValueError('Confirm actor and scope')
    if g in ('1','2') and (state['actor_match']!='yes' or state['scope_match']!='yes'):raise ValueError('Positive requires supported actor and scope')
    if g=='2' and not d:raise ValueError('Grade 2 requires Direct')
    if g=='1' and (d or not p):raise ValueError('Grade 1 requires Partial and no Direct')
    if g in ('0','unknown') and (d or p):raise ValueError('0/unknown cannot retain accepted evidence')
    if g=='unknown' and not str(state.get('note','')).strip():raise ValueError('Explain unknown judgment')
  return {**state,'pair_id':rid,'source':'human','module':self.mode,'human_verified':complete}
 def save_one(self,**kw):
  if kw.get('protocol')!='crossdoc_'+self.mode+'_v1':raise ValueError('Wrong protocol')
  kw['state']=self.checked(kw['group'],kw['relation_id'],kw['state']);return super().save_one(**kw)
 def save_batch(self,**kw):
  if kw.get('protocol')!='crossdoc_'+self.mode+'_v1':raise ValueError('Wrong protocol')
  if not isinstance(kw.get('states'),dict):raise ValueError('Invalid states')
  kw['states']={rid:self.checked(kw['group'],rid,s) for rid,s in kw['states'].items()};return super().save_batch(**kw)
 def load_latest(self,*,annotator,group):
  records=super().load_latest(annotator=annotator,group=group)
  active=self.registry.get(str(int(group)),{})
  return {pid:r for pid,r in records.items() if pid in active}
 def progress(self,annotator):
  from server import AnnotationStore
  AnnotationStore._validate_identity(annotator,'1');groups={}
  for p in self.path.glob('annotations_group_*.jsonl'):
   for line in p.read_text().splitlines():
    r=json.loads(line)
    if r['annotator']==annotator and r['relation_id'] in self.registry.get(str(r['group']),{}) and r['state'].get('human_confirmed') is True:groups.setdefault(r['group'],set()).add(r['relation_id'])
  return {g:len(v) for g,v in groups.items()}
