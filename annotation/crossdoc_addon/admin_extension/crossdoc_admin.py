"""Authenticated, read-only cross-document administration extension."""
import json
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse,parse_qs
MODULES={'crossdoc_quality_after':('Post-annotation cross-document quality assessment','quality_after'),'crossdoc_quality':('Pre-annotation cross-document quality assessment','quality'),'crossdoc_direct':('Cross-document annotation · Direct Nonempty','direct'),'crossdoc_partial':('Cross-document annotation · Only Partial','partial'),'crossdoc_empty':('Cross-document annotation · No evidence','empty')}
QUALITY_DIMS={
 'crossdoc_quality':['actor_match_quality','scope_match_quality','direct_evidence_validity','partial_evidence_validity','evidence_set_sufficiency','query_evidence_consistency'],
 'crossdoc_quality_after':['actor_match_quality','scope_match_quality','direct_evidence_validity','partial_evidence_validity','evidence_set_sufficiency','query_evidence_consistency'],
}
@lru_cache(maxsize=5)
def catalog(root,module):
 root=Path(root);bucket=MODULES[module][1];quality=bucket in ('quality','quality_after')
 folder=root/'cross_document_review'/bucket/'pages' if quality else root/'cross_document_relations'/bucket
 pairs={};pages=defaultdict(lambda:defaultdict(set));files={}
 for f in sorted(folder.glob('group_*.json')):
  x=json.loads(f.read_text());g=str(x['group']);rows=x['pairs'] if quality else [p for t in x['tasks'] for p in t['pairs']]
  for p in rows:
   pid=p['pair_id'];assert pid not in pairs
   pairs[pid]={**p,'ui_group':g};files[pid]=f;pages[g][p['relation_id']].add(pid)
 store=root/'cross_document_review'/bucket/'server_data' if quality else root/'cross_document_evidence'/bucket/'server_data'
 return pairs,dict(pages),files,store

def snapshot(root,module):
 pairs,pages,files,store=catalog(str(root),module);latest={};warnings=[]
 excluded_file=Path(root)/'crossdoc_excluded_pairs.json';excluded=set(json.loads(excluded_file.read_text()).get('pair_ids',[])) if excluded_file.exists() else set()
 if module=='crossdoc_quality':
  module_excluded=Path(root)/'cross_document_review/quality/excluded_empty_pairs.json'
  if module_excluded.exists():excluded.update(json.loads(module_excluded.read_text()).get('pair_ids',[]))
 for f in sorted(store.glob('*.jsonl')):
  for i,line in enumerate(f.read_text().splitlines(),1):
   if not line.strip():continue
   try:
    r=json.loads(line);k=(r['annotator'],r['relation_id']);assert isinstance(r['state'],dict)
    if k[1] in excluded:continue
    if k[1] not in pairs:warnings.append(f'{f.name}:{i}: unknown pair');continue
    p=pairs[k[1]];s=r['state']
    if s.get('query_id')!=p['query_id'] or s.get('doc_id')!=p['doc_id']:warnings.append(f'{f.name}:{i}: identity mismatch');continue
    prev=latest.get(k)
    if not prev or r.get('server_saved_at','')>=prev.get('server_saved_at',''):latest[k]=r
   except (KeyError,ValueError,AssertionError,TypeError):warnings.append(f'{f.name}:{i}: malformed record')
 return pairs,pages,files,latest,warnings

def state_status(s):
 if not s:return 'unseen'
 if s.get('human_confirmed') is not True:return 'draft'
 if s.get('invalid') is True:return 'invalid'
 if s.get('annotation_status')=='needs_review' or s.get('review_uncertain') is True:return 'needs_review'
 if s.get('module')=='annotation' and s.get('document_relevance') in (None,'unknown'):return 'needs_review'
 return 'confirmed'

def evidence_changes(pair,state):
 od=set(pair.get('direct_sentence_ids') or []);op=set(pair.get('partial_sentence_ids') or [])
 nd=set(state.get('direct_sentence_ids') or []);np=set(state.get('partial_sentence_ids') or [])
 return {'original_direct':len(od),'original_partial':len(op),'human_direct':len(nd),'human_partial':len(np),'added':len((nd|np)-(od|op)),'removed':len((od|op)-(nd|np)),'direct_to_partial':len(od&np),'partial_to_direct':len(op&nd),'changed':int(od!=nd or op!=np)}

def change_totals(pairs,records):
 totals=Counter();eligible=0
 for pid,r in records.items():
  state=r['state']
  if state_status(state)!='confirmed':continue
  eligible+=1;totals.update(evidence_changes(pairs[pid],state))
 return {'eligible_pairs':eligible,'excluded_pairs':len(records)-eligible,**dict(totals)}

def quality_scores(module,records):
 """Compute 02-compatible score statistics from confirmed, valid records only."""
 confirmed=[r['state'] for r in records.values() if state_status(r['state'])=='confirmed']
 out={}
 for key in QUALITY_DIMS.get(module,[]):
  values=[int(s[key]) for s in confirmed if str(s.get(key)) in ('0','1','2')]
  out[key]={
   'n':len(values),
   'mean':sum(values)/len(values) if values else None,
   'strict_2_rate':sum(v==2 for v in values)/len(values) if values else None,
   'soft_ge1_rate':sum(v>=1 for v in values)/len(values) if values else None,
   'unknown':sum(str(s.get(key))=='unknown' for s in confirmed),
  }
 return out

def completed_group_records(pairs,pages,records):
 """Return records from fully saved annotator-group assignments only."""
 completed=[];included={}
 done={pid for pid,r in records.items() if r['state'].get('human_confirmed') is True}
 touched_groups={pairs[pid]['ui_group'] for pid in records}
 for group in sorted(touched_groups,key=int):
  group_ids=set().union(*pages[group].values()) if pages.get(group) else set()
  if group_ids and group_ids<=done:
   completed.append(group)
   for pid in group_ids:included[pid]=records[pid]
 return completed,included

def pooled_quality_rates(scores):
 total=sum(x['n'] for x in scores.values())
 strict=sum(x['strict_2_rate']*x['n'] for x in scores.values() if x['n'])
 soft=sum(x['soft_ge1_rate']*x['n'] for x in scores.values() if x['n'])
 return {'score_count':total,'strict_2_rate':strict/total if total else None,'soft_ge1_rate':soft/total if total else None}

def relation_timing(items):
 times=[(i.get('record') or {}).get('state',{}).get('timing_v1',{}) for i in items]
 starts=[t['page_started_at'] for t in times if t.get('page_started_at')]
 complete=all((i.get('record') or {}).get('state',{}).get('human_confirmed') is True for i in items)
 ends=[i['record'].get('server_saved_at','') for i in items if i.get('record')]
 return {'page_started_at':min(starts) if starts else None,'page_finished_at':max(ends) if complete and ends else None,'complete':complete,'timing_pairs':sum(bool(t.get('query_started_at')) for t in times),'total_pairs':len(items),'historical_partial':any(t.get('historical_record') for t in times)}

def group_summary(pairs,pages,latest):
 rows=[]
 by_user=defaultdict(dict)
 for (who,pid),record in latest.items():by_user[who][pid]=record
 for g,rels in sorted(pages.items(),key=lambda x:int(x[0])):
  ordered=list(rels.values());group_ids=set().union(*ordered) if ordered else set()
  for who,records in sorted(by_user.items()):
   touched=group_ids & records.keys()
   if not touched:continue
   done={pid for pid in touched if records[pid]['state'].get('human_confirmed') is True}
   completed=[bool(ids) and ids<=done for ids in ordered]
   first_pending=next((i for i,ok in enumerate(completed,1) if not ok),None)
   rows.append({'group':g,'annotator':who,'pages':len(ordered),'pages_done':sum(completed),'remaining':len(ordered)-sum(completed),'first_pending':first_pending,'last_touched':max(i for i,ids in enumerate(ordered,1) if ids&touched),'saved_pairs':len(done),'total_pairs':len(group_ids),'draft_pairs':len(touched-done),'last_saved':max(records[pid].get('server_saved_at','') for pid in touched)})
 return rows

def overview(root,module,user='',group='',page=1):
 pairs,pages,files,latest,warnings=snapshot(root,module);users=[]
 for who in sorted({k[0] for k in latest}):
  rs={pid:r for (u,pid),r in latest.items() if u==who};counts=Counter(state_status(r['state']) for r in rs.values());done={pid for pid,r in rs.items() if r['state'].get('human_confirmed') is True}
  active_groups={pairs[pid]['ui_group'] for pid in rs}
  active_pair_total=sum(len(ids) for g,rels in pages.items() if g in active_groups for ids in rels.values())
  completed_groups,included=completed_group_records(pairs,pages,rs)
  scores=quality_scores(module,included) if module in QUALITY_DIMS else {}
  users.append({'active_groups':sorted(active_groups,key=int),'completed_groups':completed_groups,'completed_group_count':len(completed_groups),'included_quality_pairs':sum(state_status(r['state'])=='confirmed' for r in included.values()),'active_page_total':sum(len(rels) for g,rels in pages.items() if g in active_groups),'active_pair_total':active_pair_total,'annotator':who,'touched':len(rs),'saved':len(done),'completion_rate':len(done)/active_pair_total if active_pair_total else None,'invalid':counts['invalid'],'needs_review':counts['needs_review'],'draft':counts['draft'],'pages_done':sum(bool(ids) and ids<=done for rels in pages.values() for ids in rels.values()),'evidence_changes':change_totals(pairs,rs) if module not in ('crossdoc_quality','crossdoc_quality_after') else None,'scores':scores,'quality_rates':pooled_quality_rates(scores),'last_saved':max((r.get('server_saved_at','') for r in rs.values()),default='')})
 overall=Counter()
 for u in users:
  if u['evidence_changes']:overall.update(u['evidence_changes'])
 out={'overall_evidence_changes':dict(overall),'title':MODULES[module][0],'pair_total':len(pairs),'page_total':sum(len(v) for v in pages.values()),'group_total':len(pages),'users':users,'warnings':warnings,'groups':[],'rows':[],'page':page,'selected_user':user}
 if module in QUALITY_DIMS:
  completed_records={}
  for who in {k[0] for k in latest}:
   rs={pid:r for (u,pid),r in latest.items() if u==who}
   _groups,included=completed_group_records(pairs,pages,rs)
   completed_records.update({(who,pid):r for pid,r in included.items()})
  out['scores']=quality_scores(module,completed_records)
  out['quality_summary']={'completed_group_assignments':sum(u['completed_group_count'] for u in users),'included_quality_pairs':sum(u['included_quality_pairs'] for u in users),**pooled_quality_rates(out['scores'])}
 else:out['scores']={};out['quality_summary']={}
 out['group_summary']=group_summary(pairs,pages,latest)
 out['module_totals']={'groups':out['group_total'],'pages':out['page_total'],'pairs':out['pair_total']}
 active_groups={pairs[pid]['ui_group'] for (who,pid) in latest if not user or who==user}
 out.update(group_total=len(active_groups),page_total=sum(len(rels) for g,rels in pages.items() if g in active_groups),pair_total=sum(len(ids) for g,rels in pages.items() if g in active_groups for ids in rels.values()),progress_scope='groups_with_server_saved_records')
 if user:
  rs={pid:r for (u,pid),r in latest.items() if u==user};done={pid for pid,r in rs.items() if r['state'].get('human_confirmed') is True}
  out['groups']=[{'group':g,'pages':len(rels),'pages_done':sum(bool(ids) and ids<=done for ids in rels.values()),'pairs':sum(len(ids) for ids in rels.values()),'saved':sum(len(ids&done) for ids in rels.values())} for g,rels in pages.items() if g in active_groups]
  if module=='crossdoc_direct':
   relations=[]
   for g,rels in pages.items():
    if g not in active_groups:continue
    for index,(rid,ids) in enumerate(rels.items(),1):
     ps=[pairs[pid] for pid in sorted(ids)];ref=ps[0].get('reference',{});records={pid:rs[pid] for pid in ids if pid in rs}
     timing=relation_timing([{'record':records.get(pid)} for pid in ids]);changes=change_totals(pairs,records)
     relations.append({'group':g,'position':index,'relation_id':rid,'pair_id':ps[0]['pair_id'],'actor':ref.get('actor_name',''),'value':ref.get('value_name',''),'direction':ref.get('direction',''),'query_types':sorted({p.get('query_type','') for p in ps}),'documents':len({p['doc_id'] for p in ps}),'pairs':len(ids),'saved':len(ids&done),'touched':len(records),'invalid':sum(state_status(r['state'])=='invalid' for r in records.values()),'needs_review':sum(state_status(r['state'])=='needs_review' for r in records.values()),'status':'complete' if ids<=done else 'in_progress' if records else 'unseen','changed':changes.get('changed',0),'started_at':timing['page_started_at'],'finished_at':timing['page_finished_at'],'last_saved':max((r.get('server_saved_at','') for r in records.values()),default='')})
   out['relations']=relations
   for g in out['groups']:
    rows=[r for r in relations if r['group']==g['group']]
    g['touched']=sum(r['touched']>0 for r in rows);g['last_saved']=max((r['last_saved'] for r in rows),default='');g['changed']=sum(r['changed'] for r in rows)
  selected=[p for p in pairs.values() if p['ui_group'] in active_groups and (not group or p['ui_group']==group)];out['row_total']=len(selected)
  for p in selected[(page-1)*50:page*50]:
   r=rs.get(p['pair_id'],{});s=r.get('state',{});out['rows'].append({'pair_id':p['pair_id'],'group':p['ui_group'],'relation_id':p['relation_id'],'query_type':p.get('query_type'),'query':p['query'],'doc_id':p['doc_id'],'status':state_status(s),'direct':len(s.get('direct_sentence_ids') or []),'partial':len(s.get('partial_sentence_ids') or []),'relevance':s.get('document_relevance'),'saved_at':r.get('server_saved_at','')})
  valid=[r['state'] for r in rs.values() if state_status(r['state'])=='confirmed'];out['labels']=dict(Counter(str(s.get('document_relevance')) for s in valid)) if module not in ('crossdoc_quality','crossdoc_quality_after') else {}
  out['evidence_changes']=change_totals(pairs,rs) if module not in ('crossdoc_quality','crossdoc_quality_after') else None
  if module in QUALITY_DIMS:
   completed_groups,included=completed_group_records(pairs,pages,rs)
   out['scores']=quality_scores(module,included)
   out['quality_summary']={'completed_group_assignments':len(completed_groups),'included_quality_pairs':sum(state_status(r['state'])=='confirmed' for r in included.values()),**pooled_quality_rates(out['scores'])}
  else:out['scores']={};out['quality_summary']={}
 return out

def install(legacy):
 original_setup=legacy.setup_modules
 def setup(server,root):original_setup(server,root);server.crossdoc_root=Path(root)
 legacy.setup_modules=setup
 class Handler(legacy.Handler):
  def do_GET(self):
   parsed=urlparse(self.path)
   if parsed.path.startswith('/api/admin/crossdoc/'):
    if not self._require_admin():return
    try:
     q=parse_qs(parsed.query);module=q.get('module',[''])[0]
     if module not in MODULES:raise ValueError('Unknown crossdoc module')
     root=self.server.crossdoc_root;user=q.get('annotator',[''])[0]
     if parsed.path.endswith('/overview'):
      self._send_json({'ok':True,'data':overview(root,module,user,q.get('group',[''])[0],max(1,int(q.get('page',['1'])[0])))});return
     pairs,pages,files,latest,warnings=snapshot(root,module)
     if parsed.path.endswith('/export'):
      self._send_json({'ok':True,'data':{'module':module,'exported_at':legacy.now_iso(),'warnings':warnings,'records':[{'ui_group':pairs[pid]['ui_group'],**r} for (u,pid),r in latest.items() if not user or u==user]}});return
     if parsed.path.endswith('/detail'):
      pid=q.get('pair_id',[''])[0];p=pairs[pid];x=json.loads(files[pid].read_text());article=x['articles'][p['doc_id']] if module in ('crossdoc_quality','crossdoc_quality_after') else next(t['articles'][p['doc_id']] for t in x['tasks'] if any(z['pair_id']==pid for z in t['pairs']))
      items=[];loaded={str(files[pid]):x}
      for other in pairs.values():
       if other['relation_id']!=p['relation_id']:continue
       file=files[other['pair_id']];key=str(file)
       if key not in loaded:loaded[key]=json.loads(file.read_text())
       data=loaded[key];doc=data['articles'][other['doc_id']] if module in ('crossdoc_quality','crossdoc_quality_after') else next(t['articles'][other['doc_id']] for t in data['tasks'] if any(z['pair_id']==other['pair_id'] for z in t['pairs']))
       record=latest.get((user,other['pair_id']))
       items.append({'pair':other,'article':doc,'record':record,'status':state_status((record or {}).get('state',{}))})
      self._send_json({'ok':True,'data':{'pair':p,'article':article,'record':latest.get((user,pid)),'annotator':user,'module':module,'items':items,'timing':relation_timing(items),'evidence_changes':change_totals(pairs,{i['pair']['pair_id']:i['record'] for i in items if i.get('record')}) if module not in ('crossdoc_quality','crossdoc_quality_after') else None}});return
     raise ValueError('Unknown endpoint')
    except (ValueError,KeyError) as e:self._send_json({'ok':False,'error':str(e)},400)
    return
   return super().do_GET()
  def _serve_static(self,path):
   if path not in ('/','/index.html','/admin_dashboard.html'):return super()._serve_static(path)
   html=self.server.dashboard_file.read_text();js=(Path(__file__).parent/'crossdoc_admin.js').read_text()+'\n'+(Path(__file__).parent/'detail_ui.js').read_text()+'\n'+(Path(__file__).parent/'direct_admin_ui.js').read_text();boot='(async()=>{if(TOKEN)'
   if boot not in html:raise RuntimeError('Admin bootstrap marker missing')
   html=html.replace(boot,js+'\n'+boot,1);data=html.encode()
   self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
 legacy.Handler=Handler
