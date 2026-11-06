import json
from collections import defaultdict
from pathlib import Path
from crossdoc_addon.build_evidence_modules import OUT as SOURCE,BASE
OUT=BASE/'example_data/cross_document_relations'
def main():
 byrel=defaultdict(list);docs={};stats={}
 for bucket in ('direct','partial','empty'):
  for p in sorted((SOURCE/bucket/'pages').glob('group_*.json')):
   x=json.loads(p.read_text());docs.update(x['articles'])
   for r in x['pairs']:byrel[r['relation_id']].append({**r,'storage_bucket':bucket,'storage_group':x['group']})
 template=(BASE/'crossdoc_addon/relation_ui/index.html').read_text()
 for bucket in ('direct','partial','empty'):
  dest=OUT/bucket;dest.mkdir(parents=True,exist_ok=True);index=[];mapping={}
  tasks=[]
  for rid,rs in sorted(byrel.items()):
   targets=[r for r in rs if r['storage_bucket']==bucket]
   if not targets:continue
   didset={r['doc_id'] for r in targets}
   tasks.append({'bucket':bucket,'relation_id':rid,'pairs':targets,'articles':{d:docs[d] for d in sorted(didset)},'required_pair_ids':[r['pair_id'] for r in targets]})
  for offset in range(0,len(tasks),25):
   n=offset//25+1;batch=tasks[offset:offset+25]
   payload={'bucket':bucket,'group':n,'tasks':batch,'total_relations':len(tasks),'total_pages':(len(tasks)+24)//25}
   (dest/f'group_{n:02d}.json').write_text(json.dumps(payload,ensure_ascii=False));(dest/f'group_{n:02d}.html').write_text(template)
   ids=[r for task in batch for r in task['required_pair_ids']]
   index.append({'group':n,'relations':len(batch),'queries':len(ids),'ready':0});mapping[str(n)]=ids
  (dest/'index.json').write_text(json.dumps(index));(dest/'progress_map.json').write_text(json.dumps(mapping));stats[bucket]={'groups':len(index),'relations':len(tasks),'required_pairs':sum(r['queries'] for r in index)}
 (OUT/'manifest.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats))
if __name__=='__main__':main()
