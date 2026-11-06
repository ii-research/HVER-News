"""Partition frozen annotation candidates by evidence presence; retain the original review inputs."""
import json, hashlib
from pathlib import Path
from collections import Counter
BASE=Path(__file__).resolve().parents[1]
OLD=BASE/'example_data/cross_document_review/annotation'
OUT=BASE/'example_data/cross_document_evidence'
TITLES={'direct':'Cross-document annotation · Direct Nonempty','partial':'Cross-document annotation · Only Partial','empty':'Cross-document annotation · No evidence'}
def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False))
def main():
 buckets={k:[] for k in TITLES};docs={};mapping={};seen=set();fingerprints={}
 for p in sorted((OLD/'pages').glob('group_*.json')):
  x=json.loads(p.read_text());fingerprints[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();docs.update(x['articles'])
  for r in x['pairs']:
   rid=r['pair_id'];assert rid not in seen;seen.add(rid)
   k='direct' if r['direct_sentence_ids'] else 'partial' if r['partial_sentence_ids'] else 'empty'
   buckets[k].append(r);mapping[rid]={'old_group':x['group'],'bucket':k}
 manifest=OUT/'manifest.json'
 if manifest.exists():assert json.loads(manifest.read_text())['source_sha256']==fingerprints
 template=(OLD/'pages/group_01.html').read_text().replace('/crossdoc-assets/ui.js','/crossdoc-evidence-assets/ui.js').replace('Annotation dashboard','Annotation dashboard')
 stats={}
 for k,rs in buckets.items():
  dest=OUT/k;reg={};index=[]
  for start in range(0,len(rs),25):
   group=start//25+1;part=rs[start:start+25];articles={r['doc_id']:docs[r['doc_id']] for r in part}
   put(dest/f'pages/group_{group:02d}.json',{'module':'annotation','view_id':'evidence_v2_'+k,'title':TITLES[k],'group':group,'protocol':'crossdoc_annotation_v1','pairs':part,'articles':articles})
   (dest/f'pages/group_{group:02d}.html').write_text(template)
   reg[str(group)]={r['pair_id']:{'query_id':r['query_id'],'doc_id':r['doc_id'],'sentence_ids':[s['sentence_id'] for s in articles[r['doc_id']]['sentences']],'has_direct':bool(r['direct_sentence_ids']),'has_partial':bool(r['partial_sentence_ids'])} for r in part}
   index.append({'group':group,'relations':len(part),'queries':len(part),'ready':0})
   for r in part:mapping[r['pair_id']]['new_group']=group
  put(dest/'registry.json',reg);put(dest/'index.json',index)
  (dest/'pages/index.html').write_text('<meta http-equiv="refresh" content="0;url=/?area=crossdoc_'+('annotation' if k=='direct' else k)+'">')
  stats[k]={'pairs':len(rs),'groups':len(index),'grades':dict(Counter(str(r['document_relevance']) for r in rs))}
 assert sum(x['pairs'] for x in stats.values())==len(seen)==26104
 put(OUT/'mapping.json',mapping);put(manifest,{'source_sha256':fingerprints,'classification':'Frozen Gemini evidence: Direct nonempty; otherwise Partial nonempty; otherwise empty. Human edits do not move tasks.','modules':stats})
 print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
