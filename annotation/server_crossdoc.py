"""Add two cross-document modules without editing legacy server/pages/storage."""
import argparse,json,mimetypes
from pathlib import Path
from urllib.parse import urlparse
from server import AnnotationServer,Handler,AnnotationStore
from latest_annotation_store import ShardedAnnotationStore
from crossdoc_addon.store import CrossdocStore
BASE=Path(__file__).resolve().parent;WORK=BASE/'example_data';ADDON=BASE/'crossdoc_addon';DATA=WORK/'cross_document_review';PUBLIC_PREFIX=''
class CrossdocHandler(Handler):
 def _serve_static(self,url_path):
  # Inject two dashboard entries into the response only; old HTML remains unchanged.
  if url_path in ('/','/index.html') and getattr(self,'_area_prefix','') not in ('/crossdoc-quality/','/crossdoc-annotation/','/crossdoc-quality-after/'):
   page=(BASE/'index.html').read_text();data={};config={}
   for mode,title,desc in [('quality','Pre-annotation cross-document quality assessment','Assess the original sample of query–article pairs using Stage 1 only.'),('quality_after','Post-annotation cross-document quality assessment','Assess the revised evidence for the same sample as Module 04. Ratings are saved separately.'),('annotation','Cross-document annotation','Review all test Full candidate article，Annotate document relevance and sentence evidence。')]:
    key='crossdoc_'+mode;data[key]=json.loads((DATA/mode/'index.json').read_text());config[key]={'title':title,'desc':desc,'note':'Separate storage，Each 20  groups per file。'+('Ratings only，Pre-annotations are unchanged。' if mode.startswith('quality') else 'Model suggestions are hidden by default，Suggestions can be revealed after human judgment；Confirmation and saving are required。'),'tag':'Only Stage1' if mode.startswith('quality') else 'Cross-document human annotation','prefix':'/crossdoc-'+mode.replace('_','-')+'/'}
   injection='Object.assign(DATA,'+json.dumps(data,ensure_ascii=False)+');Object.assign(CONFIG,'+json.dumps(config,ensure_ascii=False)+');\n'
   page=page.replace('let remainingProgress=',injection+'let remainingProgress=',1).replace('grid-template-columns:repeat(3,1fr)','grid-template-columns:repeat(auto-fit,minmax(210px,1fr))').replace('One account for all three task types','One account for all five task types').replace('Three task types · Save independently','Five task types · Save independently')
   page=page.replace('${fmt(qs)} items query', "${fmt(qs)} ${key.startsWith('crossdoc_')?'pairs query–Article':'items query'}")
   page=page.replace('</body>','<script src="/crossdoc-assets/dashboard.js"></script></body>')
   self.send_content(page.encode(),'text/html; charset=utf-8');return
  if getattr(self,'_area_prefix','') in ('/crossdoc-quality/','/crossdoc-annotation/','/crossdoc-quality-after/'):
   path=(self.server.pages_dir/url_path.lstrip('/')).resolve()
   try:path.relative_to(self.server.pages_dir)
   except ValueError:self.send_error(403);return
   if path.is_dir():path=path/'index.html'
   if not path.is_file():self.send_error(404);return
   self.send_content(path.read_bytes(),('text/html; charset=utf-8' if path.suffix=='.html' else 'application/json; charset=utf-8'));return
  return super()._serve_static(url_path)
 def send_content(self,data,ctype):
  if ctype.startswith('text/html') and b'nevu_annotation_session_user_v2' in data and b'nevuAuthStorageFallbackInstalled' not in data:
   helper=(BASE/'auth_storage_fallback.js').read_bytes()
   data=data.replace(b'<script>',b'<script>\n'+helper,1)
   data=data.replace(b"el('accountMessage').textContent='';", "el('accountMessage').textContent=window.nevuAuthTemporary?.()?'Browser storage is full：Fallback sign-in is available in this tab only，Server annotations are unaffected；New accounts may need recreating after closing this tab。':'';".encode())
  if ctype.startswith('text/html') or 'javascript' in ctype:
   for method in ('getItem','setItem','removeItem'):
    for key in ('SESSION_KEY','ACCOUNT_KEY','ACCOUNT_DB_KEY',"'nevu_annotation_session_user_v2'",'"nevu_annotation_session_user_v2"',"'nevu_annotation_accounts_v2'",'"nevu_annotation_accounts_v2"'):
     data=data.replace(('localStorage.'+method+'('+key).encode(),('window.nevuAuthStorage.'+method+'('+key).encode())
  if (ctype.startswith('text/html') or 'javascript' in ctype) and PUBLIC_PREFIX and self.headers.get('X-Forwarded-Proto'):
   text=data.decode('utf-8')
   for quote in ('"', "'", '`'):
    for prefix in ('/original/','/remaining/','/quality-review/','/values.html','/api/','/crossdoc-','/relation-assets/'):
     text=text.replace(quote+prefix,quote+PUBLIC_PREFIX+prefix)
   text=text.replace('href="/"','href="'+PUBLIC_PREFIX+'/"').replace('url=/?','url='+PUBLIC_PREFIX+'/?')
   text=text.replace('/api/','/api\\u002f')
   data=text.encode('utf-8')
  self.send_response(200);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
 def do_GET(self):
  path=urlparse(self.path).path
  if path.startswith('/crossdoc-assets/'):
   name=path.removeprefix('/crossdoc-assets/')
   if name not in ('ui.js','ui.css','auth.html','dashboard.js'):self.send_error(404);return
   p=BASE/'dashboard_auth.html' if name=='auth.html' else ADDON/name
   self.send_content(p.read_bytes(),{'ui.js':'text/javascript','dashboard.js':'text/javascript','ui.css':'text/css','auth.html':'text/html'}[name]+'; charset=utf-8');return
  return super().do_GET()
def make_server(host,port,legacy_root='remaining'):
 pages=WORK/('pages' if legacy_root=='original' else 'document_annotation/pages')
 store=AnnotationStore(WORK/'server_data/annotations.jsonl') if legacy_root=='original' else ShardedAnnotationStore(WORK/'document_annotation/server_data')
 server=AnnotationServer((host,port),CrossdocHandler,pages_dir=pages,store=store)
 for mode in ['quality','annotation','quality_after']:
  directory=(DATA/mode/'server_data').resolve();registry=json.loads((DATA/mode/'registry.json').read_text());server.annotation_areas['/crossdoc-'+mode.replace('_','-')+'/']=(DATA/mode/'pages',directory);server.area_stores[directory]=CrossdocStore(directory,registry,'quality' if mode=='quality_after' else mode)
 return server
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=8002);ap.add_argument('--legacy-root',choices=['original','remaining'],default='remaining');ap.add_argument('--public-prefix',default='');a=ap.parse_args();PUBLIC_PREFIX=a.public_prefix.rstrip('/');server=make_server(a.host,a.port,a.legacy_root);print(f'Five-module dashboard: http://{a.host}:{a.port}',flush=True)
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()
