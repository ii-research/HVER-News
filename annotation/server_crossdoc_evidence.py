"""Serve evidence-specific annotation modules."""
import argparse,json
from urllib.parse import urlparse
import server_crossdoc as old
from crossdoc_addon.store import CrossdocStore
from crossdoc_addon.build_evidence_modules import OUT,TITLES
SPECS=[('direct','crossdoc_annotation','Cross-document annotation · Direct Nonempty','Direct At least one sentence；Partial May be empty。Review first。'),('partial','crossdoc_partial','Cross-document annotation · Only Partial','Direct is empty、Partial At least one sentence；Additional review if time permits。'),('empty','crossdoc_empty','Cross-document annotation · No evidence','Direct and Partial both empty；Distinguish model non-support from uncertainty，Can be spot-checked later。')]
class EvidenceHandler(old.CrossdocHandler):
 def do_GET(self):
  path=urlparse(self.path).path
  if path=='/crossdoc-assets/dashboard.js':
   self.send_content((old.ADDON/'evidence_dashboard.js').read_bytes(),'text/javascript; charset=utf-8');return
  if path=='/crossdoc-evidence-assets/ui.js':
   self.send_content((old.ADDON/'evidence_ui.js').read_bytes(),'text/javascript; charset=utf-8');return
  # Old page links lead to the new Direct queue. Old API routes remain intact
  # so an already-open v1 page cannot write its old group IDs into the new queue.
  if path.startswith('/crossdoc-annotation/') and '/api/' not in path:
   self.send_response(302);self.send_header('Location','/crossdoc-direct/group_01.html');self.end_headers();return
  return super().do_GET()
 def _serve_static(self,url_path):
  if getattr(self,'_area_prefix','') in tuple('/crossdoc-'+k+'/' for k,_,_,_ in SPECS):
   path=(self.server.pages_dir/url_path.lstrip('/')).resolve()
   try:path.relative_to(self.server.pages_dir)
   except ValueError:self.send_error(403);return
   if not path.is_file():self.send_error(404);return
   self.send_content(path.read_bytes(),'text/html; charset=utf-8' if path.suffix=='.html' else 'application/json; charset=utf-8');return
  return super()._serve_static(url_path)
 def send_content(self,data,ctype):
  if ctype.startswith('text/html') and urlparse(self.path).path in ('/','/index.html'):
   text=data.decode();entries={};config={}
   for k,key,title,desc in SPECS:
    entries[key]=json.loads((OUT/k/'index.json').read_text());config[key]={'title':title,'desc':desc,'note':'Based on the frozen Gemini Evidence classification；Human decisions can be edited。Each 20 pages per storage shard，Kept separate。','tag':'Annotate first' if k=='direct' else 'Optional follow-up review','prefix':'/crossdoc-'+k+'/'}
   injection='Object.assign(DATA,'+json.dumps(entries,ensure_ascii=False)+');Object.assign(CONFIG,'+json.dumps(config,ensure_ascii=False)+');\n'
   text=text.replace('let remainingProgress=',injection+'let remainingProgress=',1).replace('One account for all five task types','One account for all seven task types').replace('Five task types · Save independently','Eight task types · Separate storage')
   data=text.encode()
  return super().send_content(data,ctype)
def make_server(host,port,legacy_root='remaining'):
 server=old.make_server(host,port,legacy_root);server.RequestHandlerClass=EvidenceHandler
 for k,_,_,_ in SPECS:
  path=(OUT/k/'server_data').resolve();server.annotation_areas['/crossdoc-'+k+'/']=(OUT/k/'pages',path);server.area_stores[path]=CrossdocStore(path,json.loads((OUT/k/'registry.json').read_text()),'annotation')
 return server
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=8001);a=ap.parse_args();server=make_server(a.host,a.port);print(f'Evidence modules: http://{a.host}:{a.port}',flush=True)
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()
