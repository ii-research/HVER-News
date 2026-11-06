(()=>{
 const modules=[['crossdoc_quality','04 · Pre-annotation cross-document quality assessment'],['crossdoc_direct','05 · Cross-document Direct Nonempty'],['crossdoc_partial','06 · Cross-document only Partial'],['crossdoc_empty','07 · Cross-document no-evidence'],['crossdoc_quality_after','Post-annotation cross-document quality assessment']];
 for(const [v,t] of modules){const o=document.createElement('option');o.value=v;o.textContent=t;$('moduleSelect').append(o)}
 // Restore before the dashboard starts its first authenticated request.
 const moduleSelect=$('moduleSelect'), moduleStorageKey='nevu_admin_selected_module';
 try{
  const saved=localStorage.getItem(moduleStorageKey);
  if(saved&&Array.from(moduleSelect.options).some(o=>o.value===saved)){
   MODULE=saved;moduleSelect.value=saved;
  }
 }catch(e){/* Storage may be disabled; keep the normal default. */}
 moduleSelect.addEventListener('change',()=>{
  try{localStorage.setItem(moduleStorageKey,moduleSelect.value)}catch(e){}
 });
 const panel=document.createElement('section');panel.id='crossdocAdmin';panel.hidden=true;panel.style='max-width:1500px;margin:20px auto;padding:16px';document.querySelector('main').after(panel);
 let selected='',group='',page=1,lastModule='',version=0;
 const names={unseen:'Not started',draft:'Unconfirmed draft',confirmed:'Saved',invalid:'Invalid; saved',needs_review:'Saved; pending review'};
 const dims={actor_match_quality:'Actor match',scope_match_quality:'Event compatibility',direct_evidence_validity:'Direct Validity',partial_evidence_validity:'Partial Validity',evidence_set_sufficiency:'Set completeness',query_evidence_consistency:'Query Consistency with evidence'};
 const qualityModules=['crossdoc_quality','crossdoc_quality_after'];
 const rate=x=>x===null||x===undefined?'—':(100*Number(x)).toFixed(1)+'%';
 function qualityTable(scores,title='Quality metrics'){
  const rows=Object.entries(scores||{});
  return `<section class="card section"><h2>${esc(title)}</h2><p class="small">Strict：Rating=2；Soft：Rating≥1。NThe denominator includes only confirmed、valid with this dimension rated0/1/2records；Invalid、Draft、Not saved、N/Aand“Unknown”Excluded from the denominator。</p><div class="scroll"><table class="table"><thead><tr><th>Rating dimension</th><th>N</th><th>Mean / 2</th><th>Strict (=2)</th><th>Soft (≥1)</th><th>Unknown</th></tr></thead><tbody>${rows.map(([k,v])=>`<tr><td>${esc(dims[k]||k)}</td><td>${v.n}</td><td>${v.mean===null?'—':v.mean.toFixed(2)}</td><td>${rate(v.strict_2_rate)}</td><td>${rate(v.soft_ge1_rate)}</td><td>${v.unknown}</td></tr>`).join('')||'<tr><td colspan="6">No valid confirmed ratings</td></tr>'}</tbody></table></div></section>`;
 }
 function qualitySummary(summary){
  const s=summary||{};
  return `<div class="grid section">${metric('Complete Group',s.completed_group_assignments||0,'By account–GroupCalculate')}${metric('Records included in statistics',s.included_quality_pairs||0,'Complete groups onlyGroupvalid records in')}${metric('Overall Strict',rate(s.strict_2_rate),'All applicable rating dimensions pooled')}${metric('Overall Soft',rate(s.soft_ge1_rate),'All applicable rating dimensions pooled')}</div>`;
 }

 window.crossdocGroupSummary=function(d){
  const rows=d.group_summary||[];
  return `<section class="card section"><h2>Group Overview · Who is working、Remaining work</h2><p class="small">Show only groups with server-saved records Group，One row per annotator。Counted by relation page；All items on the page query Counted complete only after saving。“First incomplete”Identifies skipped pages，“Furthest page with saved activity”Shown from saved records，Not the current browser page。</p><div class="scroll" style="max-height:520px"><table class="table"><thead><tr><th>Group</th><th>Annotator</th><th>Complete / Total relation pages</th><th>Remaining relation pages</th><th>First incomplete page</th><th>Furthest page with saved activity</th><th>Confirmed query pairs</th><th>Draft query pairs</th><th>Last saved</th></tr></thead><tbody>${rows.map(r=>`<tr><td>Group ${esc(r.group)}</td><td>${esc(r.annotator)}</td><td>${r.pages_done} / ${r.pages}</td><td>${r.remaining}</td><td>${r.first_pending===null?'All complete':r.first_pending}</td><td>${r.last_touched}</td><td>${r.saved_pairs} / ${r.total_pairs}</td><td>${r.draft_pairs}</td><td>${esc(r.last_saved?new Date(r.last_saved).toLocaleString('en-US',{hour12:false}):'—')}</td></tr>`).join('')||'<tr><td colspan="9">No saved annotation records</td></tr>'}</tbody></table></div></section>`;
 };
 const previousLoad=loadOverview;
 loadOverview=async function(){
  if(!MODULE.startsWith('crossdoc_')){version++;panel.hidden=true;document.querySelector('main').hidden=false;return previousLoad()}
  document.querySelector('main').hidden=true;panel.hidden=false;$('editDatasetButton').hidden=true;
  if(lastModule!==MODULE){selected='';group='';page=1;lastModule=MODULE}
  const token=++version;panel.textContent='Loading cross-document statistics…';
  try{
   const d=await api('/api/admin/crossdoc/overview?'+new URLSearchParams({annotator:selected,group,page}));if(token!==version)return;
   $('connection').textContent='Updated '+new Date().toLocaleTimeString();
   if(MODULE==='crossdoc_direct'){
    window.renderDirectAdmin(d,panel,async who=>{selected=who;group='';page=1;await loadOverview()},async pairId=>{const x=await api('/api/admin/crossdoc/detail?'+new URLSearchParams({annotator:selected,pair_id:pairId}));window.showCrossdocAdminDetail(x)},async()=>{const result=await api('/api/admin/crossdoc/export?'+new URLSearchParams({annotator:selected}));const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=MODULE+'_'+(selected||'all')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
    return;
   }
   panel.innerHTML=`<h2>${esc(d.title)}</h2><p>${d.group_total} groups · ${d.page_total} Relation page · ${d.pair_total} pairs query–Article。Complete after every item is confirmed or saved as invalid。Count only groups with server-saved records Group；Includes unfinished pages in these groups。</p><p>Quality metrics include only fully completed account–Group；IncompleteGroupPartial records in incomplete groups are also excluded from the denominator。Invalid、Drafts and review-pending records are excluded from quality metrics。</p>${d.warnings.length?`<p class="bad">Data-reading warnings：${esc(d.warnings.join('; '))}</p>`:''}<div class="toolbar"><select id="cdUser"><option value="">Select annotator</option>${d.users.map(u=>`<option value="${esc(u.annotator)}" ${selected===u.annotator?'selected':''}>${esc(u.annotator)}</option>`).join('')}</select><button id="cdExport">Export${selected?'Current annotator':'This module'}Record JSON</button></div>${qualityModules.includes(MODULE)?qualitySummary(d.quality_summary)+qualityTable(d.scores,'Overall quality metrics'):''}<div class="card scroll"><table class="table"><thead><tr><th>Annotator</th><th>Complete Group</th><th>Completed pages / Total pages</th><th>Saved pairs / Total pairs</th><th>Completion rate</th><th>Invalid</th><th>Pending review</th><th>Draft</th><th>Pairs with evidence changes</th><th>Last saved</th></tr></thead><tbody>${d.users.map(u=>`<tr class="click" data-user="${esc(u.annotator)}"><td><button data-open-user="${esc(u.annotator)}">${esc(u.annotator)}</button></td><td>${u.completed_group_count}/${u.active_groups.length}</td><td>${u.pages_done}/${u.active_page_total}</td><td>${u.saved}/${u.active_pair_total}</td><td>${rate(u.completion_rate)}</td><td>${u.invalid}</td><td>${u.needs_review}</td><td>${u.draft}</td><td>${u.evidence_changes?`${u.evidence_changes.changed||0} / ${u.evidence_changes.eligible_pairs||0}`:'Quality ratings only'}</td><td>${esc(u.last_saved)}</td></tr>`).join('')||'<tr><td colspan="10">No human saves yet</td></tr>'}</tbody></table></div><div id="cdDetails"></div>`;
   panel.insertAdjacentHTML('afterbegin',window.crossdocGroupSummary(d));
   $('cdUser').onchange=()=>{selected=$('cdUser').value;group='';page=1;loadOverview()};
   panel.querySelectorAll('[data-open-user]').forEach(button=>button.onclick=()=>{selected=button.dataset.openUser;group='';page=1;loadOverview()});
   $('cdExport').onclick=async()=>{try{const result=await api('/api/admin/crossdoc/export?'+new URLSearchParams({annotator:selected}));const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=MODULE+'_'+(selected||'all')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}catch(e){$('connection').textContent=e.message}};
   if(!selected)return;
   $('cdDetails').innerHTML=`${window.crossdocChangeSummary(d.evidence_changes)}<h3>${esc(selected)} · Groups and records</h3>${qualityModules.includes(MODULE)?qualitySummary(d.quality_summary)+qualityTable(d.scores,'Quality metrics for this annotator'):''}<div class="toolbar"><select id="cdGroup"><option value="">All Group</option>${d.groups.map(g=>`<option value="${g.group}" ${group===g.group?'selected':''}>Group ${g.group} · ${g.pages_done}/${g.pages} pages · ${g.saved}/${g.pairs} pairs</option>`).join('')}</select><button id="cdPrev" ${page<=1?'disabled':''}>Previous page</button><span>${page}/${Math.max(1,Math.ceil(d.row_total/50))} · ${d.row_total} pairs</span><button id="cdNext" ${page*50>=d.row_total?'disabled':''}>Next page</button></div>${qualityModules.includes(MODULE)?'':`<p>Valid saved document labels：${esc(JSON.stringify(d.labels))}</p>`}<div class="scroll"><table class="table"><tr><th>Group</th><th>Query</th><th>Candidate article</th><th>Status</th><th>Human D/P</th><th></th></tr>${d.rows.map(r=>`<tr><td>${r.group}</td><td>${esc(r.query_type)} · ${esc(r.query)}</td><td>${esc(r.doc_id)}</td><td>${names[r.status]}</td><td>${r.status==='unseen'?'—':r.direct+'/'+r.partial}</td><td><button data-detail="${esc(r.pair_id)}">View ratings and full article</button></td></tr>`).join('')}</table></div><div id="cdRecord" class="card" hidden></div>`;
   $('cdGroup').onchange=()=>{group=$('cdGroup').value;page=1;loadOverview()};$('cdPrev').onclick=()=>{page--;loadOverview()};$('cdNext').onclick=()=>{page++;loadOverview()};
   panel.querySelectorAll('[data-detail]').forEach(b=>b.onclick=async()=>{try{const x=await api('/api/admin/crossdoc/detail?'+new URLSearchParams({annotator:selected,pair_id:b.dataset.detail}));window.showCrossdocAdminDetail(x);}catch(e){$('connection').textContent=e.message}});
  }catch(e){if(token===version)panel.textContent='Load failed：'+e.message;throw e}
 };
 $('refreshBtn').onclick=()=>loadOverview().catch(e=>{$('connection').textContent=e.message});
 function stateStatus(s){return !Object.keys(s).length?'unseen':!s.human_confirmed?'draft':s.invalid?'invalid':s.annotation_status==='needs_review'||s.review_uncertain?'needs_review':'confirmed'}
})();
