let DATASET_PAGE=1,DATASET_RECORD=null,DATASET_REQUEST=0;
const datasetPanel=document.createElement('div');datasetPanel.id='datasetPanel';datasetPanel.className='editor hidden';
datasetPanel.innerHTML=`<div class="editorHeader"><button id="closeDataset">← Back to admin</button><strong>02 · Edit reassessment data</strong><div class="spacer"></div><button id="exportDataset">Export current data JSON</button></div>
<div class="editorMain"><div class="card"><p>These edits change the reassessment content seen by everyone，Not an individual annotator score。Original test_human Snapshot retained，Revisions are stored separately。After saving，Reopen the relevant 02 page to see the changes；The revised item needs new ratings；Only this account’s latest ratings are retained。</p>
<div class="toolbar"><select id="datasetGroup"><option value="">All Group</option></select><input id="datasetSearch" placeholder="Search relation、guid、query or content"><button id="searchDataset">Search</button><span id="datasetCount" class="small"></span></div>
<div class="scroll"><table class="table"><thead><tr><th>Group</th><th>Relation / GUID</th><th>Actor / Value</th><th>Version</th><th></th></tr></thead><tbody id="datasetRows"></tbody></table></div><div class="toolbar"><button id="datasetPrev">Previous page</button><button id="datasetNext">Next page</button><span id="datasetMessage" class="msg"></span></div></div>
<div id="datasetForm" class="section hidden"></div></div>`;
document.body.append(datasetPanel);
const datasetButton=document.createElement('button');datasetButton.id='editDatasetButton';datasetButton.textContent='Edit 02 Reassessment data';datasetButton.hidden=true;
$('moduleSelect').after(datasetButton);
const previousApplyModuleView=applyModuleView;
applyModuleView=function(){previousApplyModuleView();datasetButton.hidden=MODULE!=='quality'};
async function loadDatasetList(){
 const request=++DATASET_REQUEST;
 const d=await api('/api/admin/dataset/list?'+new URLSearchParams({group:$('datasetGroup').value,q:$('datasetSearch').value,page:String(DATASET_PAGE)}));
 if(request!==DATASET_REQUEST)return;
 const selected=$('datasetGroup').value;$('datasetGroup').innerHTML='<option value="">All Group</option>'+d.groups.map(g=>`<option value="${esc(g)}">Group ${esc(g)}</option>`).join('');$('datasetGroup').value=selected;
 $('datasetCount').textContent=`${d.total} items relation · No. ${d.page}/${Math.max(1,Math.ceil(d.total/50))} pages`;
 $('datasetPrev').disabled=d.page<=1;$('datasetNext').disabled=d.page*50>=d.total;
 $('datasetRows').innerHTML=d.rows.map(r=>`<tr><td>${esc(r.group)}</td><td>${esc(r.relation_id)}<div class="small">${esc(r.guid)}</div></td><td>${esc(r.actor_name)}<br>${esc(r.value_name)}</td><td>${r._dataset_revision?'Revised':'Initial version'}</td><td><button data-edit-relation="${esc(r.relation_id)}">Edit data</button></td></tr>`).join('');
 $('datasetRows').querySelectorAll('[data-edit-relation]').forEach(button=>button.onclick=()=>datasetAction(()=>openDatasetRelation(button.dataset.editRelation)));
}
async function datasetAction(action){try{await action();$('datasetMessage').textContent=''}catch(e){$('datasetMessage').textContent=e.message;$('datasetMessage').className='msg bad'}}
async function openDatasetRelation(rid){
 const d=await api('/api/admin/dataset/relation?relation_id='+encodeURIComponent(rid));DATASET_RECORD=d;
 const r=d.relation;$('datasetForm').classList.remove('hidden');
 $('datasetForm').innerHTML=`<div class="card"><h2>Group ${esc(r.group)} · ${esc(r.relation_id)}</h2><div class="small">GUID: ${esc(r.guid)} · Scope: ${esc(r.unit_level)} · Current version: ${esc(r._dataset_revision||'Initial version')}</div>
 <div class="two">${[['actor_name','Actor'],['value_name','Value'],['direction','Direction'],['scope_title','Scope title'],['unit_text','Semantic unit'],['nevu_explanation','NEVU explanation']].map(([k,label])=>`<label>${label}<textarea id="dataset_${k}" class="reason">${esc(r[k]||'')}</textarea></label>`).join('')}</div></div>
 <div class="section card"><h2>Query Text</h2>${r.queries.map((q,i)=>`<div class="query"><label>${esc(q.query_type)} · ${esc(q.query_id)}<textarea id="dataset_query_${i}">${esc(q.query)}</textarea></label></div>`).join('')}</div>
 <div class="section card"><h2>Evidence labels</h2><p class="small">Original article text is read-only；Select Direct、Partial or Non-support。Each sentence can have only one label。</p>${r.article_sentences.map((x,i)=>{const id=String(x.doc_id),kind=r.direct_doc_ids.includes(id)?'direct':r.partial_doc_ids.includes(id)?'partial':'none';return `<div class="sentence ${kind}"><span>S${esc(x.sentence_id)}</span><span>${esc(x.text)}</span><select id="dataset_evidence_${i}" aria-label="Sentence ${esc(x.sentence_id)} Evidence labels">${[['none','Non-support'],['direct','Direct'],['partial','Partial']].map(([k,l])=>`<option value="${k}" ${kind===k?'selected':''}>${l}</option>`).join('')}</select></div>`}).join('')}</div>
 <div class="section card"><h2>Change log</h2>${d.history.length?d.history.slice().reverse().map(h=>`<p>${esc(shortTime(h.edited_at))} · ${esc(h.reason)}</p>`).join(''):'No changes yet'}</div>
 <div class="saveBox"><textarea id="datasetReason" class="reason" placeholder="Required：Reason for this data edit"></textarea><button id="saveDataset" class="primary">Save and update 02 Page</button><span id="datasetSaveMsg" class="msg"></span></div>`;
 $('saveDataset').onclick=saveDataset;
 $('datasetForm').querySelectorAll('.sentence select').forEach(el=>el.onchange=()=>{el.closest('.sentence').className='sentence '+el.value});
 $('datasetForm').scrollIntoView({block:'start'});
}
async function saveDataset(){
 const r=DATASET_RECORD.relation,changes={},reason=$('datasetReason').value.trim();
 if(!reason){$('datasetSaveMsg').textContent='Enter a reason for the change';return}
 for(const k of ['actor_name','value_name','direction','scope_title','unit_text','nevu_explanation'])changes[k]=$('dataset_'+k).value;
 changes.queries=Object.fromEntries(r.queries.map((q,i)=>[q.query_id,$('dataset_query_'+i).value]));
 changes.direct_doc_ids=[];changes.partial_doc_ids=[];
 r.article_sentences.forEach((x,i)=>{const kind=$('dataset_evidence_'+i).value;if(kind!=='none')changes[kind+'_doc_ids'].push(String(x.doc_id))});
 $('saveDataset').disabled=true;
 try{
  const result=await api('/api/admin/dataset/save',{method:'POST',body:JSON.stringify({group:r.group,relation_id:r.relation_id,expected_revision:r._dataset_revision||'0',changes,reason})});
  await openDatasetRelation(r.relation_id);$('datasetSaveMsg').textContent=result.changed?'Saved。Reopen the relevant 02 page to assess the new version。':'No changes，No new version needed。';$('datasetSaveMsg').className='msg ok';
  await loadDatasetList();await loadOverview();
 }catch(e){$('datasetSaveMsg').textContent=e.message;$('datasetSaveMsg').className='msg bad'}finally{$('saveDataset').disabled=false}
}
datasetButton.onclick=()=>{datasetPanel.classList.remove('hidden');datasetAction(loadDatasetList)};
$('closeDataset').onclick=()=>datasetPanel.classList.add('hidden');
$('searchDataset').onclick=()=>{DATASET_PAGE=1;datasetAction(loadDatasetList)};
$('datasetSearch').onkeydown=e=>{if(e.key==='Enter')$('searchDataset').click()};
$('datasetGroup').onchange=()=>{DATASET_PAGE=1;datasetAction(loadDatasetList)};
$('datasetPrev').onclick=()=>{DATASET_PAGE--;datasetAction(loadDatasetList)};
$('datasetNext').onclick=()=>{DATASET_PAGE++;datasetAction(loadDatasetList)};
$('exportDataset').onclick=()=>datasetAction(async()=>{
 const data=await api('/api/admin/dataset/export'),url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');
 a.href=url;a.download='quality_review_effective_relations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
