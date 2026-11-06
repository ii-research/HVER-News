'use strict';
const el=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data,states={},idx=0,user=null,ready=false,busy=false,loadVersion=0;
const mount='/crossdoc-quality-after/';
const publicBase=location.pathname.split(mount)[0];
const prefix=publicBase+mount;
const savedOnServer=new Set();
const extraFields=['actor_match_quality','scope_match_quality'];
const fields=[['actor_match_quality','Actor-match correctness'],['scope_match_quality','Event compatibility'],['direct_evidence_validity','Direct Evidence Validity — Direct evidence validity'],['partial_evidence_validity','Partial Evidence Validity — Partial evidence validity'],['evidence_set_sufficiency','Evidence Set Sufficiency — Evidence-set completeness'],['query_evidence_consistency','Query–Evidence Consistency — Query-evidence consistency']];
function key(){return `crossdoc_quality_after_v1_${data.group}_${user}`}
function getState(p){return states[p.pair_id]||(states[p.pair_id]={module:'quality',query_id:p.query_id,doc_id:p.doc_id,note:'',human_confirmed:false})}
function notice(s){el('status').textContent=s}

// Timing v1: page-open and first card interaction; end is the server save time.
const pageTimingStarts=new Map();
function trackTiming(p,interaction=false){
 if(!ready||!user)return;
 const s=getState(p),key=user+'|'+p.relation_id,now=new Date().toISOString();
 if(!pageTimingStarts.has(key))pageTimingStarts.set(key,now);
 if(!s.timing_v1)s.timing_v1={page_started_at:pageTimingStarts.get(key),relation_id:p.relation_id,clock:'browser',historical_record:s.human_confirmed===true};
 if(interaction&&!s.timing_v1.query_started_at){s.timing_v1.query_started_at=now;local()}
}

function local(){
 if(!user||!data||!ready)return true;
 try{
  const pending=Object.fromEntries(Object.entries(states).filter(([pid])=>!savedOnServer.has(pid)));
  if(Object.keys(pending).length)localStorage.setItem(key(),JSON.stringify(pending));else localStorage.removeItem(key());
  const w=el('qualityStorageWarning');if(w)w.hidden=true;return true;
 }catch(e){
  let w=el('qualityStorageWarning');if(!w){w=document.createElement('p');w.id='qualityStorageWarning';w.setAttribute('role','alert');w.style='padding:12px;background:#fff3cd;color:#664d03';el('status').after(w)}
  w.hidden=false;w.textContent='Browser draft cache unavailable，Server saves are unaffected。Confirm and save each item；Unsaved edits remain on the current page，Do not refresh or close，You can export a backup first。';return false;
 }
}
function scoreApplicable(k,p){return k==='direct_evidence_validity'?p.direct_sentence_ids.length:k==='partial_evidence_validity'?p.partial_sentence_ids.length:true}
function scoreComplete(s,p){return s.invalid===true||fields.every(([k])=>!scoreApplicable(k,p)||(extraFields.includes(k)?['0','1','2','unknown']:['0','1','2']).includes(s[k]))}
function collect(){if(!ready)return;
 el('app').querySelectorAll('[data-pair]').forEach(card=>{const p=data.pairs.find(p=>p.pair_id===card.dataset.pair);const s=states[p.pair_id]||(states[p.pair_id]={module:'quality',query_id:p.query_id,doc_id:p.doc_id,note:'',human_confirmed:false});let changed=false;
 card.querySelectorAll('[data-field]').forEach(input=>{if(!input.disabled&&String(s[input.dataset.field]??'')!==input.value){s[input.dataset.field]=input.value;changed=true}});
 for(const [k] of fields)if(!scoreApplicable(k,p))s[k]=null;
 if(changed){s.human_confirmed=false;savedOnServer.delete(p.pair_id);s._client_updated_at=new Date().toISOString();const label=card.querySelector('.quality-saved');if(label)label.textContent='Modified，Not yet saved';}
 });local();counter();}
function select(field,value,options){return `<select data-field="${field}"><option value="">Please select</option>${options.map(([v,label])=>`<option value="${v}" ${String(value)===v?'selected':''}>${label}</option>`).join('')}</select>`}
function counter(){el('counter').textContent=`Relation ${idx+1}/${data.tasks.length} · Saved in this group ${data.pairs.filter(p=>savedOnServer.has(p.pair_id)).length}/${data.pairs.length} pairs`;el('jump').max=data.tasks.length;el('jump').value=idx+1;}
function render(){if(!ready){el('app').innerHTML='<p>Start after sign-in and progress loading。</p>';return}
 const task=data.tasks[idx];for(const p of task)trackTiming(p);el('title').textContent=`Post-annotation cross-document quality assessment · Stage1 · Group ${data.group}`;
 el('notice').textContent='Sampled queries for the same triple query Grouped display，Each has separate ratings and evidence。Assess final annotations only（05Human／06 Fable），Do not modify query or evidence；White, unselected text does not imply non-support。';
 el('app').innerHTML=`<article><h2>Relation ${idx+1} · ${esc(task[0].reference.actor_name)}</h2><details><summary>Source relation reference（Not a fact about the candidate article）</summary><p>${esc(task[0].reference.value_name)} · ${esc(task[0].reference.direction)}</p><p>${esc(task[0].reference.unit_text)}</p><p>${esc(task[0].reference.nevu_explanation)}</p></details><p class="small">${task.length} sampled items query · Different query may have different candidate articles，Check each separately。</p><div class="quality-query-grid">${task.map(p=>{const s=states[p.pair_id]||{},doc=data.articles[p.doc_id];return `<section class="quality-query-card ${s.invalid?'quality-invalid':''}" data-pair="${p.pair_id}"><span class="badge">${esc(p.query_type)}</span><h3>${esc(p.query)}</h3><p class="small">${esc(p.query_id)}</p><button data-action="invalid" aria-pressed="${s.invalid===true}">${s.invalid?'Undo invalid flag':'Mark invalid／Unrelated'}</button>${s.invalid?'<p class="quality-invalid-notice">Marked invalid，No further ratings required。Please save this query；This item is excluded separately，Not recorded as all-zero scores。</p>':''}<div class="hint">${p.source==='claude_fable'?'Fable Post-annotation results':'Post-annotation result'}（Read-only）${p.post_annotation_status==='invalid'?' · Previously marked invalid':''}：${p.document_relevance===null?(p.post_annotation_status==='invalid'?'Invalid':'Unknown'):p.document_relevance+'grade'} · Actor ${esc(p.actor_match??'Unknown')} · Scope ${esc(p.scope_match??'Unknown')} · Direct ${p.direct_sentence_ids.length} sentences · Partial ${p.partial_sentence_ids.length} sentences</div><h3>Stage1 · Quality ratings</h3><fieldset class="quality-scores" ${s.invalid?'disabled':''}>${fields.map(([k,label])=>`<label>${esc(label)}${scoreApplicable(k,p)?select(k,s[k],[['2','2 — Matches'],['1','1 — Partially matches'],['0','0 — Does not match'],...(extraFields.includes(k)?[['unknown','Unknown']]:[])]):'<span class="small"> N/A — No evidence of this type</span>'}</label>`).join('')}</fieldset><label>Notes<textarea data-field="note">${esc(s.note||'')}</textarea></label><h3>Full candidate article · ${esc(doc.title)}</h3><p class="small">${esc(p.doc_id)} · Total ${doc.sentences.length} sentences<br>Green Direct · Yellow Partial</p><div class="quality-article-window" tabindex="0" role="region" aria-label="${esc(p.query_type)} Full candidate article">${doc.sentences.map((sentence,i)=>{const tag=p.direct_sentence_ids.includes(sentence.sentence_id)?'Direct':p.partial_sentence_ids.includes(sentence.sentence_id)?'Partial':'';return `<div class="quality-sentence ${tag}"><span class="small" title="${esc(sentence.sentence_id)}">S${i} ${tag}</span><p>${esc(sentence.text)}</p></div>`}).join('')}</div><button data-action="save-query" class="quality-save">${s.invalid?'Save as invalid and skip':'Confirm this query and save'}</button><p class="quality-saved">${savedOnServer.has(p.pair_id)?(s.invalid?'Invalid; saved':'Saved'):'Not yet saved'}</p></section>`}).join('')}</div><p class="small">Inspect the full article for missing evidence when rating completeness；Model judgment0Also check for missing evidence。Query-evidence consistency assesses only the current query Matches the candidate evidence。</p></article>`;
 el('app').querySelectorAll('[data-pair]').forEach(card=>{const p=data.pairs.find(p=>p.pair_id===card.dataset.pair);card.onpointerdown=card.onfocusin=()=>trackTiming(p,true);card.querySelector('.quality-article-window').onscroll=()=>trackTiming(p,true)});
 el('app').querySelectorAll('[data-action]').forEach(button=>{button.disabled=busy;button.onclick=()=>{if(busy)return;const card=button.closest('[data-pair]'),p=data.pairs.find(p=>p.pair_id===card.dataset.pair);if(button.dataset.action==='save-query'){saveQuery(p);return}collect();const s=getState(p);s.invalid=!s.invalid;s.human_confirmed=false;savedOnServer.delete(p.pair_id);s._client_updated_at=new Date().toISOString();local();render()}});
 if(busy)el('app').querySelectorAll('select,textarea,input').forEach(x=>x.disabled=true);
 el('app').oninput=()=>{collect();notice('Ratings modified；Click this query Use Save below to submit to the server。')};el('app').onchange=()=>collect();counter();}
async function saveQuery(p){
 if(!ready||busy)return;collect();const s=getState(p);
 if(!scoreComplete(s,p)){notice(`${p.query_type} Some ratings are incomplete，Complete the fields or mark invalid。`);return}
 trackTiming(p,true);s.timing_v1.query_submitted_at=new Date().toISOString();
 const payload=structuredClone({...s,quality_schema:'quality_query_save_v4',assessment_phase:'post_annotation',evidence_version:p.evidence_version,query_quality_not_assessed:true,human_confirmed:true,_client_updated_at:new Date().toISOString()});
 busy=true;render();notice('Saving this query…');
 try{const res=await fetch(prefix+'api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({annotator:user,group:String(data.group),relation_id:p.pair_id,state:payload,protocol:data.protocol,manual_save:true})});const x=await res.json();if(!res.ok||!x.ok)throw Error(x.error||'Save failed');states[p.pair_id]=payload;savedOnServer.add(p.pair_id);notice(`${p.query_type} ${s.invalid?'Invalid; saved':'Saved'}`);local()}
 catch(e){savedOnServer.delete(p.pair_id);s.human_confirmed=false;notice('Not saved：'+e.message);local()}
 finally{busy=false;render()}
}
function move(target){
 if(!ready||busy)return;target=Math.max(0,Math.min(data.tasks.length-1,target));collect();
 if(target>idx){const pending=data.tasks[idx].filter(p=>!savedOnServer.has(p.pair_id));if(pending.length){notice(`This relation still has ${pending.length} items query Not saved，Save each item before continuing。`);el('jump').value=idx+1;const card=el('app').querySelector(`[data-pair="${pending[0].pair_id}"]`);card?.scrollIntoView({behavior:'smooth',block:'center'});return}}
 idx=target;render();window.scrollTo(0,0)
}
async function auth(){if(!data)return;const version=++loadVersion;local();ready=false;render();user=localStorage.getItem('nevu_annotation_session_user_v2');let accounts={};try{accounts=JSON.parse(localStorage.getItem('nevu_annotation_accounts_v2')||'{}')}catch{}if(!user||!accounts[user]){user=null;return}
 states={};try{const res=await fetch(prefix+'api/load?annotator='+encodeURIComponent(user)+'&group='+data.group,{cache:'no-store'});const x=await res.json();if(!res.ok||!x.ok)throw Error(x.error||'Load failed');if(version!==loadVersion)return;states=x.states||{};savedOnServer.clear();for(const p of data.pairs){const s=states[p.pair_id];if(s?.human_confirmed&&scoreComplete(s,p))savedOnServer.add(p.pair_id)}const localStates=JSON.parse(localStorage.getItem(key())||'{}');for(const [k,v] of Object.entries(localStates)){if(!states[k]||String(v._client_updated_at||'')>String(states[k]._client_updated_at||'')){states[k]={...v,human_confirmed:false};savedOnServer.delete(k)}}states=Object.fromEntries(Object.entries(states).filter(([pid])=>data.pairs.some(p=>p.pair_id===pid)));ready=true;render();notice('Progress loaded')}catch(e){notice('Load failed，Editing disabled：'+e.message)}}
(async()=>{try{data=await (await fetch(location.pathname.replace(/\.html$/,'.json'))).json();const grouped=new Map();for(const p of data.pairs){if(!grouped.has(p.relation_id))grouped.set(p.relation_id,[]);grouped.get(p.relation_id).push(p)}data.tasks=[...grouped.values()].map(ps=>ps.sort((a,b)=>a.query_type.localeCompare(b.query_type)));if(!data.tasks.length){el('app').textContent='Candidates in this group were removed during article deduplication，Return to the annotation dashboard。';notice('This group has been cleared');return}el('prev').textContent='Previous relation';el('next').textContent='Next relation';el('jump').setAttribute('aria-label','Jump to relation number');el('title').textContent=(data.module==='quality'?'Post-annotation cross-document quality assessment · Stage1':'Cross-document annotation')+' · Group '+data.group;const authText=await (await fetch(publicBase+'/crossdoc-assets/auth.html')).text();el('login').innerHTML=authText;for(const old of el('login').querySelectorAll('script')){const script=document.createElement('script');script.textContent=old.textContent;old.replaceWith(script)}window.addEventListener('annotation-auth-change',auth);window.addEventListener('storage',e=>{if(e.key==='nevu_annotation_session_user_v2')auth()});await auth();
 el('prev').onclick=()=>move(idx-1);el('next').onclick=()=>move(idx+1);el('go').onclick=()=>move(Number(el('jump').value)-1);el('save').hidden=true;el('export').onclick=()=>{if(!ready)return;local();const blob=new Blob([JSON.stringify({module:data.module,group:data.group,annotator:user,protocol:data.protocol,states},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`crossdoc_${data.module}_${user}_group_${data.group}.json`;a.click();URL.revokeObjectURL(a.href)};window.addEventListener('beforeunload',local);
 }catch(e){notice('Page load failed：'+e.message)}})();
