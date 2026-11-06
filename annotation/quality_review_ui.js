function defaultState(r){
  const qs={};
  r.original.queries.forEach(q=>qs[q.query_id]={query_quality:"",query_evidence_consistency:"",note:""});
  return {direct_evidence_validity:r.original.direct_doc_ids.length?"":null,
    partial_evidence_validity:r.original.partial_doc_ids.length?"":null,
    evidence_set_sufficiency:"",evidence_note:"",queries:qs,completed:false,completed_at:""};
}
function getState(r){const rid=r.original.relation_id;return states[rid]||(states[rid]=defaultState(r))}
function scored(x){return ["0","1","2"].includes(String(x))}
function scoresComplete(r,s){
  return (!r.original.direct_doc_ids.length||scored(s.direct_evidence_validity)) &&
    (!r.original.partial_doc_ids.length||scored(s.partial_evidence_validity)) &&
    scored(s.evidence_set_sufficiency) && r.original.queries.every(q=>{
      const z=s.queries[q.query_id];return scored(z.query_quality)&&scored(z.query_evidence_consistency);
    });
}
function harvestQuality(){
  if(!annotator||!RELATIONS.length)return;
  const r=RELATIONS[idx],s=getState(r);
  for(const [id,key] of [["dv","direct_evidence_validity"],["pv","partial_evidence_validity"],["es","evidence_set_sufficiency"],["evnote","evidence_note"]]){
    const el=document.getElementById(id);if(el)s[key]=el.value;
  }
  for(const q of r.original.queries){
    const z=s.queries[q.query_id];
    for(const [prefix,key] of [["qq_","query_quality"],["qc_","query_evidence_consistency"],["note_","note"]]){
      const el=document.getElementById(prefix+q.query_id);if(el)z[key]=el.value;
    }
  }
  const completed=scoresComplete(r,s);
  if(completed&&!s.completed)s.completed_at=isoNow();
  if(!completed)s.completed_at="";
  s.completed=completed;save();updateCounter();
}
function updateCounter(){
  const n=RELATIONS.filter(r=>states[r.original.relation_id]?.completed).length;
  document.getElementById("counter").textContent=`Relation ${idx+1}/${RELATIONS.length} · Complete ${n}/${RELATIONS.length} · Current${getState(RELATIONS[idx]).completed?"All fields completed（Editable）":"To rate"}`;
}
function render(){
  const r=RELATIONS[idx],o=r.original,s=getState(r);
  const sentences=o.article_sentences.map(x=>{
    const d=o.direct_doc_ids.includes(String(x.doc_id)),p=o.partial_doc_ids.includes(String(x.doc_id));
    return `<div class="sentence ${d?"direct":p?"partial":""}" style="grid-template-columns:70px 1fr"><div class="sid">S${esc(x.sentence_id)}<br>${d?"Direct":p?"Partial":""}</div><div>${esc(x.text)}</div></div>`;
  }).join("");
  const queries=o.queries.map(q=>{
    const z=s.queries[q.query_id];
    return `<div class="qbox"><div class="qtype">${esc(q.query_type)} · ${esc(q.query_id)}</div>
      <div class="small">${esc(queryTypeHint(q.query_type))}</div>
      <div class="original">${esc(q.query)}</div>
      <div class="rating"><label>Query Semantic Faithfulness ${scoreSelect("qq_"+q.query_id,z.query_quality,"query_faithfulness")}</label>
      <label>Query–Evidence Consistency ${scoreSelect("qc_"+q.query_id,z.query_evidence_consistency,"query_evidence_consistency")}</label></div>
      <input id="note_${esc(q.query_id)}" value="${esc(z.note)}" placeholder="Optional：Rating instructions" style="width:100%"></div>`;
  }).join("");
  document.getElementById("app").innerHTML=`<div class="card"><div class="meta">${[
    ["Scope",o.unit_level],["Actor",o.actor_name],["Value",o.value_name],["Direction",o.direction],
    ["Relation ID",o.relation_id],["Article",o.article_title||o.guid],["Unit ID",o.unit_id],["Scope title",o.scope_title]
  ].map(([label,value])=>`<div><div class="label">${esc(label)}</div><div class="val">${esc(value)}</div></div>`).join("")}</div>
  <div class="block"><h3>Semantic unit</h3><div class="explanation">${esc(o.unit_text)}</div></div>
  <div class="block"><h3>NEVU explanation</h3><div class="explanation">${esc(o.nevu_explanation)}</div></div>
  <div class="block"><h3>test_human Full article and revised evidence（Read-only）</h3><div class="sentences">${sentences}</div></div>
  <div class="stageBox"><div class="stageTitle">Stage 1 — Post-revision data quality assessment</div>
  <div class="small">Assess only the current test_human。Direct Should independently support the complete relation，Partial Provides relevant but incomplete support；Sufficiency Assess evidence-set sufficiency。Judge semantically query，Preserve valid implicit wording。Complete all applicable ratings，Editable at any time。</div>
  <div class="rating"><label>Direct Evidence Validity ${o.direct_doc_ids.length?scoreSelect("dv",s.direct_evidence_validity,"direct_evidence_validity"):"N/A — None Direct Evidence"}</label>
  <label>Partial Evidence Validity ${o.partial_doc_ids.length?scoreSelect("pv",s.partial_evidence_validity,"partial_evidence_validity"):"N/A — None Partial Evidence"}</label>
  <label>Evidence Set Sufficiency ${scoreSelect("es",s.evidence_set_sufficiency,"evidence_set_sufficiency")}</label></div>
  <textarea id="evnote" class="notes" placeholder="Optional：Evidence-rating instructions">${esc(s.evidence_note)}</textarea>
  ${queries}</div></div>`;
  updateCounter();
}
function move(d){harvestQuality();idx=(idx+d+RELATIONS.length)%RELATIONS.length;render();window.scrollTo(0,0)}
function download(name,text,type="application/json"){
  const blob=new Blob([text],{type}),a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=name;a.click();URL.revokeObjectURL(a.href);
}
function exportReview(completedOnly){
  harvestQuality();
  const exported={};
  RELATIONS.forEach(r=>{const rid=r.original.relation_id,s=states[rid];if(s&&(!completedOnly||s.completed))exported[rid]=s});
  download(`quality_review_${safeAnnotator()}_group_${String(GROUP).padStart(2,"0")}_${completedOnly?"completed":"all"}.json`,JSON.stringify({annotator,group:GROUP,protocol:ANNOTATION_PROTOCOL,dataset:"phase5/output/test_human",exported_at:isoNow(),states:exported},null,2));
}
document.getElementById("prev").onclick=()=>move(-1);
document.getElementById("next").onclick=()=>move(1);
document.getElementById("saveProgress").onclick=async()=>{harvestQuality();if(SERVER_SYNC_ENABLED)await syncAllStates()};
document.getElementById("exportData").onclick=()=>exportReview(true);
document.getElementById("exportLog").onclick=()=>exportReview(false);
document.getElementById("logoutBtn").onclick=logoutAccount;
document.getElementById("loginBtn").onclick=loginAccount;
document.getElementById("registerBtn").onclick=registerAccount;
document.getElementById("authPass").addEventListener("keydown",e=>{if(e.key==="Enter")loginAccount()});
document.getElementById("authPass2").addEventListener("keydown",e=>{if(e.key==="Enter")registerAccount()});
document.addEventListener("input",e=>{if(e.target.closest("#app"))harvestQuality()});
document.addEventListener("change",e=>{if(e.target.closest("#app"))harvestQuality()});
window.addEventListener("beforeunload",()=>{harvestQuality();beaconCurrentState()});
bootAuth();
