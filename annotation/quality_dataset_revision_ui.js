// Version the evaluated content independently from the annotator's scores.
const originalQualityGetState=getState;
getState=function(r){
  const rid=r.original.relation_id,rev=String(r.original._dataset_revision||'0'),s=states[rid];
  if(s&&String(s._dataset_revision||'0')!==rev){
    try{localStorage.setItem(storageKey()+':previous:'+rid+':'+String(s._dataset_revision||'0'),JSON.stringify(s))}catch(e){}
    states[rid]=defaultState(r);
  }
  const result=originalQualityGetState(r);result._dataset_revision=rev;return result;
};
const originalQualityRender=render;
render=function(){
  // Reset obsolete scores before counters or bulk saves can reuse them.
  RELATIONS.forEach(r=>{if(states[r.original.relation_id])getState(r)});
  originalQualityRender();
  if(RELATIONS[idx]?.original?._dataset_revision){
    const notice=document.createElement('div');notice.className='warning';notice.textContent='An administrator revised this item；These ratings assess revision。The new version needs new ratings；Only this account’s latest ratings are retained。';
    document.getElementById('app').prepend(notice);
  }
};

// Save only real edits, and let either explicit button flush the current item.
const qualitySeenContent=new Map();
const qualityOriginalLoad=loadServerStates;
loadServerStates=async function(){await qualityOriginalLoad();qualitySeenContent.clear()};
function qualityContent(s){const x=cloneJson(s);delete x._client_updated_at;delete x._manual_confirmed;delete x._manual_confirmed_at;return JSON.stringify(x)}
const qualityGetStateWithRevision=getState;
getState=function(r){
  const s=qualityGetStateWithRevision(r),rid=r.original.relation_id;
  if(!qualitySeenContent.has(rid))qualitySeenContent.set(rid,qualityContent(s));
  return s;
};
save=function(){
  if(!annotator)return;
  const [rid,state]=currentRelationAndState();if(!rid||!state)return;
  const signature=qualityContent(state),changed=qualitySeenContent.get(rid)!==signature;
  if(changed){state._manual_confirmed=false;state._client_updated_at=isoNow();qualitySeenContent.set(rid,signature)}
  localStorage.setItem(storageKey(),JSON.stringify(states));
  if(changed){const footer=document.getElementById('qualitySaveFooter');if(footer)footer.style.background='';markSaved(SERVER_SYNC_ENABLED?'This item is temporarily stored，Synchronizing automatically…':'This item is saved in the browser');queueServerSaveRelation(rid,state)}
};
let qualityExplicitSaving=false;
async function saveQualityCurrent(confirm=false){
  if(qualityExplicitSaving||!annotator)return false;
  qualityExplicitSaving=true;
  const buttons=[document.getElementById('saveQualityCurrent'),document.getElementById('saveProgress')].filter(Boolean);
  buttons.forEach(b=>b.disabled=true);
  try{
    harvestQuality();
    const [rid,state]=currentRelationAndState();if(!rid||!state)return false;
    if(confirm&&!scoresComplete(RELATIONS[idx],state))throw new Error("Complete all applicable ratings first，Then confirm and save。");
    clearTimeout(serverSaveTimers[rid]);
    const payload={annotator,group:String(GROUP),relation_id:rid,state:cloneJson(state),protocol:ANNOTATION_PROTOCOL,client_updated_at:String(state._client_updated_at||'')};
    if(confirm){payload.state._manual_confirmed=true;payload.state._manual_confirmed_at=state._manual_confirmed&&state._manual_confirmed_at?state._manual_confirmed_at:isoNow()}
    const response=await sendStateToServer(payload);
    if(!response.ok)throw new Error('Open this page through the server before saving');
    if(confirm&&qualityContent(state)===qualityContent(payload.state)){state._manual_confirmed=true;state._manual_confirmed_at=payload.state._manual_confirmed_at;localStorage.setItem(storageKey(),JSON.stringify(states))}
    markSaved((state._manual_confirmed?'This item is confirmed and saved to the server':'Draft saved for this item；Confirm below before proceeding')+' · '+(state.completed?'Ratings complete':'Ratings incomplete'));
    const footer=document.getElementById('qualitySaveFooter');if(footer)footer.style.background=state._manual_confirmed?'#ecfdf3':'';
    return true;
  }catch(e){markSaved('Failed to save this item：'+e.message,true);return false}
  finally{qualityExplicitSaving=false;buttons.forEach(b=>b.disabled=false)}
}
// Legacy logout also calls syncAllStates: flush only the item being left.
syncAllStates=()=>saveQualityCurrent(false);
const qualityRenderWithRevision=render;
render=function(){
  qualityRenderWithRevision();
  const footer=document.createElement('div');footer.className='stageBox';footer.id='qualitySaveFooter';footer.style.background=getState(RELATIONS[idx])._manual_confirmed?'#ecfdf3':'';
  footer.innerHTML='<button id="saveQualityCurrent" type="button">Confirm this item and save to server</button><span id="qualityCurrentSaveStatus" class="small" style="margin-left:12px">Click here to confirm before proceeding。Autosave is a draft backup only。</span>';
  document.getElementById('app').append(footer);
  document.getElementById('saveQualityCurrent').onclick=()=>saveQualityCurrent(true);
};
const qualityMarkSaved=markSaved;
markSaved=function(message,isError=false){qualityMarkSaved(message,isError);const el=document.getElementById('qualityCurrentSaveStatus');if(el){el.textContent=message;el.style.color=isError?'#b42318':'#087a3e'}};
document.getElementById('saveProgress').textContent='Save this item’s draft';
document.getElementById('saveProgress').onclick=()=>saveQualityCurrent(false);
move=function(d){
  harvestQuality();
  if(d>0&&!getState(RELATIONS[idx])._manual_confirmed){alert('First click below“Confirm this item and save to server”，Save successfully before the next relation。');return}
  idx=Math.max(0,Math.min(RELATIONS.length-1,idx+d));render();window.scrollTo(0,0);
};
