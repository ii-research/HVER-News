// Additive cross-document progress; legacy rendering remains unchanged.
(()=>{
 const selectionKey='nevu_dashboard_selected_area_v1';
 const requestedArea=new URLSearchParams(location.search).get('area');
 let rememberedArea=null;try{rememberedArea=localStorage.getItem(selectionKey)}catch{}
 if(!requestedArea&&rememberedArea&&Object.hasOwn(CONFIG,rememberedArea))active=rememberedArea;
 function rememberArea(){
  try{localStorage.setItem(selectionKey,active)}catch{}
  const url=new URL(location.href);url.searchParams.set('area',active);
  try{history.replaceState(history.state,'',url)}catch{}
 }
 const previousChoose=choose;
 choose=function(area){if(!Object.hasOwn(CONFIG,area))return;previousChoose(area);rememberArea()};
 rememberArea();
 const oldRender=render,oldGroups=renderGroups;let pageProgress={},progress={},loading=false,loadedKey='',generation=0;
 function decorate(){if(!active.startsWith('crossdoc_'))return;
  const rows=DATA[active],user=localStorage.getItem('nevu_annotation_session_user_v2'),sum=document.getElementById('annotationSummary');sum.hidden=false;
  const byPage=CONFIG[active].prefix.startsWith('/crossdoc-rel-');
  sum.textContent=!user?'Sign in to view personal progress':loading?'Loading cross-document progress…':byPage?`Current account ${user} · Complete ${Object.values(pageProgress).reduce((a,x)=>a+Number(x),0)} / ${rows.reduce((a,x)=>a+x.relations,0)} pages（One page per relation；Complete only after all items on the page are confirmed）`:`Current account ${user} · Confirmed ${Object.values(progress).reduce((a,x)=>a+Number(x),0)} / ${rows.reduce((a,x)=>a+x.queries,0)} pairs（Modules save independently）`;
  document.querySelectorAll('#groups a').forEach(link=>{const match=link.getAttribute('href').match(/group_(\d+)\.html/);if(!match)return;const n=Number(match[1]),r=rows.find(x=>x.group===n);if(!r)return;const meta=link.querySelector('.meta');if(meta)meta.textContent=byPage?`${r.relations} pages · Total ${r.queries} pairs query–Article`:`${r.queries} pairs query–Article`;if(user&&!loading){let extra=link.querySelector('.cross-progress');if(!extra){extra=document.createElement('div');extra.className='cross-progress small';link.append(extra)}extra.textContent=byPage?`Complete ${pageProgress[n]||0}/${r.relations} pages`:`${progress[n]||0}/${r.queries} Confirmed`;link.classList.toggle('confirmed',byPage?(pageProgress[n]||0)>=r.relations:(progress[n]||0)>=r.queries)}});
 }
 async function refresh(){if(!active.startsWith('crossdoc_'))return;const user=localStorage.getItem('nevu_annotation_session_user_v2'),key=active+'|'+user;if(key===loadedKey){decorate();return}loadedKey=key;progress={};pageProgress={};loading=!!user;decorate();if(!user)return;const version=++generation;
  try{const res=await fetch(CONFIG[active].prefix+'api/progress?annotator='+encodeURIComponent(user),{cache:'no-store'}),x=await res.json();if(!res.ok||!x.ok)throw Error('Failed to load progress');if(version!==generation||key!==loadedKey)return;progress=x.groups;pageProgress=x.page_groups||{};loading=false;decorate()}catch{if(key===loadedKey){loading=false;document.getElementById('annotationSummary').textContent='Failed to read progress，Refresh and retry。'}}
 }
 renderGroups=function(){oldGroups();decorate()};render=function(){oldRender();refresh()};window.addEventListener('annotation-auth-change',()=>{loadedKey='';refresh()});window.addEventListener('focus',()=>{loadedKey='';refresh()});render();
})();
