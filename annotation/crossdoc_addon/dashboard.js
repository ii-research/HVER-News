// Additive cross-document progress; legacy rendering remains unchanged.
(()=>{
 const oldRender=render,oldGroups=renderGroups;let progress={},loading=false,loadedKey='',generation=0;
 function decorate(){if(!active.startsWith('crossdoc_'))return;
  const rows=DATA[active],user=localStorage.getItem('nevu_annotation_session_user_v2'),sum=document.getElementById('annotationSummary');sum.hidden=false;
  sum.textContent=!user?'Sign in to view personal progress':loading?'Loading cross-document progress…':`Current account ${user} · Confirmed ${Object.values(progress).reduce((a,x)=>a+Number(x),0)} / ${rows.reduce((a,x)=>a+x.queries,0)} pairs（The two modules save independently）`;
  document.querySelectorAll('#groups a').forEach(link=>{const match=link.getAttribute('href').match(/group_(\d+)\.html/);if(!match)return;const n=Number(match[1]),r=rows.find(x=>x.group===n);if(!r)return;const meta=link.querySelector('.meta');if(meta)meta.textContent=`${r.queries} pairs query–Article`;if(user&&!loading){let extra=link.querySelector('.cross-progress');if(!extra){extra=document.createElement('div');extra.className='cross-progress small';link.append(extra)}extra.textContent=`${progress[n]||0}/${r.queries} Confirmed`;link.classList.toggle('confirmed',(progress[n]||0)>=r.queries)}});
 }
 async function refresh(){if(!active.startsWith('crossdoc_'))return;const user=localStorage.getItem('nevu_annotation_session_user_v2'),key=active+'|'+user;if(key===loadedKey){decorate();return}loadedKey=key;progress={};loading=!!user;decorate();if(!user)return;const version=++generation;
  try{const res=await fetch(CONFIG[active].prefix+'api/progress?annotator='+encodeURIComponent(user),{cache:'no-store'}),x=await res.json();if(!res.ok||!x.ok)throw Error('Failed to load progress');if(version!==generation||key!==loadedKey)return;progress=x.groups;loading=false;decorate()}catch{if(key===loadedKey){loading=false;document.getElementById('annotationSummary').textContent='Failed to read progress，Refresh and retry。'}}
 }
 renderGroups=function(){oldGroups();decorate()};render=function(){oldRender();refresh()};window.addEventListener('annotation-auth-change',()=>{loadedKey='';refresh()});window.addEventListener('focus',()=>{loadedKey='';refresh()});render();
})();
