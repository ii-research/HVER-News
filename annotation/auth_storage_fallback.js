// Limit fallback to account/session metadata; never delete annotation drafts.
(()=>{
 if(window.nevuAuthStorageFallbackInstalled)return;
 const authKeys=new Set(['nevu_annotation_accounts_v2','nevu_annotation_session_user_v2']);
 const proto=Storage.prototype,read=proto.getItem,write=proto.setItem,remove=proto.removeItem;
 const fallbackKey=k=>'nevu_auth_fallback_v1:'+k;
 window.nevuAuthTemporary=()=>[...authKeys].some(k=>read.call(sessionStorage,fallbackKey(k))!==null);
 proto.getItem=function(k){
  if(this===localStorage&&authKeys.has(String(k))){const value=read.call(sessionStorage,fallbackKey(k));if(value!==null)return value;}
  return read.call(this,k);
 };
 proto.setItem=function(k,v){
  if(this!==localStorage||!authKeys.has(String(k)))return write.call(this,k,v);
  try{write.call(this,k,v);remove.call(sessionStorage,fallbackKey(k));}
  catch(e){
   if(e.name!=='QuotaExceededError'&&e.name!=='NS_ERROR_DOM_QUOTA_REACHED')throw e;
   try{write.call(sessionStorage,fallbackKey(k),String(v));}
   catch(_){throw Error('Browser sign-in storage is also full。Export unsaved annotations first，Then sign in in a new browser window；Do not clear site data without a backup。');}
  }
 };
 proto.removeItem=function(k){
  if(this===localStorage&&authKeys.has(String(k)))remove.call(sessionStorage,fallbackKey(k));
  return remove.call(this,k);
 };
 window.nevuAuthStorage={
  getItem:k=>proto.getItem.call(localStorage,k),
  setItem:(k,v)=>proto.setItem.call(localStorage,k,v),
  removeItem:k=>proto.removeItem.call(localStorage,k)
 };
 window.nevuAuthStorageFallbackInstalled=true;
})();
