"""Share local annotation login across tabs without editing frozen group files."""

def enhance_group_auth(text: str) -> str:
    if 'const SESSION_KEY="nevu_annotation_session_user_v2"' not in text:
        return text
    for method in ('getItem', 'setItem', 'removeItem'):
        text = text.replace(f'sessionStorage.{method}(SESSION_KEY', f'localStorage.{method}(SESSION_KEY')
    # A logout triggered by another tab must never erase a newer user's login.
    text = text.replace('  localStorage.removeItem(SESSION_KEY);\n  annotator=null;',
                        '  if(localStorage.getItem(SESSION_KEY)===annotator)localStorage.removeItem(SESSION_KEY);\n  sessionStorage.removeItem(SESSION_KEY);\n  annotator=null;')
    listener = '''
let sharedAuthTask=Promise.resolve();
window.addEventListener("storage",event=>{
  if(event.key!==SESSION_KEY)return;
  // Block further edits immediately while the old account's work is saved.
  if(annotator!==localStorage.getItem(SESSION_KEY))showAuth("Switching sign-in state…");
  sharedAuthTask=sharedAuthTask.then(async()=>{
    const username=localStorage.getItem(SESSION_KEY);
    if(username===annotator)return;
    if(annotator)await logoutAccount();
    const current=localStorage.getItem(SESSION_KEY);
    if(current&&getAccounts()[current])await enterWorkspace(current);
    else showAuth("Signed out，Return to the dashboard to sign in。");
  }).catch(e=>{console.warn("shared login update failed",e);showAuth("Refresh to synchronize sign-in state。");});
});
'''
    from pathlib import Path
    helper = (Path(__file__).parent / 'auth_storage_fallback.js').read_text()
    text = text.replace('<script>', '<script>\n' + helper, 1)
    text = text.replace('\nbootAuth();', listener + '\nbootAuth();')
    for method in ('getItem', 'setItem', 'removeItem'):
        for key in ('SESSION_KEY', 'ACCOUNT_KEY', 'ACCOUNT_DB_KEY'):
            text = text.replace('localStorage.'+method+'('+key, 'window.nevuAuthStorage.'+method+'('+key)
    return text
