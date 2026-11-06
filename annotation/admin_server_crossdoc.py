"""Add crossdoc monitoring while retaining the original admin server and auth."""
import admin_server
from crossdoc_addon.admin_extension.crossdoc_admin import install
if __name__=='__main__':
 install(admin_server)
 admin_server.main()
