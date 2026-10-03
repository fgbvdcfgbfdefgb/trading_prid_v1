The active Cerebrium deployment files now live at the repo root:
  - ../cerebrium.toml
  - ../main.py

They have to be at the project root because Cerebrium's default Cortex
runtime packages whatever directory contains cerebrium.toml (per its
`include`/`exclude` globs) and loads `main.py` from that same directory as
the endpoint module. Keeping them nested in cerebrium/ would have meant
training/, models/, simulator/, etc. never got uploaded with the deployment.
