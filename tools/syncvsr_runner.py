"""Loader script invoked from a tiny bootstrap cell on Kaggle.

The Kaggle Monaco/Jupyter editor mangles indentation on automated typing,
so the bootstrap cell has to be flat. This script does the heavy work:
fetch tools/export_syncvsr_to_onnx.ipynb from this repo's main branch,
walk every code cell, exec it in the bootstrap's global namespace, and
abort on the first failure with the full traceback.

From a Kaggle cell, paste:

    import urllib.request, os
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    exec(urllib.request.urlopen(
        "https://raw.githubusercontent.com/HereLiesAz/Liperty/main/tools/syncvsr_runner.py"
    ).read())
"""
import urllib.request
import json
import traceback

NB_URL = "https://raw.githubusercontent.com/HereLiesAz/Liperty/main/tools/export_syncvsr_to_onnx.ipynb"

print(f"[syncvsr_runner] Fetching {NB_URL}")
nb = json.loads(urllib.request.urlopen(NB_URL).read())

# globals() inside exec() refers to *this* module's globals (the runner's),
# which is empty. We want the cells to share state with the caller -- the
# Kaggle bootstrap cell -- so we mutate the caller's globals via the frame.
import sys
caller_globals = sys._getframe(1).f_globals if hasattr(sys, "_getframe") else globals()

for cell_index, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    bar = "=" * 70
    print(f"\n{bar}\n=== Cell {cell_index} ===\n{bar}")
    try:
        exec(compile(src, f"<cell_{cell_index}>", "exec"), caller_globals)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print(f"\n!!! Cell {cell_index} FAILED -- stopping !!!")
        raise

print("\n=== ALL CELLS COMPLETE ===")
