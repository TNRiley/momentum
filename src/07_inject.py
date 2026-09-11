"""
Splice payload.json into template.html and finish the page.

Runs wrap_for_pages.py and add_catalog_link.py as its last two steps, because regenerating
the page without them silently drops the doctype and the breadcrumb back to the catalog.

    python 07_inject.py
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")
PROJ = os.path.join(HERE, "..")


def workspace_root(p):
    while True:
        if os.path.isdir(os.path.join(p, "projects")):
            return p
        nxt = os.path.dirname(p)
        if nxt == p:
            raise SystemExit("could not find the workspace root")
        p = nxt


with open(os.path.join(HERE, "template.html"), encoding="utf-8") as fh:
    tpl = fh.read()
with open(os.path.join(RAW, "payload.json"), encoding="utf-8") as fh:
    data = fh.read()

if "__DATA__" not in tpl:
    raise SystemExit("template has no __DATA__ placeholder")
out = tpl.replace("__DATA__", data)

dest = os.path.join(PROJ, "index.html")
with open(dest, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(out)
print("wrote %s  (%.1f KB)" % (dest, os.path.getsize(dest) / 1024))

root = workspace_root(HERE)
tools = os.path.join(root, "catalog", "tools")
for script in ("wrap_for_pages.py", "add_catalog_link.py"):
    r = subprocess.run([sys.executable, os.path.join(tools, script), dest],
                       capture_output=True, text=True)
    print("  %-24s %s" % (script, (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr).strip() else "ok"))
    if r.returncode:
        raise SystemExit(r.returncode)
