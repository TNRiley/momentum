"""Why do 2,172 matches fail the point-numbering check? Look at the actual sequences."""
import collections, importlib.util, os, sys

spec = importlib.util.spec_from_file_location("p", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_parse.py"))
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

kinds = collections.Counter()
examples = collections.defaultdict(list)
for mid, rows in p.point_rows():
    pts = [p.i(r["Pt"]) for r in rows]
    if pts == list(range(1, len(rows) + 1)):
        continue
    if pts[0] != 1:
        k = "starts at %s" % pts[0]
    elif sorted(pts) == list(range(1, len(rows) + 1)):
        k = "out of order"
    else:
        # where does it first break?
        b = next(j for j in range(len(pts)) if pts[j] != j + 1)
        dup = len(pts) != len(set(pts))
        k = ("duplicated rows" if dup else "gap") + " first at row %d" % b
        if dup and pts[:b] == list(range(1, b + 1)):
            # does the tail look like a second, complete chart of the same match?
            k = "restarts/duplicate chart" if pts[b] == 1 else "duplicated rows"
    kinds[k] += 1
    if len(examples[k]) < 2:
        examples[k].append((mid, pts[:6], pts[-4:], len(pts)))

for k, n in kinds.most_common(12):
    print("%6d  %s" % (n, k))
    for e in examples[k]:
        print("         ", e)
