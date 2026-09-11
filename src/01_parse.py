"""
Parse the Match Charting Project point files into validated game- and point-level tables.

The points files carry only the pre-point score counters (Set1/Set2, Gm1/Gm2, Pts), the
server and the point winner. Games, tiebreaks, sets and break points are all reconstructed
here, and every match is checked for internal consistency before it is kept.

Player1 is always the player who served first, so Set1/Gm1 refer to that player throughout.

    python 01_parse.py            # reads ../raw, writes ../raw/parsed.npz
"""
import csv, os, collections
import numpy as np

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
csv.field_size_limit(10_000_000)


def read_matches():
    meta = {}
    for gender, fn in (("M", "charting-m-matches.csv"), ("W", "charting-w-matches.csv")):
        with open(os.path.join(RAW, fn), encoding="utf-8", errors="replace", newline="") as fh:
            for r in csv.DictReader(fh):
                mid = r["match_id"]
                if not mid or mid in meta:
                    continue
                try:
                    best_of = int(r.get("Best of") or 3)
                except ValueError:
                    best_of = 3
                meta[mid] = dict(
                    gender=gender, p1=r["Player 1"], p2=r["Player 2"],
                    date=(r.get("Date") or "").strip(), tourn=(r.get("Tournament") or "").strip(),
                    rnd=(r.get("Round") or "").strip(), surface=(r.get("Surface") or "").strip(),
                    best_of=best_of, final_tb=(r.get("Final TB?") or "").strip(),
                    charter=(r.get("Charted by") or "").strip(),
                )
    return meta


def point_rows():
    """Yield (match_id, [row, ...]) with the rows of one match, in file order."""
    files = [f for f in sorted(os.listdir(RAW)) if "-points-" in f and f.endswith(".csv")]
    for fn in files:
        with open(os.path.join(RAW, fn), encoding="utf-8", errors="replace", newline="") as fh:
            cur, rows = None, []
            for r in csv.DictReader(fh):
                mid = r["match_id"]
                if mid != cur:
                    if cur is not None:
                        yield cur, rows
                    cur, rows = mid, []
                rows.append(r)
            if cur is not None:
                yield cur, rows


def i(v, default=-1):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def is_break_point(pts):
    """Break point for the returner. Pts is written server-first."""
    return pts.strip() in ("30-40", "40-AD", "0-40", "15-40")


def parse_match(rows):
    """Return ((games, points), None) for one match, or (None, reason) if it fails validation."""
    # About 2,200 matches arrive rotated — every point from 1..N is present, but the file
    # starts partway through and wraps (…147,148,149,1,2,…). Re-sorting by Pt restores them;
    # the game-counter check at the bottom is the independent proof that it worked. Anything
    # whose point numbers are not exactly 1..N is a genuinely broken chart and is dropped.
    pts = [i(r["Pt"]) for r in rows]
    if pts != list(range(1, len(rows) + 1)):
        if sorted(pts) != list(range(1, len(rows) + 1)):
            return None, "point numbering"
        rows = sorted(rows, key=lambda r: i(r["Pt"]))

    runs, key = [], None
    for r in rows:
        k = (i(r["Set1"]), i(r["Set2"]), i(r["Gm1"]), i(r["Gm2"]))
        if any(x < 0 for x in k):
            return None, "bad score counter"
        if k != key:
            runs.append([k, []])
            key = k
        runs[-1][1].append(r)

    games, points = [], []
    prev = None                      # previous game in this same set
    set_no, s1, s2 = 1, 0, 0
    for (k, rr) in runs:
        S1, S2, G1, G2 = k
        if (S1, S2) != (s1, s2):
            if not ((S1 == s1 + 1 and S2 == s2) or (S2 == s2 + 1 and S1 == s1)):
                return None, "set counter jump"
            s1, s2 = S1, S2
            set_no = s1 + s2 + 1
            prev = None

        servers = {i(r["Svr"]) for r in rr}
        if servers - {1, 2}:
            return None, "bad server"
        is_tb = len(servers) == 2     # the server alternates inside a tiebreak
        winner = i(rr[-1]["PtWinner"])
        if winner not in (1, 2):
            return None, "bad point winner"

        if is_tb:
            srv = i(rr[0]["Svr"])
            bp = bps = 0
        else:
            srv = servers.pop()
            bp = sum(1 for r in rr if is_break_point(r["Pts"]))
            bps = bp if winner == srv else bp - 1     # the converted one is not a save
            if bps < 0:
                return None, "break point accounting"

        ret = 3 - srv
        srv_gm, ret_gm = (G1, G2) if srv == 1 else (G2, G1)

        for r in rr:
            ps = i(r["Svr"])
            points.append((len(games), ps, 1 if i(r["PtWinner"]) == ps else 0))

        games.append(dict(
            set_no=set_no, srv=srv, ret=ret, is_tb=int(is_tb), won_by_srv=int(winner == srv),
            srv_gm=srv_gm, ret_gm=ret_gm, n_pts=len(rr), bp=bp, bp_saved=bps,
            sfs=int(not is_tb and srv_gm >= 5 and srv_gm - ret_gm >= 1),
            stay=int(not is_tb and ret_gm >= 5 and ret_gm - srv_gm >= 1),
            after_own_break=int(prev is not None and not prev["is_tb"]
                                and prev["ret"] == srv and prev["won_by_srv"] == 0),
        ))
        prev = games[-1]

    # the reconstructed game counters must agree with the file's own, set by set
    per_set = collections.defaultdict(lambda: [0, 0])
    for g in games:
        c = per_set[g["set_no"]]
        exp = (c[0], c[1]) if g["srv"] == 1 else (c[1], c[0])
        if (g["srv_gm"], g["ret_gm"]) != exp:
            return None, "game counter mismatch"
        w = g["srv"] if g["won_by_srv"] else g["ret"]
        c[w - 1] += 1
    return (games, points), None


def main():
    meta = read_matches()
    G, P, M = [], [], []
    reasons = collections.Counter()
    for mid, rows in point_rows():
        m = meta.get(mid)
        if m is None:
            reasons["no metadata"] += 1
            continue
        out, why = parse_match(rows)
        if out is None:
            reasons[why] += 1
            continue
        games, points = out
        midx = len(M)
        M.append((mid, m))
        base = len(G)
        for g in games:
            g["match"] = midx
            G.append(g)
        for (gi, ps, sw) in points:
            P.append((base + gi, ps, sw))
        if len(M) % 2000 == 0:
            print("  %d matches parsed" % len(M), flush=True)

    print("kept %d matches, %d games, %d points" % (len(M), len(G), len(P)))
    print("dropped:", dict(reasons))

    cols = ["match", "set_no", "srv", "ret", "is_tb", "won_by_srv", "srv_gm", "ret_gm",
            "n_pts", "bp", "bp_saved", "sfs", "stay", "after_own_break"]
    games = np.array([[int(g[c]) for c in cols] for g in G], dtype=np.int32)
    points = np.array(P, dtype=np.int32)
    np.savez_compressed(
        os.path.join(RAW, "parsed.npz"),
        games=games, game_cols=np.array(cols), points=points,
        match_id=np.array([mid for mid, _ in M]),
        gender=np.array([m["gender"] for _, m in M]),
        p1=np.array([m["p1"] for _, m in M]), p2=np.array([m["p2"] for _, m in M]),
        date=np.array([m["date"] for _, m in M]), tourn=np.array([m["tourn"] for _, m in M]),
        rnd=np.array([m["rnd"] for _, m in M]), surface=np.array([m["surface"] for _, m in M]),
        best_of=np.array([m["best_of"] for _, m in M], dtype=np.int32),
        charter=np.array([m["charter"] for _, m in M]),
    )
    print("wrote", os.path.join(RAW, "parsed.npz"))


if __name__ == "__main__":
    main()
