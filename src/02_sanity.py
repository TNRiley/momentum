"""Sanity checks on the parse. Known quantities first, then one famous match by hand."""
import os
import numpy as np

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
d = np.load(os.path.join(RAW, "parsed.npz"), allow_pickle=False)
G = d["games"]
C = {c: k for k, c in enumerate(d["game_cols"])}
gender, tourn, rnd, date, p1, p2 = d["gender"], d["tourn"], d["rnd"], d["date"], d["p1"], d["p2"]
best_of = d["best_of"]

mg = gender[G[:, C["match"]]]
ord_ = G[:, C["is_tb"]] == 0          # ordinary (non-tiebreak) games

print("matches %d   games %d   points %d" % (len(gender), len(G), len(d["points"])))
print("  men %d  women %d" % ((gender == "M").sum(), (gender == "W").sum()))
print()
print("--- hold rate (ordinary service games) ---")
for g in ("M", "W"):
    m = ord_ & (mg == g)
    print("  %s  %.1f%%  of %d games" % (g, 100 * G[m, C["won_by_srv"]].mean(), m.sum()))

print()
print("--- break points ---")
for g in ("M", "W"):
    m = ord_ & (mg == g)
    bp, sv = G[m, C["bp"]].sum(), G[m, C["bp_saved"]].sum()
    print("  %s  %d faced, %d saved -> %.1f%% converted by the returner"
          % (g, bp, sv, 100 * (bp - sv) / bp))
    print("     games with >=1 bp: %.1f%%   deuce-ish games (>=7 pts): %.1f%%"
          % (100 * (G[m, C["bp"]] > 0).mean(), 100 * (G[m, C["n_pts"]] >= 7).mean()))

print()
print("--- tiebreaks ---")
tb = G[:, C["is_tb"]] == 1
print("  %d tiebreaks, %.2f per match, median %d points"
      % (tb.sum(), tb.sum() / len(gender), np.median(G[tb, C["n_pts"]])))
print("  points per ordinary game: mean %.2f  (expect ~6.5)" % G[ord_, C["n_pts"]].mean())

print()
print("--- pressure-game universe sizes ---")
for lbl in ("sfs", "stay", "after_own_break"):
    m = ord_ & (G[:, C[lbl]] == 1)
    print("  %-16s %7d games   server holds %.1f%%" % (lbl, m.sum(), 100 * G[m, C["won_by_srv"]].mean()))

print()
print("--- sample shape: what gets charted ---")
yr = np.array([int(x[:4]) if len(x) >= 4 and x[:4].isdigit() else 0 for x in date])
for lo, hi in ((1900, 1999), (2000, 2009), (2010, 2014), (2015, 2019), (2020, 2026)):
    m = (yr >= lo) & (yr <= hi)
    print("  %d-%d  %5d matches" % (lo, hi, m.sum()))
u, c = np.unique(tourn, return_counts=True)
print("  top tournaments:", ", ".join("%s %d" % (a, b) for a, b in
      sorted(zip(u, c), key=lambda t: -t[1])[:8]))
u, c = np.unique(rnd, return_counts=True)
print("  top rounds:", ", ".join("%s %d" % (a, b) for a, b in
      sorted(zip(u, c), key=lambda t: -t[1])[:8]))

print()
print("--- checkpoint: 2019 Wimbledon final, Djokovic d. Federer 7-6 1-6 7-6 4-6 13-12 ---")
print("    (Federer served for the match at 8-7 in the fifth, 40-15, and was broken)")
idx = np.where((date == "20190714") & (tourn == "Wimbledon"))[0]
for mi in idx:
    print("   ", d["match_id"][mi])
    g = G[G[:, C["match"]] == mi]
    sets = {}
    for row in g:
        s = row[C["set_no"]]
        w = row[C["srv"]] if row[C["won_by_srv"]] else row[C["ret"]]
        sets.setdefault(s, [0, 0])[w - 1] += 1
    print("     set scores by player1/player2:", {k: tuple(v) for k, v in sorted(sets.items())})
    fifth = g[(g[:, C["set_no"]] == 5) & (g[:, C["is_tb"]] == 0)]
    sfs = fifth[fifth[:, C["sfs"]] == 1]
    print("     serving-for-the-set games in set 5: %d, server held in %d"
          % (len(sfs), sfs[:, C["won_by_srv"]].sum()))
    for row in sfs:
        print("       at %d-%d (server=p%d): %s, %d points, %d bp faced"
              % (row[C["srv_gm"]], row[C["ret_gm"]], row[C["srv"]],
                 "HELD" if row[C["won_by_srv"]] else "BROKEN", row[C["n_pts"]], row[C["bp"]]))
