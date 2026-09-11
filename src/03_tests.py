"""
Build the six momentum trials and evaluate each one at four levels of control.

Every test is a 2x2 table repeated inside strata: trials are flagged (the momentum
situation) or baseline, and each trial succeeds or fails. Under the null that the flag
carries no information, the flag is exchangeable within its stratum, so the count of
successes among flagged trials follows a hypergeometric with the stratum's margins fixed.
Summing over strata gives the exact conditional mean and variance (Mantel-Haenszel).

The four stratum levels are the argument of the whole page:
    pooled      every trial in one bucket - no control at all
    player      a player is compared with themselves across their whole career
    match       a player is compared with themselves inside one match
    set         a player is compared with themselves inside one set

Strata that are all-flagged or all-baseline, or all-success or all-failure, carry no
information and drop out of both the statistic and the variance on their own.

    python 03_tests.py            # reads ../raw/parsed.npz, writes ../raw/trials.npz
"""
import os
import numpy as np

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
d = np.load(os.path.join(RAW, "parsed.npz"), allow_pickle=False)
G = d["games"].astype(np.int64)
C = {c: k for k, c in enumerate(d["game_cols"])}
p1, p2, gender, date, best_of = d["p1"], d["p2"], d["gender"], d["date"], d["best_of"]
surface = d["surface"]

match = G[:, C["match"]]
year = np.array([int(x[:4]) if len(x) >= 4 and x[:4].isdigit() else 0 for x in date])

# ---------------------------------------------------------------- player identity
names = np.concatenate([p1, p2])
uniq, inv = np.unique(names, return_inverse=True)
pl_of_match = np.stack([inv[:len(p1)], inv[len(p1):]], axis=1)      # match -> (p1, p2) index
srv_player = pl_of_match[match, G[:, C["srv"]] - 1]
ret_player = pl_of_match[match, G[:, C["ret"]] - 1]
print("%d distinct players" % len(uniq))

ord_ = G[:, C["is_tb"]] == 0
set_id = match * 16 + G[:, C["set_no"]]          # unique per (match, set)

# game order within each set, and the previous game in the same set
order = np.lexsort((np.arange(len(G)), set_id))
assert (order == np.arange(len(G))).all(), "games are not in file order within sets"
prev_same_set = np.r_[-1, np.arange(len(G) - 1)]
prev_same_set[np.r_[True, set_id[1:] != set_id[:-1]]] = -1

# in an ordinary set the serve alternates, so the previous game is this server's return game
chk = (prev_same_set >= 0) & ord_ & (G[:, C["is_tb"]] == 0)
alt = G[prev_same_set[chk], C["srv"]] == G[chk, C["ret"]]
print("serve alternation holds in %.3f%% of consecutive game pairs" % (100 * alt.mean()))


# ---------------------------------------------------------------- the machinery
def conditional(strata, flag, ok):
    """Exact conditional mean/variance of successes among flagged trials, over strata."""
    _, s = np.unique(strata, return_inverse=True)
    n_s = np.bincount(s)                                  # trials in stratum
    k_s = np.bincount(s, weights=ok)                      # successes in stratum
    f_s = np.bincount(s, weights=flag)                    # flagged in stratum
    x = float((flag * ok).sum())
    N, K, n = n_s.astype(float), k_s.astype(float), f_s.astype(float)
    live = (N > 1) & (n > 0) & (n < N) & (K > 0) & (K < N)
    N, K, n = N[live], K[live], n[live]
    E = float((n * K / N).sum())
    V = float((n * K * (N - K) * (N - n) / (N * N * (N - 1))).sum())
    # restrict the observed count to the same informative strata
    keep = live[s]
    x = float((flag * ok)[keep].sum())
    nf = float(flag[keep].sum())
    return dict(x=x, E=E, V=V, n_flag=nf, n_trials=int(keep.sum()), n_strata=int(live.sum()),
                z=(x - E) / np.sqrt(V) if V > 0 else 0.0,
                excess=100 * (x - E) / nf if nf else 0.0)


def report(name, strata_levels, flag, ok, extra=""):
    print("\n=== %s ===%s" % (name, extra))
    print("   flagged %d of %d trials   raw: flagged %.1f%%  baseline %.1f%%  diff %+.2f pp"
          % (flag.sum(), len(flag), 100 * ok[flag == 1].mean(), 100 * ok[flag == 0].mean(),
             100 * (ok[flag == 1].mean() - ok[flag == 0].mean())))
    for lvl, strata in strata_levels:
        r = conditional(strata, flag, ok)
        print("   %-8s excess %+6.2f per 100   z = %+6.2f   (%d strata, %d flagged trials)"
              % (lvl, r["excess"], r["z"], r["n_strata"], r["n_flag"]))


def levels(pl, mt, st):
    return [("pooled", np.zeros(len(pl), dtype=np.int64)),
            ("player", pl),
            ("match", pl.astype(np.int64) * 100000 + mt),
            ("set", pl.astype(np.int64) * 1000000 + st)]


TRIALS = {}

def add(key, sel, flag, ok, pl, mt, st, title, claim):
    TRIALS[key] = dict(title=title, claim=claim, flag=flag.astype(np.int8), ok=ok.astype(np.int8),
                       player=pl.astype(np.int32), match=mt.astype(np.int32), set_=st.astype(np.int64),
                       gender=gender[mt], year=year[mt], best_of=best_of[mt], surface=surface[mt])
    report(title, levels(pl, mt, st), flag.astype(np.int64), ok.astype(np.float64))


# ---- T1 serving for the set -------------------------------------------------
sel = ord_ & (G[:, C["stay"]] == 0)          # exclude the mirror situation from the baseline
add("serve_for_set", sel,
    G[sel, C["sfs"]], G[sel, C["won_by_srv"]],
    srv_player[sel], match[sel], set_id[sel],
    "Serving for the set",
    "Closing out a set is the hardest hold there is.")

# ---- T2 serving to stay in the set -----------------------------------------
sel = ord_ & (G[:, C["sfs"]] == 0)
add("serve_to_stay", sel,
    G[sel, C["stay"]], G[sel, C["won_by_srv"]],
    srv_player[sel], match[sel], set_id[sel],
    "Serving to stay in the set",
    "Serve to stay in the set and the pressure is all on you.")

# ---- T3 the game after you break -------------------------------------------
# universe: ordinary service games with a previous game in the same set (always this
# server's return game). flagged = they just broke; baseline = they just failed to break.
sel = ord_ & (prev_same_set >= 0) & (G[prev_same_set, C["is_tb"]] == 0)
sel &= np.r_[False, (G[:-1, C["srv"]] == G[1:, C["ret"]])]      # serve really did alternate
add("after_break", sel,
    G[sel, C["after_own_break"]], G[sel, C["won_by_srv"]],
    srv_player[sel], match[sel], set_id[sel],
    "The game after you break",
    "The hardest game to hold is the one right after a break.")

# ---- T4 wasted break points ------------------------------------------------
# universe: return games whose previous game in the same set was a hold by the opponent.
# flagged = that hold contained at least one break point the returner did not convert.
prev = prev_same_set
sel = ord_ & (prev >= 0) & (G[prev, C["is_tb"]] == 0) & (G[prev, C["won_by_srv"]] == 1)
sel &= np.r_[False, (G[:-1, C["srv"]] == G[1:, C["ret"]])]
flag = (G[prev[sel], C["bp"]] > 0).astype(np.int8)
add("wasted_bp", sel,
    flag, 1 - G[sel, C["won_by_srv"]],                # outcome: the returner breaks now
    ret_player[sel], match[sel], set_id[sel],
    "Wasted break points",
    "Missed chances come back to haunt you.")

# ---- T6 winning the first set ----------------------------------------------
# one trial per match that reached a deciding set. outcome: the first-set winner wins it.
pts = d["points"]
gwin = G[:, C["won_by_srv"]]
set_pts = {}
first = {}
for mi in range(len(p1)):
    pass  # per-match work is done vectorised below

# points won by each side in each set
pt_game = pts[:, 0]
pt_srv = pts[:, 1]
pt_win_srv = pts[:, 2]
pt_winner = np.where(pt_win_srv == 1, pt_srv, 3 - pt_srv)        # 1 or 2
pt_set = G[pt_game, C["set_no"]]
pt_match = match[pt_game]
key = pt_match * 16 + pt_set
w1 = np.bincount(key, weights=(pt_winner == 1).astype(float), minlength=len(p1) * 16)
w2 = np.bincount(key, weights=(pt_winner == 2).astype(float), minlength=len(p1) * 16)

# sets won by each side, and who won each set
gw_side = np.where(gwin == 1, G[:, C["srv"]], G[:, C["ret"]])
gkey = match * 16 + G[:, C["set_no"]]
sg1 = np.bincount(gkey, weights=(gw_side == 1).astype(float), minlength=len(p1) * 16)
sg2 = np.bincount(gkey, weights=(gw_side == 2).astype(float), minlength=len(p1) * 16)
set_exists = (sg1 + sg2) > 0
set_winner = np.where(sg1 > sg2, 1, 2)

rows = []
for mi in range(len(p1)):
    base = mi * 16
    ns = [s for s in range(1, 6) if set_exists[base + s]]
    if not ns or ns != list(range(1, len(ns) + 1)):
        continue
    need = 2 if best_of[mi] == 3 else 3
    wins = [0, 0]
    dec = None
    for s in ns:
        w = set_winner[base + s]
        if max(wins) == need - 1 and min(wins) == need - 1:
            dec = s
            break
        wins[w - 1] += 1
    if dec is None:
        continue
    s1w = set_winner[base + 1]
    margin = (w1[base + 1] - w2[base + 1]) * (1 if s1w == 1 else -1)   # set-1 winner's points margin
    s1_games = (sg1[base + 1], sg2[base + 1])
    tb = int(max(s1_games) == 7 and min(s1_games) == 6)
    rows.append((mi, int(s1w == set_winner[base + dec]), int(margin), tb,
                 int(w1[base + 1] + w2[base + 1])))
rows = np.array(rows, dtype=np.int64)
print("\n=== Winning the first set ===")
print("   %d matches reached a deciding set" % len(rows))
for cut in (100, 12, 8, 6, 4, 2):
    m = np.abs(rows[:, 2]) <= cut
    if m.sum() < 30:
        continue
    r = rows[m, 1].mean()
    se = np.sqrt(0.25 / m.sum())
    print("   set-1 margin <= %3d pts: first-set winner takes the decider %.1f%% "
          "(n=%5d, z vs 50%% = %+5.2f)" % (cut, 100 * r, m.sum(), (r - 0.5) / se))
m = rows[:, 3] == 1
print("   set 1 was a tiebreak:     first-set winner takes the decider %.1f%% (n=%d, z = %+.2f)"
      % (100 * rows[m, 1].mean(), m.sum(), (rows[m, 1].mean() - 0.5) / np.sqrt(0.25 / m.sum())))

np.savez_compressed(os.path.join(RAW, "trials.npz"),
                    first_set=rows,
                    fs_gender=gender[rows[:, 0]], fs_year=year[rows[:, 0]],
                    fs_best_of=best_of[rows[:, 0]], fs_surface=surface[rows[:, 0]],
                    **{f"{k}__{f}": v for k, t in TRIALS.items() for f, v in t.items()
                       if f not in ("title", "claim")})
print("\nwrote", os.path.join(RAW, "trials.npz"))
