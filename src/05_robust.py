"""
Everything that has to hold before the one surviving effect can be believed.

  1. the score control: compare points played at the identical game score, so the effect
     cannot be the way servers play 40-0 rather than a real sequence effect
  2. subgroups: men, women, era, surface, best-of
  3. what it is worth: put a dependence of exactly this size back into simulated tennis
     and see what it changes about the matches

    python 05_robust.py [n_replicates]
"""
import os, sys, time
import numpy as np
import momentum_lib as ml

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
C = ml.C
TAU = 30

d = np.load(os.path.join(RAW, "parsed.npz"), allow_pickle=False)
G, P = d["games"].astype(np.int64), d["points"]
p1, p2, best_of, gender = d["p1"], d["p2"], d["best_of"], d["gender"]
date, surface = d["date"], d["surface"]

complete = ml.match_complete(G, best_of)
remap = -np.ones(len(complete), dtype=np.int64); remap[complete] = np.arange(complete.sum())
gsel = complete[G[:, C["match"]]]
gmap = -np.ones(len(G), dtype=np.int64); gmap[gsel] = np.arange(gsel.sum())
G = G[gsel]; G[:, C["match"]] = remap[G[:, C["match"]]]
P = P[gsel[P[:, 0]]]; P[:, 0] = gmap[P[:, 0]]
p1, p2, best_of, gender = p1[complete], p2[complete], best_of[complete], gender[complete]
date, surface = date[complete], surface[complete]
nm = len(p1)
year = np.array([int(x[:4]) if len(x) >= 4 and x[:4].isdigit() else 0 for x in date])

names = np.concatenate([p1, p2])
uniq, inv = np.unique(names, return_inverse=True)
pl_of_match = np.stack([inv[:nm], inv[nm:]], axis=1)
match = G[:, C["match"]]
srv_player = pl_of_match[match, G[:, C["srv"]] - 1]

pkey = match[P[:, 0]] * 2 + (P[:, 1] - 1)
served = np.bincount(pkey, minlength=nm * 2).astype(float).reshape(nm, 2)
won = np.bincount(pkey, weights=P[:, 2].astype(float), minlength=nm * 2).reshape(nm, 2)
pl_served = np.bincount(pl_of_match.ravel(), weights=served.ravel(), minlength=len(uniq))
pl_won = np.bincount(pl_of_match.ravel(), weights=won.ravel(), minlength=len(uniq))
grand = won.sum() / served.sum()
career = np.divide(pl_won, pl_served, out=np.full(len(uniq), grand), where=pl_served > 0)
p_serve = np.clip((won + TAU * career[pl_of_match]) / (served + TAU), 0.30, 0.88)

LV = ["pooled", "player", "match", "set", "score", "match+score", "set+score"]


def levels(pl, mt, st, state):
    pl = pl.astype(np.int64)
    return {"pooled": np.zeros(len(pl), dtype=np.int64),
            "player": pl,
            "match": pl * 1000003 + mt,
            "set": pl * 1000003 + st,
            "score": pl * 32 + state,
            "match+score": (pl * 1000003 + mt) * 32 + state,
            "set+score": (pl * 1000003 + st) * 32 + state}


def hh(G_, P_, srv_pl, subset=None, only=None):
    flag, ok, pl, mt, st, state = ml.hot_hand_trials(G_, P_, srv_pl, with_state=True)
    if subset is not None:
        m = subset[mt]
        flag, ok, pl, mt, st, state = flag[m], ok[m], pl[m], mt[m], st[m], state[m]
    lv = levels(pl, mt, st, state)
    keys = only or LV
    return {k: ml.conditional(lv[k], flag.astype(np.int64), ok.astype(float)) for k in keys}


GROUPS = [("men", gender == "M"), ("women", gender == "W"),
          ("best of 5", best_of == 5), ("best of 3", best_of == 3),
          ("hard", surface == "Hard"), ("clay", surface == "Clay"),
          ("grass", surface == "Grass"),
          ("to 2009", (year > 0) & (year <= 2009)),
          ("2010-2019", (year >= 2010) & (year <= 2019)),
          ("2020 on", year >= 2020)]
GROUPS = [(n, m) for n, m in GROUPS if m.sum() >= 100]

print("--- 1. the hot hand under the score control ---")
real = hh(G, P, srv_player)
real_sub = {n: hh(G, P, srv_player, subset=m, only=["match+score"])["match+score"]
            for n, m in GROUPS}

sims, sims_sub = [], {n: [] for n, _ in GROUPS}
for r in range(NREP):
    t0 = time.time()
    sG, sP = ml.simulate(p_serve, best_of, seed=2000 + r)
    sG = sG.astype(np.int64)
    s_srv = pl_of_match[sG[:, C["match"]], sG[:, C["srv"]] - 1]
    sims.append(hh(sG, sP, s_srv))
    for n, m in GROUPS:
        sims_sub[n].append(hh(sG, sP, s_srv, subset=m, only=["match+score"])["match+score"]["excess"])
    print("   replicate %d of %d (%.0fs)" % (r + 1, NREP, time.time() - t0), flush=True)

print()
print("%-14s %8s %11s %11s %9s %10s %9s"
      % ("control", "real", "memoryless", "difference", "+/- 95%", "trials", "strata"))
for lvl in LV:
    r = real[lvl]
    sv = np.array([s[lvl]["excess"] for s in sims])
    se = np.sqrt((100 * np.sqrt(r["V"]) / r["n_flag"]) ** 2 + sv.var(ddof=1) / len(sv))
    print("%-14s %+8.2f %+11.2f %+11.2f %9.2f %10d %9d"
          % (lvl, r["excess"], sv.mean(), r["excess"] - sv.mean(), 1.96 * se,
             r["n_flag"], r["n_strata"]))

print()
print("--- 2. subgroups, at the match+score control ---")
print("%-12s %8s %11s %11s %9s %10s" % ("group", "real", "memoryless", "difference",
                                        "+/- 95%", "trials"))
sub_out = []
for n, m in GROUPS:
    r = real_sub[n]
    sv = np.array(sims_sub[n])
    se = np.sqrt((100 * np.sqrt(r["V"]) / r["n_flag"]) ** 2 + sv.var(ddof=1) / len(sv))
    sub_out.append((n, r["excess"], sv.mean(), r["excess"] - sv.mean(), 1.96 * se, r["n_flag"]))
    print("%-12s %+8.2f %+11.2f %+11.2f %9.2f %10d"
          % (n, r["excess"], sv.mean(), r["excess"] - sv.mean(), 1.96 * se, r["n_flag"]))

print()
print("--- 3. what a dependence this size is worth ---")
target = real["match+score"]["excess"] - np.mean([s["match+score"]["excess"] for s in sims])
print("   target effect: %+.2f per 100 points" % target)


def match_stats(G_, best_of_, p_serve_):
    m_ = G_[:, C["match"]].astype(np.int64)
    n_ = int(m_.max()) + 1
    key = m_ * 16 + G_[:, C["set_no"]]
    side = np.where(G_[:, C["won_by_srv"]] == 1, G_[:, C["srv"]], G_[:, C["ret"]])
    a = np.bincount(key, weights=(side == 1).astype(float), minlength=n_ * 16).reshape(n_, 16)
    b = np.bincount(key, weights=(side == 2).astype(float), minlength=n_ * 16).reshape(n_, 16)
    played = (a + b) > 0
    setw = np.where(a > b, 1, 2)
    s1 = ((setw == 1) & played).sum(1)
    s2 = ((setw == 2) & played).sum(1)
    winner = np.where(s1 > s2, 1, 2)
    stronger = np.where(p_serve_[:, 0] >= p_serve_[:, 1], 1, 2)
    nsets = played[:, 1:6].sum(1)
    need = np.where(best_of_ == 3, 2, 3)
    return dict(upset=float((winner != stronger).mean()),
                decider=float((nsets == 2 * need - 1).mean()),
                pts_per_match=float(len(P) and 0),
                tb=float((G_[:, C["is_tb"]] == 1).sum() / n_))


for delta in (0.0, 0.004, 0.008, 0.015):
    ex, st = [], []
    for r in range(3):
        sG, sP = ml.simulate(p_serve, best_of, seed=5000 + r, delta=delta)
        sG = sG.astype(np.int64)
        s_srv = pl_of_match[sG[:, C["match"]], sG[:, C["srv"]] - 1]
        ex.append(hh(sG, sP, s_srv, only=["match+score"])["match+score"]["excess"])
        s = match_stats(sG, best_of, p_serve)
        s["pts"] = len(sP) / len(best_of)
        st.append(s)
    print("   delta %.3f -> measured %+.2f per 100 | upsets %.1f%% | deciders %.1f%% | "
          "points/match %.1f | tiebreaks/match %.2f"
          % (delta, np.mean(ex), 100 * np.mean([x["upset"] for x in st]),
             100 * np.mean([x["decider"] for x in st]), np.mean([x["pts"] for x in st]),
             np.mean([x["tb"] for x in st])))

np.savez_compressed(os.path.join(RAW, "robust.npz"),
                    levels=np.array(LV),
                    real=np.array([real[l]["excess"] for l in LV]),
                    real_V=np.array([real[l]["V"] for l in LV]),
                    nflag=np.array([real[l]["n_flag"] for l in LV]),
                    nstrata=np.array([real[l]["n_strata"] for l in LV]),
                    sim=np.array([[s[l]["excess"] for l in LV] for s in sims]),
                    sub_names=np.array([r[0] for r in sub_out]),
                    sub=np.array([[r[1], r[2], r[3], r[4], r[5]] for r in sub_out]))
print("\nwrote", os.path.join(RAW, "robust.npz"))
