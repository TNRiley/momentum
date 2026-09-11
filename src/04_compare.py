"""
Run the six tests on the real matches and on memoryless replicates of the same sample.

    python 04_compare.py [n_replicates]

Every simulated match has the same two players, the same best-of, and two serve strengths
shrunk toward each player's career rate - and no memory whatsoever. Anything a test reports
there is a property of the test, not of tennis.

The shrinkage constant is not a free parameter: it is chosen so that the simulator reproduces
the real distribution of scorelines (deciders, tiebreaks, lopsided sets), which is checked
and printed before any test is run.
"""
import os, sys, time
import numpy as np
import momentum_lib as ml

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 12
C = ml.C

d = np.load(os.path.join(RAW, "parsed.npz"), allow_pickle=False)
G, P = d["games"].astype(np.int64), d["points"]
p1, p2, best_of, gender = d["p1"], d["p2"], d["best_of"], d["gender"]
date, tourn, surface = d["date"], d["tourn"], d["surface"]

# ---- keep only matches charted through to a finished score
complete = ml.match_complete(G, best_of)
print("%d of %d matches are complete; %d dropped (retirements and part-charted matches)"
      % (complete.sum(), len(complete), (~complete).sum()))
remap = -np.ones(len(complete), dtype=np.int64)
remap[complete] = np.arange(complete.sum())
gsel = complete[G[:, C["match"]]]
gmap = -np.ones(len(G), dtype=np.int64)
gmap[gsel] = np.arange(gsel.sum())
G = G[gsel]
G[:, C["match"]] = remap[G[:, C["match"]]]
P = P[gsel[P[:, 0]]]
P[:, 0] = gmap[P[:, 0]]
p1, p2, best_of, gender = p1[complete], p2[complete], best_of[complete], gender[complete]
date, tourn, surface = date[complete], tourn[complete], surface[complete]
nm = len(p1)
print("   %d matches, %d games, %d points" % (nm, len(G), len(P)))

names = np.concatenate([p1, p2])
uniq, inv = np.unique(names, return_inverse=True)
pl_of_match = np.stack([inv[:nm], inv[nm:]], axis=1)
match = G[:, C["match"]]
srv_player = pl_of_match[match, G[:, C["srv"]] - 1]
ret_player = pl_of_match[match, G[:, C["ret"]] - 1]

# ---- serve strength: points won on serve, per player per match
pkey = match[P[:, 0]] * 2 + (P[:, 1] - 1)
served = np.bincount(pkey, minlength=nm * 2).astype(float).reshape(nm, 2)
won = np.bincount(pkey, weights=P[:, 2].astype(float), minlength=nm * 2).reshape(nm, 2)
pl_served = np.bincount(pl_of_match.ravel(), weights=served.ravel(), minlength=len(uniq))
pl_won = np.bincount(pl_of_match.ravel(), weights=won.ravel(), minlength=len(uniq))
grand = won.sum() / served.sum()
career = np.divide(pl_won, pl_served, out=np.full(len(uniq), grand), where=pl_served > 0)


def p_serve_for(tau):
    """Match serve rate shrunk toward the player's career rate by tau points of prior."""
    prior = career[pl_of_match]
    p = (won + tau * prior) / (served + tau)
    return np.clip(p, 0.30, 0.88)


# ---------------------------------------------------------------- calibration
def scoreline_stats(G_, best_of_):
    m_ = G_[:, C["match"]].astype(np.int64)
    n_ = int(m_.max()) + 1
    key = m_ * 16 + G_[:, C["set_no"]]
    side = np.where(G_[:, C["won_by_srv"]] == 1, G_[:, C["srv"]], G_[:, C["ret"]])
    a = np.bincount(key, weights=(side == 1).astype(float), minlength=n_ * 16).reshape(n_, 16)
    b = np.bincount(key, weights=(side == 2).astype(float), minlength=n_ * 16).reshape(n_, 16)
    played = (a + b) > 0
    nsets = played[:, 1:6].sum(1)
    need = np.where(best_of_ == 3, 2, 3)
    decider = nsets == (2 * need - 1)
    hi, lo = np.maximum(a, b)[played], np.minimum(a, b)[played]
    return dict(
        decider=float(decider.mean()),
        sets_per_match=float(nsets.mean()),
        tb_share=float((G_[:, C["is_tb"]] == 1).sum() / played.sum()),
        lopsided=float(((hi >= 6) & (lo <= 1)).mean()),
        hold=float(G_[G_[:, C["is_tb"]] == 0, C["won_by_srv"]].mean()),
        games_per_set=float(played.sum() and (hi + lo).mean()),
    )


real_stats = scoreline_stats(G, best_of)
print()
print("--- calibrating the simulator against the real scorelines ---")
print("%6s %9s %9s %9s %9s %9s" % ("tau", "decider", "tiebreak", "6-0/6-1", "hold", "gms/set"))
print("%6s %8.1f%% %8.1f%% %8.1f%% %8.1f%% %9.2f" % ("real", 100 * real_stats["decider"],
      100 * real_stats["tb_share"], 100 * real_stats["lopsided"], 100 * real_stats["hold"],
      real_stats["games_per_set"]))
best_tau, best_err = None, 1e9
for tau in (0, 30, 60, 120, 250, 500, 100000):
    sG, _ = ml.simulate(p_serve_for(tau), best_of, seed=7)
    s = scoreline_stats(sG.astype(np.int64), best_of)
    err = sum(abs(s[k] - real_stats[k]) / max(real_stats[k], 1e-9)
              for k in ("decider", "tb_share", "lopsided"))
    print("%6d %8.1f%% %8.1f%% %8.1f%% %8.1f%% %9.2f   err %.3f"
          % (tau, 100 * s["decider"], 100 * s["tb_share"], 100 * s["lopsided"],
             100 * s["hold"], s["games_per_set"], err))
    if err < best_err:
        best_tau, best_err = tau, err
TAU = best_tau
print("   chosen tau = %d points of prior" % TAU)
p_serve = p_serve_for(TAU)


# ---------------------------------------------------------------- the tests
def run_all(G_, P_, srv_pl, ret_pl, best_of_):
    tr = ml.build_trials(G_, srv_pl, ret_pl)
    tr["hot_hand"] = ml.hot_hand_trials(G_, P_, srv_pl)
    out = {}
    for t, (flag, ok, pl, mt, st) in tr.items():
        out[t] = {lvl: ml.conditional(ml.strata_for(lvl, pl, mt, st),
                                      flag.astype(np.int64), ok.astype(float))
                  for lvl in ml.LEVELS}
    fs = ml.first_set_trials(G_, P_, best_of_)
    out["first_set"] = {}
    for cut in CUTS:
        m = np.abs(fs[:, 2]) <= cut
        out["first_set"]["cut%d" % cut] = dict(
            rate=float(fs[m, 1].mean()) if m.sum() else np.nan, n=int(m.sum()))
    return out, fs


CUTS = (999, 16, 12, 10, 8, 6, 4, 2)
real, fs_real = run_all(G, P, srv_player, ret_player, best_of)

sims = []
for r in range(NREP):
    t0 = time.time()
    sG, sP = ml.simulate(p_serve, best_of, seed=1000 + r)
    sG = sG.astype(np.int64)
    s_srv = pl_of_match[sG[:, C["match"]], sG[:, C["srv"]] - 1]
    s_ret = pl_of_match[sG[:, C["match"]], sG[:, C["ret"]] - 1]
    sims.append(run_all(sG, sP, s_srv, s_ret, best_of))
    print("   replicate %2d of %d  (%.1fs)" % (r + 1, NREP, time.time() - t0), flush=True)

print()
print("=" * 100)
print("%-27s %-7s %8s %11s %11s %9s %8s"
      % ("test", "control", "real", "memoryless", "difference", "+/- 95%", "trials"))
print("=" * 100)
rows = []
for t in ml.TESTS:
    print(ml.TITLES[t])
    for lvl in ml.LEVELS:
        r = real[t][lvl]
        sv = np.array([s[0][t][lvl]["excess"] for s in sims])
        # the real statistic's own sampling error, plus Monte-Carlo error on the sim mean
        se_real = 100 * np.sqrt(r["V"]) / r["n_flag"] if r["n_flag"] else np.nan
        se = np.sqrt(se_real ** 2 + sv.var(ddof=1) / len(sv))
        diff = r["excess"] - sv.mean()
        rows.append((t, lvl, r["excess"], sv.mean(), diff, 1.96 * se, r["n_flag"]))
        print("%-27s %-7s %+8.2f %+11.2f %+11.2f %9.2f %8d"
              % ("", lvl, r["excess"], sv.mean(), diff, 1.96 * se, r["n_flag"]))
print("-" * 100)
print("%s  (share of deciding sets won by the first-set winner)" % ml.TITLES["first_set"])
for cut in CUTS:
    k = "cut%d" % cut
    r = real["first_set"][k]
    sv = np.array([s[0]["first_set"][k]["rate"] for s in sims])
    ns = np.mean([s[0]["first_set"][k]["n"] for s in sims])
    se = np.sqrt(0.25 / r["n"] + sv.var(ddof=1) / len(sv))
    print("%-27s %-7s %7.1f%% %10.1f%% %+11.2f %9.2f %8d"
          % ("", "<=%d" % cut if cut < 999 else "all", 100 * r["rate"], 100 * sv.mean(),
             100 * (r["rate"] - sv.mean()), 100 * 1.96 * se, r["n"]))

np.savez_compressed(
    os.path.join(RAW, "compare.npz"),
    tests=np.array(ml.TESTS), levels=np.array(ml.LEVELS), cuts=np.array(CUTS), tau=TAU,
    real=np.array([[real[t][l]["excess"] for l in ml.LEVELS] for t in ml.TESTS]),
    real_V=np.array([[real[t][l]["V"] for l in ml.LEVELS] for t in ml.TESTS]),
    nflag=np.array([[real[t][l]["n_flag"] for l in ml.LEVELS] for t in ml.TESTS]),
    nstrata=np.array([[real[t][l]["n_strata"] for l in ml.LEVELS] for t in ml.TESTS]),
    sim=np.array([[[s[0][t][l]["excess"] for l in ml.LEVELS] for t in ml.TESTS] for s in sims]),
    fs_real_rate=np.array([real["first_set"]["cut%d" % c]["rate"] for c in CUTS]),
    fs_real_n=np.array([real["first_set"]["cut%d" % c]["n"] for c in CUTS]),
    fs_sim=np.array([[s[0]["first_set"]["cut%d" % c]["rate"] for c in CUTS] for s in sims]),
    real_stats=np.array([real_stats[k] for k in sorted(real_stats)]),
    stat_names=np.array(sorted(real_stats)))
print("\nwrote", os.path.join(RAW, "compare.npz"))
