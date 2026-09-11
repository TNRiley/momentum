"""
Assemble everything the page needs into one JSON payload.

Includes the serve strengths of all 11,452 matches, quantised to a byte each, so the page
can replay the memoryless control in the browser instead of asking the reader to take the
result on trust.

    python 06_payload.py
"""
import os, json, base64, collections
import numpy as np
import momentum_lib as ml

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")
C = ml.C
TAU, PMIN, PMAX = 30, 0.30, 0.88

d = np.load(os.path.join(RAW, "parsed.npz"), allow_pickle=False)
G, P = d["games"].astype(np.int64), d["points"]
p1, p2, best_of, gender = d["p1"], d["p2"], d["best_of"], d["gender"]
date, tourn, rnd, surface, charter = d["date"], d["tourn"], d["rnd"], d["surface"], d["charter"]

complete = ml.match_complete(G, best_of)
remap = -np.ones(len(complete), dtype=np.int64); remap[complete] = np.arange(complete.sum())
gsel = complete[G[:, C["match"]]]
gmap = -np.ones(len(G), dtype=np.int64); gmap[gsel] = np.arange(gsel.sum())
G = G[gsel]; G[:, C["match"]] = remap[G[:, C["match"]]]
P = P[gsel[P[:, 0]]]; P[:, 0] = gmap[P[:, 0]]
for a in ("p1", "p2", "best_of", "gender", "date", "tourn", "rnd", "surface", "charter"):
    globals()[a] = globals()[a][complete]
nm = len(p1)
year = np.array([int(x[:4]) if len(x) >= 4 and x[:4].isdigit() else 0 for x in date])

names = np.concatenate([p1, p2])
uniq, inv = np.unique(names, return_inverse=True)
pl_of_match = np.stack([inv[:nm], inv[nm:]], axis=1)
match = G[:, C["match"]]
srv_player = pl_of_match[match, G[:, C["srv"]] - 1]
ret_player = pl_of_match[match, G[:, C["ret"]] - 1]

pkey = match[P[:, 0]] * 2 + (P[:, 1] - 1)
served = np.bincount(pkey, minlength=nm * 2).astype(float).reshape(nm, 2)
won = np.bincount(pkey, weights=P[:, 2].astype(float), minlength=nm * 2).reshape(nm, 2)
pl_served = np.bincount(pl_of_match.ravel(), weights=served.ravel(), minlength=len(uniq))
pl_won = np.bincount(pl_of_match.ravel(), weights=won.ravel(), minlength=len(uniq))
grand = won.sum() / served.sum()
career = np.divide(pl_won, pl_served, out=np.full(len(uniq), grand), where=pl_served > 0)
p_serve = np.clip((won + TAU * career[pl_of_match]) / (served + TAU), PMIN, PMAX)

cmp_ = np.load(os.path.join(RAW, "compare.npz"), allow_pickle=False)
rob = np.load(os.path.join(RAW, "robust.npz"), allow_pickle=False)

CLAIMS = {
    "serve_for_set": ("Serving for the set",
                      "The hardest game to hold is the one where you can win the set.",
                      "Servers hold %(f).1f%% of the games they serve for the set, against "
                      "%(b).1f%% of their other service games."),
    "serve_to_stay": ("Serving to stay in the set",
                      "Serve to stay in the set and the pressure is all on you.",
                      "Servers hold %(f).1f%% of the games they serve to stay in the set, "
                      "against %(b).1f%% of their other service games."),
    "after_break":   ("The game after you break",
                      "The hardest game to hold is the one right after you break.",
                      "Players hold %(f).1f%% of the games that follow a break of their "
                      "opponent, against %(b).1f%% of the games that follow a failed one."),
    "wasted_bp":     ("Wasted break points",
                      "Missed chances come back to haunt you.",
                      "Returners break in %(f).1f%% of the games that follow a game where "
                      "they had break points and missed them, against %(b).1f%% of the games "
                      "that follow a game with no break point at all."),
    "hot_hand":      ("The hot hand",
                      "Win a point and you are more likely to win the next one.",
                      "Servers win %(f).1f%% of the points that follow a point they won, "
                      "against %(b).1f%% of the points that follow a point they lost."),
}

tr = ml.build_trials(G, srv_player, ret_player)
tr["hot_hand"] = ml.hot_hand_trials(G, P, srv_player)
raw = {}
for t, (flag, ok, pl, mt, st) in tr.items():
    f, o = flag.astype(bool), ok.astype(float)
    raw[t] = dict(flagRate=100 * o[f].mean(), baseRate=100 * o[~f].mean(),
                  nFlag=int(f.sum()), nBase=int((~f).sum()))

tests = []
for ti, t in enumerate(ml.TESTS):
    title, claim, fmt = CLAIMS[t]
    lv = []
    for li, lvl in enumerate(ml.LEVELS):
        real = float(cmp_["real"][ti, li])
        sims = cmp_["sim"][:, ti, li]
        nf = float(cmp_["nflag"][ti, li])
        se = np.sqrt((100 * np.sqrt(cmp_["real_V"][ti, li]) / nf) ** 2 + sims.var(ddof=1) / len(sims))
        lv.append(dict(level=lvl, real=real, sim=float(sims.mean()), diff=real - float(sims.mean()),
                       ci=float(1.96 * se), nflag=int(nf), nstrata=int(cmp_["nstrata"][ti, li])))
    tests.append(dict(key=t, title=title, claim=claim,
                      raw=fmt % {"f": raw[t]["flagRate"], "b": raw[t]["baseRate"]},
                      flagRate=raw[t]["flagRate"], baseRate=raw[t]["baseRate"],
                      nFlag=raw[t]["nFlag"], nBase=raw[t]["nBase"], levels=lv))

# the hot hand also gets the score controls
hot_levels = []
for li, lvl in enumerate(rob["levels"]):
    sims = rob["sim"][:, li]
    nf = float(rob["nflag"][li])
    se = np.sqrt((100 * np.sqrt(rob["real_V"][li]) / nf) ** 2 + sims.var(ddof=1) / len(sims))
    hot_levels.append(dict(level=str(lvl), real=float(rob["real"][li]), sim=float(sims.mean()),
                           diff=float(rob["real"][li] - sims.mean()), ci=float(1.96 * se),
                           nflag=int(nf), nstrata=int(rob["nstrata"][li])))

fs = ml.first_set_trials(G, P, best_of)
cuts = [int(c) for c in cmp_["cuts"]]
first_set = []
for ci_, c in enumerate(cuts):
    sims = cmp_["fs_sim"][:, ci_]
    n = int(cmp_["fs_real_n"][ci_])
    se = np.sqrt(0.25 / n + sims.var(ddof=1) / len(sims))
    first_set.append(dict(cut=c, real=100 * float(cmp_["fs_real_rate"][ci_]),
                          sim=100 * float(sims.mean()),
                          diff=100 * float(cmp_["fs_real_rate"][ci_] - sims.mean()),
                          ci=100 * float(1.96 * se), n=n))

subs = []
for i, n in enumerate(rob["sub_names"]):
    r = rob["sub"][i]
    subs.append(dict(name=str(n), real=float(r[0]), sim=float(r[1]), diff=float(r[2]),
                     ci=float(r[3]), nflag=int(r[4])))

# ---- sample shape
def counts(arr, top=None, as_int=False):
    u, c = np.unique(arr, return_counts=True)
    o = sorted(zip(u, c), key=lambda t: -t[1])
    if top:
        o = o[:top]
    return [[int(a) if as_int else str(a), int(b)] for a, b in o]

yr_u, yr_c = np.unique(year[year > 0], return_counts=True)
sample = dict(
    byYear=[[int(a), int(b)] for a, b in zip(yr_u, yr_c)],
    byRound=counts(rnd, 10), byTourn=counts(tourn, 12), bySurface=counts(surface),
    byGender=counts(gender), charters=int(len(np.unique(charter))),
    topPlayers=[[str(uniq[i]), int(c)] for i, c in
                sorted(collections.Counter(pl_of_match.ravel()).items(), key=lambda t: -t[1])[:14]],
    players=int(len(uniq)),
)

# ---- real match-level stats, for the consequence panel
ord_ = G[:, C["is_tb"]] == 0
key = match * 16 + G[:, C["set_no"]]
side = np.where(G[:, C["won_by_srv"]] == 1, G[:, C["srv"]], G[:, C["ret"]])
a = np.bincount(key, weights=(side == 1).astype(float), minlength=nm * 16).reshape(nm, 16)
b = np.bincount(key, weights=(side == 2).astype(float), minlength=nm * 16).reshape(nm, 16)
played = (a + b) > 0
setw = np.where(a > b, 1, 2)
s1 = ((setw == 1) & played).sum(1); s2 = ((setw == 2) & played).sum(1)
winner = np.where(s1 > s2, 1, 2)
stronger = np.where(p_serve[:, 0] >= p_serve[:, 1], 1, 2)
need = np.where(best_of == 3, 2, 3)
real_match = dict(upset=100 * float((winner != stronger).mean()),
                  decider=100 * float((played[:, 1:6].sum(1) == 2 * need - 1).mean()),
                  pts=float(len(P) / nm),
                  tb=float((G[:, C["is_tb"]] == 1).sum() / nm),
                  hold=100 * float(G[ord_, C["won_by_srv"]].mean()),
                  holdM=100 * float(G[ord_ & (gender[match] == "M"), C["won_by_srv"]].mean()),
                  holdW=100 * float(G[ord_ & (gender[match] == "W"), C["won_by_srv"]].mean()))

# ---- delta sweep, re-run here so the numbers travel with the payload
sweep = []
for delta in (0.0, 0.002, 0.004, 0.008, 0.015):
    ex, up, dec, pts, tbs = [], [], [], [], []
    for r in range(3):
        sG, sP = ml.simulate(p_serve, best_of, seed=5000 + r, delta=delta)
        sG = sG.astype(np.int64)
        s_srv = pl_of_match[sG[:, C["match"]], sG[:, C["srv"]] - 1]
        f, o, pl, mt, st, state = ml.hot_hand_trials(sG, sP, s_srv, with_state=True)
        strata = (pl.astype(np.int64) * 1000003 + mt) * 32 + state
        ex.append(ml.conditional(strata, f.astype(np.int64), o.astype(float))["excess"])
        m_ = sG[:, C["match"]]; k_ = m_ * 16 + sG[:, C["set_no"]]
        sd_ = np.where(sG[:, C["won_by_srv"]] == 1, sG[:, C["srv"]], sG[:, C["ret"]])
        aa = np.bincount(k_, weights=(sd_ == 1).astype(float), minlength=nm * 16).reshape(nm, 16)
        bb = np.bincount(k_, weights=(sd_ == 2).astype(float), minlength=nm * 16).reshape(nm, 16)
        pl_ = (aa + bb) > 0
        sw = np.where(aa > bb, 1, 2)
        w_ = np.where(((sw == 1) & pl_).sum(1) > ((sw == 2) & pl_).sum(1), 1, 2)
        up.append(100 * (w_ != stronger).mean())
        dec.append(100 * (pl_[:, 1:6].sum(1) == 2 * need - 1).mean())
        pts.append(len(sP) / nm)
        tbs.append((sG[:, C["is_tb"]] == 1).sum() / nm)
    sweep.append(dict(delta=delta, measured=float(np.mean(ex)), upset=float(np.mean(up)),
                      decider=float(np.mean(dec)), pts=float(np.mean(pts)), tb=float(np.mean(tbs))))
    print("   delta %.3f -> %+.2f per 100, upsets %.1f%%" % (delta, np.mean(ex), np.mean(up)))

base_ex = sweep[0]["measured"]
slope = np.mean([(s["measured"] - base_ex) / s["delta"] for s in sweep if s["delta"] > 0])
bound = hot_levels[[h["level"] for h in hot_levels].index("match+score")]["ci"] / slope

# ---- calibration table
calib = dict(real=dict(decider=30.0, tb=16.2, lopsided=13.7, hold=75.8, gps=9.69), rows=[])
for tau, dec_, tb_, lop, hold_, gps in [
        (0, 24.7, 16.0, 16.2, 75.5, 9.56), (30, 30.9, 17.9, 12.6, 76.0, 9.78),
        (60, 33.4, 17.8, 11.1, 76.2, 9.85), (120, 37.6, 18.8, 9.8, 76.4, 9.95),
        (250, 39.8, 19.4, 9.3, 76.6, 9.99), (10 ** 5, 41.3, 19.9, 8.6, 76.7, 10.04)]:
    calib["rows"].append(dict(tau=tau, decider=dec_, tb=tb_, lopsided=lop, hold=hold_, gps=gps))

# ---- the simulator's own inputs, for the browser
q = np.clip(np.round((p_serve - PMIN) / (PMAX - PMIN) * 255), 0, 255).astype(np.uint8)
sim_blob = dict(
    p=base64.b64encode(q.tobytes()).decode(), pmin=PMIN, pmax=PMAX,
    bo=base64.b64encode(best_of.astype(np.uint8).tobytes()).decode(),
    g=base64.b64encode((gender == "M").astype(np.uint8).tobytes()).decode(),
    pl=base64.b64encode(pl_of_match.astype(np.uint16).tobytes()).decode(),
    n=nm)

payload = dict(
    meta=dict(matches=nm, games=int(len(G)), points=int(len(P)), players=int(len(uniq)),
              charters=sample["charters"], tau=TAU,
              nrep=int(cmp_["sim"].shape[0]), nrepRobust=int(rob["sim"].shape[0]),
              dropped_incomplete=int((~complete).sum()),
              firstYear=int(year[year > 0].min()), lastYear=int(year.max()),
              generated="2026-09-10"),
    tests=tests, hotLevels=hot_levels, firstSet=first_set, subgroups=subs,
    sample=sample, realMatch=real_match, sweep=sweep, calib=calib,
    bound=dict(slope=float(slope), delta=float(bound), pp=float(bound * 100)),
    sim=sim_blob)

out = os.path.join(RAW, "payload.json")
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
print("wrote %s  (%.1f KB)" % (out, os.path.getsize(out) / 1024))
print("bound: a dependence bigger than %.4f per point (%.2f pp) is excluded" % (bound, bound * 100))
