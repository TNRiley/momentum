"""
Shared machinery: the games schema, the six trials, the conditional test, and a simulator
that plays memoryless tennis into exactly the same schema.

The simulator is the control for the whole page. A test that reports an effect on matches
where every point is an independent coin flip is reporting an artifact of its own design,
and three of the six do.
"""
import numpy as np

COLS = ["match", "set_no", "srv", "ret", "is_tb", "won_by_srv", "srv_gm", "ret_gm",
        "n_pts", "bp", "bp_saved", "sfs", "stay", "after_own_break"]
C = {c: k for k, c in enumerate(COLS)}

LEVELS = ["pooled", "player", "match", "set"]

TESTS = ["serve_for_set", "serve_to_stay", "after_break", "wasted_bp", "hot_hand"]

TITLES = {
    "serve_for_set": "Serving for the set",
    "serve_to_stay": "Serving to stay in the set",
    "after_break":   "The game after you break",
    "wasted_bp":     "Wasted break points",
    "hot_hand":      "The hot hand",
    "first_set":     "Winning the first set",
}


# ------------------------------------------------------------------ the test
def conditional(strata, flag, ok):
    """Successes among flagged trials, against the exact conditional null over strata.

    Within a stratum the flag is exchangeable under the null, so the flagged successes are
    hypergeometric with the stratum margins fixed. Strata that are entirely flagged, entirely
    baseline, entirely success or entirely failure carry no information and drop out.
    """
    _, s = np.unique(strata, return_inverse=True)
    N = np.bincount(s).astype(float)
    K = np.bincount(s, weights=ok).astype(float)
    n = np.bincount(s, weights=flag).astype(float)
    live = (N > 1) & (n > 0) & (n < N) & (K > 0) & (K < N)
    keep = live[s]
    Nl, Kl, nl = N[live], K[live], n[live]
    E = float((nl * Kl / Nl).sum())
    V = float((nl * Kl * (Nl - Kl) * (Nl - nl) / (Nl * Nl * (Nl - 1))).sum())
    x = float((flag * ok)[keep].sum())
    nf = float(flag[keep].sum())
    return dict(x=x, E=E, V=V, n_flag=nf, n_strata=int(live.sum()),
                excess=(100.0 * (x - E) / nf) if nf else 0.0,
                z=((x - E) / np.sqrt(V)) if V > 0 else 0.0)


def strata_for(level, player, match, set_id):
    if level == "pooled":
        return np.zeros(len(player), dtype=np.int64)
    if level == "player":
        return player.astype(np.int64)
    if level == "match":
        return player.astype(np.int64) * 1000003 + match
    if level == "set":
        return player.astype(np.int64) * 1000003 + set_id
    raise ValueError(level)


# ------------------------------------------------------- building the trials
def build_trials(G, srv_player, ret_player):
    """Return {test: (flag, ok, player, match, set_id)} from a games array."""
    match = G[:, C["match"]].astype(np.int64)
    set_id = match * 16 + G[:, C["set_no"]]
    ord_ = G[:, C["is_tb"]] == 0

    prev = np.r_[-1, np.arange(len(G) - 1)]
    prev[np.r_[True, set_id[1:] != set_id[:-1]]] = -1
    alternates = np.r_[False, (G[:-1, C["srv"]] == G[1:, C["ret"]])]

    out = {}

    sel = ord_ & (G[:, C["stay"]] == 0)
    out["serve_for_set"] = (G[sel, C["sfs"]], G[sel, C["won_by_srv"]],
                            srv_player[sel], match[sel], set_id[sel])

    sel = ord_ & (G[:, C["sfs"]] == 0)
    out["serve_to_stay"] = (G[sel, C["stay"]], G[sel, C["won_by_srv"]],
                            srv_player[sel], match[sel], set_id[sel])

    # the previous game in the set is always this server's return game, so the baseline is
    # "you just failed to break" and the flag is "you just broke"
    sel = ord_ & (prev >= 0) & alternates & (G[prev, C["is_tb"]] == 0)
    out["after_break"] = (G[sel, C["after_own_break"]], G[sel, C["won_by_srv"]],
                          srv_player[sel], match[sel], set_id[sel])

    # return games following a game the returner failed to break; flagged if they had a
    # break point in it. outcome: do they break now
    sel = ord_ & (prev >= 0) & alternates & (G[prev, C["is_tb"]] == 0) & (G[prev, C["won_by_srv"]] == 1)
    out["wasted_bp"] = ((G[prev[sel], C["bp"]] > 0).astype(np.int64), 1 - G[sel, C["won_by_srv"]],
                        ret_player[sel], match[sel], set_id[sel])
    return out


def game_score_state(g, won):
    """Index the score before each point of a game: 0-15 for the ordinary scores, then
    deuce, advantage server, advantage returner. Points are in order within each game."""
    a = b = 0
    out = np.empty(len(g), dtype=np.int64)
    prev = -1
    for k in range(len(g)):
        if g[k] != prev:
            a = b = 0
            prev = g[k]
        if a >= 3 and b >= 3:
            out[k] = 16 if a == b else (17 if a > b else 18)
        else:
            out[k] = min(a, 3) * 4 + min(b, 3)
        if won[k]:
            a += 1
        else:
            b += 1
    return out


def hot_hand_trials(G, P, srv_player, with_state=False):
    """Point-level trials: flagged = the server won the previous point of the same game.

    With with_state, also return the score before the point. Two points played at the same
    score can have arrived there by different paths - 30-15 is reached both by winning the
    last point (from 15-15) and by losing it (from 30-0) - so holding the score fixed
    separates a real sequence effect from the way servers play particular scores.
    """
    ord_games = G[:, C["is_tb"]] == 0
    g = P[:, 0]
    keep = ord_games[g]
    g, won = g[keep], P[keep, 2]
    first_of_game = np.r_[True, g[1:] != g[:-1]]
    flag = np.r_[0, won[:-1]]                    # previous point in the same game
    sel = ~first_of_game
    match = G[:, C["match"]].astype(np.int64)
    set_id = match * 16 + G[:, C["set_no"]]
    base = (flag[sel], won[sel], srv_player[g[sel]], match[g[sel]], set_id[g[sel]])
    if not with_state:
        return base
    return base + (game_score_state(g, won)[sel],)


def first_set_trials(G, P, best_of):
    """One trial per match that reached a deciding set: did the first-set winner win it,
    and how many points separated the players in the first set."""
    match = G[:, C["match"]].astype(np.int64)
    nm = int(match.max()) + 1 if len(match) else 0
    key = match * 16 + G[:, C["set_no"]]
    gw_side = np.where(G[:, C["won_by_srv"]] == 1, G[:, C["srv"]], G[:, C["ret"]])
    sg1 = np.bincount(key, weights=(gw_side == 1).astype(float), minlength=nm * 16)
    sg2 = np.bincount(key, weights=(gw_side == 2).astype(float), minlength=nm * 16)

    pt_srv = P[:, 1]
    pt_winner = np.where(P[:, 2] == 1, pt_srv, 3 - pt_srv)
    pkey = key[P[:, 0]]
    w1 = np.bincount(pkey, weights=(pt_winner == 1).astype(float), minlength=nm * 16)
    w2 = np.bincount(pkey, weights=(pt_winner == 2).astype(float), minlength=nm * 16)

    exists = (sg1 + sg2) > 0
    winner = np.where(sg1 > sg2, 1, 2)

    rows = []
    for mi in range(nm):
        b = mi * 16
        ns = [s for s in range(1, 6) if exists[b + s]]
        if not ns or ns != list(range(1, len(ns) + 1)):
            continue
        need = 2 if best_of[mi] == 3 else 3
        wins, dec = [0, 0], None
        for s in ns:
            if wins[0] == need - 1 and wins[1] == need - 1:
                dec = s
                break
            wins[winner[b + s] - 1] += 1
        if dec is None:
            continue
        s1w = winner[b + 1]
        margin = int((w1[b + 1] - w2[b + 1]) * (1 if s1w == 1 else -1))
        rows.append((mi, int(s1w == winner[b + dec]), margin,
                     int(max(sg1[b + 1], sg2[b + 1]) == 7 and min(sg1[b + 1], sg2[b + 1]) == 6)))
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def match_complete(G, best_of):
    """True for matches whose reconstructed sets add up to a finished match."""
    match = G[:, C["match"]].astype(np.int64)
    nm = int(match.max()) + 1 if len(match) else 0
    key = match * 16 + G[:, C["set_no"]]
    gw_side = np.where(G[:, C["won_by_srv"]] == 1, G[:, C["srv"]], G[:, C["ret"]])
    sg1 = np.bincount(key, weights=(gw_side == 1).astype(float), minlength=nm * 16)
    sg2 = np.bincount(key, weights=(gw_side == 2).astype(float), minlength=nm * 16)
    exists = ((sg1 + sg2) > 0).reshape(nm, 16)
    winner = np.where(sg1 > sg2, 1, 2).reshape(nm, 16)
    ok = np.zeros(nm, dtype=bool)
    for mi in range(nm):
        ns = [s for s in range(1, 6) if exists[mi, s]]
        if not ns or ns != list(range(1, len(ns) + 1)):
            continue
        need = 2 if best_of[mi] == 3 else 3
        w = [0, 0]
        for s in ns:
            w[winner[mi, s] - 1] += 1
        ok[mi] = max(w) == need and min(w) < need
    return ok


# ---------------------------------------------------------------- simulator
def sim_match(mi, ps1, ps2, best_of, rng, games, points, delta=0.0):
    """Play one match and emit rows in the same schema as the parse.

    With delta = 0 every point is an independent coin flip whose bias depends only on who is
    serving. With delta > 0 the server's chance rises by delta after a point they won and
    falls by delta*p/(1-p) after one they lost, which leaves the long-run rate alone and puts
    in a first-order dependence of a known size - the knob used to ask what the one surviving
    effect is actually worth.
    """
    need = 2 if best_of == 3 else 3
    sets = [0, 0]
    server = 1
    set_no = 0
    R = rng.random
    while max(sets) < need:
        set_no += 1
        gm = [0, 0]
        prev_break_by = 0        # who won the previous game of this set as returner
        while True:
            srv, ret = server, 3 - server
            p = ps1 if srv == 1 else ps2
            srv_gm, ret_gm = gm[srv - 1], gm[ret - 1]
            tb = (gm[0] == 6 and gm[1] == 6)
            gi = len(games)
            if tb:
                a = b = 0                        # a = points for the player serving first
                k = 0
                while True:
                    # 1, then 2 at a time: server is srv for point 0, then alternates in pairs
                    s = srv if (k == 0 or ((k - 1) // 2) % 2 == 1) else ret
                    pp = ps1 if s == 1 else ps2
                    w = 1 if R() < pp else 0     # 1 = the server of this point won it
                    points.append((gi, s, w))
                    winner_side = s if w else 3 - s
                    if winner_side == srv:
                        a += 1
                    else:
                        b += 1
                    k += 1
                    if max(a, b) >= 7 and abs(a - b) >= 2:
                        break
                won_by_srv = int(a > b)
                npts, bp, bps = k, 0, 0
            else:
                a = b = 0                         # server, returner points
                bp = 0
                last = None
                down = delta * p / (1.0 - p) if delta else 0.0
                while True:
                    if b >= 3 and b > a:
                        bp += 1
                    pe = p if last is None else (p + delta if last else p - down)
                    w = 1 if R() < pe else 0
                    last = w
                    points.append((gi, srv, w))
                    a, b = (a + 1, b) if w else (a, b + 1)
                    if max(a, b) >= 4 and abs(a - b) >= 2:
                        break
                won_by_srv = int(a > b)
                npts = a + b
                bps = bp if won_by_srv else bp - 1
            games.append((mi, set_no, srv, ret, int(tb), won_by_srv, srv_gm, ret_gm, npts,
                          bp, max(bps, 0),
                          int(not tb and srv_gm >= 5 and srv_gm - ret_gm >= 1),
                          int(not tb and ret_gm >= 5 and ret_gm - srv_gm >= 1),
                          int(prev_break_by == srv)))
            gw = srv if won_by_srv else ret
            prev_break_by = ret if not won_by_srv else 0
            gm[gw - 1] += 1
            server = ret
            if (max(gm) >= 6 and abs(gm[0] - gm[1]) >= 2) or max(gm) == 7:
                break
        sets[(0 if gm[0] > gm[1] else 1)] += 1


def simulate(p_serve, best_of, seed=0, delta=0.0):
    """Simulate one replicate of the whole sample; delta=0 is memoryless."""
    import random
    py = random.Random(seed)
    games, points = [], []
    for mi in range(len(best_of)):
        sim_match(mi, float(p_serve[mi, 0]), float(p_serve[mi, 1]), int(best_of[mi]), py,
                  games, points, delta)
    return (np.array(games, dtype=np.int32).reshape(-1, len(COLS)),
            np.array(points, dtype=np.int32).reshape(-1, 3))
