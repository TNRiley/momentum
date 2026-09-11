# Rebuilding Momentum

Enough to reproduce this project from an empty directory, a shell, and nothing else.

## 1. What is being built

A single self-contained page that tests six commentator claims about momentum in tennis against
every point of 11,452 charted professional matches, and then tests them again against the same
matches played as independent coin flips.

**The finding is that none of the six survives**, and the reason the page exists is the second
half of that sentence rather than the first. A plain comparison says servers hold **80.4%** of the
games they serve for the set against **75.8%** of their other service games. Controlling for the
player — comparing a player with himself inside a single set — turns that into **−14.8 per 100
games**. Neither number is the answer. Memoryless tennis, simulated with the same players and
formats and no memory at all, also produces **−14.8**, because *winning a game ends the set*: a
player who holds while serving for the set plays no more service games in it, and one who is
broken plays several more, so the outcome being measured decides how many baseline games exist.

The whole method is: run every test on real matches and on memoryless replicates of the same
sample, and report the difference.

## 2. Data

**The Tennis Abstract Match Charting Project**, `https://github.com/JeffSackmann/tennis_MatchChartingProject`
(branch `master`). Files needed, ~190 MB total:

```
charting-m-matches.csv      charting-w-matches.csv
charting-m-points-to-2009.csv   charting-m-points-2010s.csv   charting-m-points-2020s.csv
charting-w-points-to-2009.csv   charting-w-points-2010s.csv   charting-w-points-2020s.csv
```

**Licence: CC BY-NC-SA 4.0.** Attribution required, non-commercial only, share-alike. The repo's
README is emphatic about this. Ship **no point-level data** — this project redistributes only
aggregate counts, and those carry the same licence. `raw/` is gitignored.

Quirks that will bite:

- **Jeff Sackmann's other repos are gone.** `tennis_atp` and `tennis_slam_pointbypoint` 404 as of
  2026-09-10; his account has one public repo left, this one. Anything that wants the match archive
  (rankings, Elo, complete draws) has to do without. There are **no player rankings anywhere in this
  data** — player strength has to be estimated from the points themselves.
- **The points files have fewer columns than `data_dictionary.txt` describes.** Actual header:
  `match_id,Pt,Set1,Set2,Gm1,Gm2,Pts,Gm#,TbSet,Svr,1st,2nd,Notes,PtWinner`. There is no
  `isSvrWinner`, no `TB?`, no `rallyCount`. Games, tiebreaks, sets and break points must all be
  reconstructed from the score counters.
- **`Player 1` is always the player who served first**, so `Set1`/`Gm1` refer to that player.
- **About 2,200 matches arrive rotated.** Every point from 1..N is present but the file starts
  partway through and wraps (`…147,148,149,1,2,…`). Re-sort by `Pt`. Rejecting these instead —
  the obvious first move — throws away 19% of the sample for no reason.
- `Pts` is written server-first; a break point for the returner is `30-40`, `40-AD`, `0-40`, `15-40`.
- A tiebreak is a run of rows sharing one `(Set1,Set2,Gm1,Gm2)` in which the server alternates.
  That detection is more reliable than `TbSet`.

## 3. Pipeline

```
python 01_parse.py       # points files  -> raw/parsed.npz     (~3 min)
python 02_sanity.py      # known quantities and one famous match
python 04_compare.py 20  # six tests, real vs 20 memoryless replicates -> raw/compare.npz
python 05_robust.py 12   # score control, subgroups, dependence sweep  -> raw/robust.npz
python 06_payload.py     # everything the page needs           -> raw/payload.json
python 07_inject.py      # payload + template.html -> index.html, wrapped and catalog-linked
node test_sim.js         # the browser simulator must match the Python one
node test_page.js        # nothing may be drawn outside its canvas
```

`03_tests.py` is the first-pass exploration kept for the record; `04_compare.py` supersedes it.
`momentum_lib.py` holds the shared schema, the tests, the conditional statistic and the simulator,
so real and simulated data go through byte-identical code paths — which is the point.

### Decisions that are not obvious

- **Validation.** Every match's reconstructed game counters are checked against the file's own,
  set by set. A match that disagrees is dropped, not repaired. 11,547 of 11,646 survive; then
  95 more go for being retirements or part-charted (the reconstructed score never reaches a
  finished match), leaving **11,452**. Simulated matches are always complete, so comparing them
  with part-charted real ones would be an unfair fight.
- **The test.** Each claim is a 2×2 table repeated inside strata. Under the null the flag is
  exchangeable within its stratum, so flagged successes are hypergeometric with the margins fixed;
  summing over strata gives an exact conditional mean and variance (Mantel–Haenszel). Strata that
  are entirely flagged, entirely baseline, entirely success or entirely failure carry no
  information and drop out on their own — so the *effective* sample shrinks as the control tightens
  and must be reported.
- **Four control levels**: pooled, player, player×match, player×set. The hot hand gets three more
  that also hold the game score fixed.
- **Serve strength for the simulator** is each player's serve points won in that match, shrunk
  toward their career rate by τ = 30 points of prior. **τ is not a free parameter**: it is chosen
  by matching the real distribution of scorelines. τ=0 gives 24.7% deciding sets against 30.0%
  real; τ=∞ gives 41.3%; τ=30 gives 30.9%. Report the calibration table, it is evidence.
- **Intervals** combine the real sample's conditional standard error with the Monte-Carlo error of
  the control mean: `sqrt((100*sqrt(V)/n_flag)^2 + var(sim)/n_rep)`. Using only the spread of the
  simulated replicates is wrong and makes everything look significant — an early version did this
  and reported z = +7.8 for an effect that is z ≈ 0.7.

## 4. Verification table

Check these before trusting anything downstream.

| quantity | expected |
|---|---|
| matches parsed / kept | 11,547 of 11,646; 11,452 after completeness |
| games, points | 289,260 and 1,830,441 |
| hold rate, men / women | 79.9% / 66.1% |
| break points converted, men / women | 39.1% / 45.3% |
| tiebreaks per match | 0.42, median 11 points |
| points per ordinary game | 6.23 |
| distinct players / charters | 1,729 / 192 |
| 2019 Wimbledon final | reconstructs as 6-7, 6-1, 6-7, 6-4, 12-13 from Federer's side; his one serving-for-the-set game in set five (at 8-7) comes back **BROKEN**, 8 points, 1 break point faced |
| serving for the set, pooled / set control | +4.30 / −14.84 per 100 |
| the same, memoryless | +3.44 / −14.79 |
| hot hand, match control / match+score control | +0.28 ± 0.06 / **+0.04 ± 0.11** |
| upper bound on any dependence | 0.10 percentage points per point |

The 2019 Wimbledon final is the single best checkpoint: it is the most famous
serving-for-the-match game of the modern era, and a parser with an off-by-one in its game
boundaries will report it as held.

## 5. The page

900px measure, warm paper palette, Fraunces display with an italic accent, Archivo body, IBM Plex
Mono for figures, full light/dark. Eight sections: the six claims as selectable cards, the control
ladder, the simulator, the board of six results, the hot hand and the score control, the power
bound and its consequences, what gets charted, and method.

**The interaction that matters** is that the page ships the serve strengths of all 11,452 matches,
quantised to a byte each (~23 KB), and a JavaScript port of the simulator. The reader can replay
the entire sample in their browser and watch the memoryless result land on the real one. Do not cut
this: an argument that rests entirely on a control is worth more if the reader can run the control.

Two traps in the page:

- **Device pixel ratio.** Scaling the 2D context by `dpr` without also scaling the canvas *bitmap
  height* clips the bottom ~20% of every chart — axis labels first, so it looks like a styling
  choice rather than a bug. `test_page.js` runs every test/control combination through a recording
  context and fails if anything is drawn outside its canvas; it catches this and caught a highlight
  band that overflowed the plot at the last control level.
- **Correlated replicates.** Four simulated replicates shared across several tests make their
  deviations look systematic. When comparing the JS simulator with the Python one, give each
  replicate an independent seed, or a 1.5-sigma coincidence will read as a rules mismatch.

## 6. What the page must say about itself

- The sample is **matches volunteers chose to chart**: Grand Slams, finals, and stars. Roger
  Federer is in 714 of them. A momentum effect present only in matches nobody charts would not
  appear here.
- The score control removes any effect operating *through* the score rather than beyond it. If
  being ahead in a game is itself what lifts a player, that is real and this page sets it aside as
  not being a claim about sequence. Say so rather than claiming more.
- A null is a bound, not a proof. The bound is 0.10 percentage points per point, and the page
  states it.
- The simulator is very slightly *closer* than reality (30.9% deciders against 30.0%, 17.9%
  tiebreaks against 16.2%). That residual is real and is not hidden.
