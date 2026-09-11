# src — the pipeline

Run from this directory. Python 3.13 + numpy; Node only for the two tests.
`REBUILD.md` in the project root has the full recipe, the data quirks and the verification table.

## Get the data

`raw/` is gitignored — nothing in it is committed, because the Match Charting Project data is
CC BY-NC-SA 4.0 and this repo redistributes no point-level data.

```bash
cd ../raw
B=https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master
curl -s -O $B/charting-m-matches.csv -O $B/charting-w-matches.csv \
        -O $B/charting-m-points-to-2009.csv -O $B/charting-m-points-2010s.csv \
        -O $B/charting-m-points-2020s.csv  -O $B/charting-w-points-to-2009.csv \
        -O $B/charting-w-points-2010s.csv  -O $B/charting-w-points-2020s.csv
```

About 190 MB. The points files carry only the score before each point, the server and the point
winner — games, tiebreaks, sets and break points are all reconstructed in `01_parse.py`.

## Run

```bash
python 01_parse.py        # -> raw/parsed.npz          ~3 min
python 02_sanity.py       # known rates + the 2019 Wimbledon final as a checkpoint
python 04_compare.py 20   # six tests, real vs 20 memoryless replicates -> raw/compare.npz
python 05_robust.py 12    # score control, subgroups, dependence sweep  -> raw/robust.npz
python 06_payload.py      # -> raw/payload.json
python 07_inject.py       # -> ../index.html, wrapped for Pages and catalog-linked
node test_sim.js          # the JS simulator must agree with the Python one
node test_page.js         # nothing may be drawn outside its canvas
```

On Windows use `python`, not `python3`, and prefix anything that prints with
`PYTHONIOENCODING=utf-8`.

## Files

| file | what it is |
|---|---|
| `momentum_lib.py` | the schema, the six trials, the conditional test, and the memoryless simulator — shared so real and simulated data go through identical code |
| `01_parse.py` | points files → validated game and point tables; drops matches whose reconstructed score disagrees with the file's own counters |
| `02_sanity.py` | hold rates, break-point conversion, and one famous match checked by hand |
| `03_tests.py` | first-pass exploration, kept for the record; `04_compare.py` supersedes it |
| `04_compare.py` | the main run: every test, real against memoryless, with the simulator calibrated to the real scorelines |
| `05_robust.py` | the game-score control that kills the hot hand, subgroups, and the power sweep |
| `06_payload.py` | assembles `payload.json`, including the per-match serve strengths the browser simulator replays |
| `07_inject.py` | splices the payload into `template.html`, then wraps for Pages and adds the catalog link |
| `template.html` | the page, with a `__DATA__` placeholder |
| `test_sim.js` | checks the browser simulator reproduces the Python one within Monte-Carlo error |
| `test_page.js` | runs every test/control combination through a recording 2D context; fails if anything is drawn outside its canvas |
| `diag_pt.py` | one-off: diagnosed the 2,172 matches that fail a naive point-numbering check (they are rotated, not broken) |

`07_inject.py` runs `wrap_for_pages.py` and `add_catalog_link.py` as its last two steps.
Regenerating the page any other way silently drops the doctype and the breadcrumb.
