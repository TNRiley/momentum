# 🎾 Momentum

**Six things commentators say about momentum, tested on 11,452 charted matches — and tested again on tennis played by coins, which says the same thing.**

→ **[Open it](https://tnriley.github.io/momentum/)**

Every point of 11,452 professional matches from the Match Charting Project, rebuilt from the score counters up, and used to test the six things commentators say about momentum: serving for the set, serving to stay in it, the game after a break, wasted break points, the hot hand, and winning the first set. All six are true as plainly stated — servers do hold 80.4% of the games they serve for the set against 75.8% of their others. The page exists because a plain comparison cannot tell you anything here, and neither can a careful one: hold the player fixed and compare him with himself inside one set, and serving for the set swings from +4.3 per 100 games to −14.8. So every test is run again on the same matches played as independent coin flips, with the same players, the same formats and serve strengths calibrated to reproduce the real distribution of scorelines. Memoryless tennis produces −14.8 too, because winning a game ends the set and so decides how many baseline games exist. Nothing survives: the largest gap between real tennis and coins is within its own confidence interval. The hot hand comes closest, and dies on a scoring detail — two points played at 30–15 have identical win-loss counts and differ only in order, and compared that way the effect is +0.04 ± 0.11 per 100 points. You can run the control yourself: the page replays all 11,452 matches in your browser.

## Running it

One self-contained HTML file. No build step, no server, no network access at runtime — open `index.html` in a browser, or serve the directory with any static host.

```bash
python3 -m http.server 8000   # then visit http://localhost:8000
```

## Rebuilding it from scratch

[REBUILD.md](REBUILD.md) is written for an LLM with a shell and nothing else: the data sources and their quirks, the processing decisions, the page's structure and interactions, and a table of expected values to check the result against.

## Source

The full build pipeline is in [`src/`](src/), with a README describing how to regenerate the page from scratch.

## Data

- **[The Tennis Abstract Match Charting Project, crowdsourced shot-by-shot data (Jeff Sackmann and the 192 volunteer charters credited in these files)](https://github.com/JeffSackmann/tennis_MatchChartingProject)** — CC BY-NC-SA 4.0 — attribution, non-commercial, share-alike; no point-level data is redistributed here, and the derived aggregates on the page carry the same licence

Every figure on the page is computed from the data shipped with it. Check the page's own methods panel for how each number is derived and where it should not be pushed.

## Built with

python 3, numpy, memoryless match simulator, Mantel-Haenszel conditional test, vanilla JS, canvas, in-browser simulation, node canvas-geometry test.

## Licence

Code is MIT (see [LICENSE](LICENSE)). Data keeps the licence of its source, listed above.

---

Part of [Quick Projects](https://github.com/TNRiley/quick-projects) — one self-contained thing, built in one session. First published 2026-09-11.
