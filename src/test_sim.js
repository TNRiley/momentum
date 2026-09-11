// Check the browser simulator against the Python one: it must reproduce the memoryless
// numbers in the payload to within Monte-Carlo error.
//   node test_sim.js
const fs = require("fs");
const path = require("path");

globalThis.matchMedia = () => ({matches: false});
globalThis.document = {getElementById: () => ({})};
globalThis.devicePixelRatio = 1;

const tpl = fs.readFileSync(path.join(__dirname, "template.html"), "utf8");
const js = tpl.slice(tpl.indexOf("const D = __DATA__"), tpl.indexOf("/* ─── state"));
const payload = fs.readFileSync(path.join(__dirname, "..", "raw", "payload.json"), "utf8");
const mod = new Function(js.replace("__DATA__", payload) + "\nreturn {runReplicate, D, NM};");
const {runReplicate, D, NM} = mod();

console.log(`${NM} matches loaded`);
const CASES = [
  ["serve_for_set", "pooled"], ["serve_for_set", "match"], ["serve_for_set", "set"],
  ["serve_to_stay", "set"], ["after_break", "match"], ["wasted_bp", "match"],
  ["hot_hand", "match"], ["hot_hand", "match+score"],
];
const REPS = 4;
let worst = 0;
console.log("test              control        python      browser        diff");
for (const [t, lvl] of CASES) {
  const vals = [];
  for (let r = 0; r < REPS; r++) vals.push(runReplicate(t, lvl, 4242 + r, 0));
  const mine = vals.reduce((a, b) => a + b, 0) / vals.length;
  const rows = t === "hot_hand" && lvl.includes("score")
    ? D.hotLevels : (D.tests.find(x => x.key === t).levels.find(l => l.level === lvl)
        ? D.tests.find(x => x.key === t).levels : D.hotLevels);
  const py = rows.find(l => l.level === lvl).sim;
  const d = mine - py;
  worst = Math.max(worst, Math.abs(d));
  console.log(`${t.padEnd(16)}  ${lvl.padEnd(12)}  ${py.toFixed(3).padStart(8)}  ${mine.toFixed(3).padStart(10)}  ${d.toFixed(3).padStart(10)}`);
}
console.log(`\nlargest discrepancy ${worst.toFixed(3)} per 100`);

// the first-set statistic too
const fsPy = D.firstSet.find(f => f.cut > 900);
const fsMine = [0, 1, 2, 3].map(r => runReplicate("first_set", "999", 777 + r, 0));
const m = fsMine.reduce((a, b) => a + b, 0) / fsMine.length;
console.log(`first_set (all)   python ${fsPy.sim.toFixed(2)}%   browser ${m.toFixed(2)}%   diff ${(m - fsPy.sim).toFixed(2)}`);
