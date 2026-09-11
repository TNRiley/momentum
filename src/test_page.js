// Smoke-test the page script outside a browser: run every draw with a recording 2D context
// and check that nothing is drawn outside its canvas. This is the check that would have
// caught the device-pixel-ratio bug, where the bitmap height was never scaled and the
// bottom ~20% of every chart (axis labels included) was silently clipped away.
//   node test_page.js
const fs = require("fs");
const path = require("path");

const DPR = +(process.env.DPR || 2);
const CANVAS_W = +(process.env.CW || 814);

const drawn = [];                    // {id, x, y}
let current = null;

function recorder(id, w, h) {
  const push = (x, y) => { if (Number.isFinite(x) && Number.isFinite(y)) drawn.push({id, x, y}); };
  return {
    canvas: {width: w, height: h},
    setTransform(){}, save(){}, restore(){}, beginPath(){}, stroke(){}, fill(){},
    setLineDash(){}, closePath(){},
    clearRect(){}, measureText(){ return {width: 40}; },
    fillRect(x, y, w2, h2){ push(x, y); push(x + w2, y + h2); },
    strokeRect(x, y, w2, h2){ push(x, y); push(x + w2, y + h2); },
    moveTo(x, y){ push(x, y); }, lineTo(x, y){ push(x, y); },
    arc(x, y, r){ push(x - r, y - r); push(x + r, y + r); },
    fillText(s, x, y){ push(x, y - 6); push(x, y + 6); },
    getImageData(){ return {data: new Uint8ClampedArray(4)}; },
    set fillStyle(v){}, get fillStyle(){ return "#000"; },
    set strokeStyle(v){}, get strokeStyle(){ return "#000"; },
    set lineWidth(v){}, get lineWidth(){ return 1; },
    set font(v){}, get font(){ return ""; },
    set textAlign(v){}, get textAlign(){ return "left"; },
    set textBaseline(v){}, get textBaseline(){ return "top"; },
    set globalAlpha(v){}, get globalAlpha(){ return 1; },
  };
}

const CANVASES = ["ladder", "hist", "board", "hotladder", "subs", "sweep", "sample"];
const els = new Map();
function el(id) {
  if (els.has(id)) return els.get(id);
  const isCanvas = CANVASES.includes(id);
  const o = {
    id, style: {}, dataset: {}, textContent: "", innerHTML: "", disabled: false,
    clientWidth: CANVAS_W, offsetWidth: 150, offsetHeight: 60,
    width: 0, height: {ladder:330, hist:280, board:300, hotladder:300, subs:300, sweep:290, sample:260}[id] || 0,
    attrs: {},
    appendChild(){}, addEventListener(){}, removeEventListener(){},
    setAttribute(k, v){ this.attrs[k] = v; }, getAttribute(k){ return this.attrs[k]; },
    getBoundingClientRect(){ return {left:0, top:0, width:CANVAS_W, height:this.height}; },
    getContext(){ return recorder(id, this.width, this.height); },
    querySelectorAll(){ return []; },
  };
  if (isCanvas) o.attrs.height = o.height;
  els.set(id, o);
  return o;
}

globalThis.devicePixelRatio = DPR;
globalThis.matchMedia = () => ({matches: false});
globalThis.getComputedStyle = () => ({getPropertyValue: () => "#888888"});
globalThis.addEventListener = () => {};
globalThis.setTimeout = (f) => { return 0; };
globalThis.clearTimeout = () => {};
globalThis.atob = (s) => Buffer.from(s, "base64").toString("binary");
globalThis.document = {
  getElementById: el,
  createElement: () => el("_tmp_" + Math.random()),
  querySelectorAll: () => [],
  documentElement: {getAttribute: () => null, setAttribute: () => {}, style: {}},
};

const tpl = fs.readFileSync(path.join(process.cwd(), "template.html"), "utf8");
const js = tpl.slice(tpl.indexOf("const D = __DATA__"), tpl.lastIndexOf("addEventListener(\"resize\""));
const payload = fs.readFileSync(path.join(process.cwd(), "..", "raw", "payload.json"), "utf8");

let api;
try {
  api = new Function(js.replace("__DATA__", payload) + "\nreturn {redraw, testOf, levelsOf, TIGHT, ALL, setSel:(s,l)=>{sel=s;lvl=l;}};")();
} catch (e) {
  console.error("page script threw on load:", e.message);
  process.exit(1);
}
console.log("page script loaded and drew the initial state");

// redraw every test at every one of its controls
let checked = 0;
for (const t of api.ALL) {
  for (const lv of api.levelsOf(t.key)) {
    api.setSel(t.key, lv);
    drawn.length = 0;
    api.redraw();
    checked++;
    for (const c of CANVASES) {
      const e = el(c), h = +e.dataset.h || e.height;
      const bad = drawn.filter(d => d.id === c && (d.y < -1 || d.y > h + 1 || d.x < -1 || d.x > CANVAS_W + 1));
      if (bad.length) {
        const worst = bad.reduce((a, b) => Math.abs(b.y - h / 2) > Math.abs(a.y - h / 2) ? b : a);
        console.error(`FAIL ${t.key}/${lv}: ${bad.length} marks outside #${c} ` +
          `(${CANVAS_W}x${h}); worst at y=${worst.y.toFixed(1)}, x=${worst.x.toFixed(1)}`);
        process.exit(1);
      }
    }
  }
}
console.log(`checked ${checked} test/control combinations, ${CANVASES.length} canvases each`);

// the bitmap must be the CSS size times the device pixel ratio, or content is clipped
for (const c of CANVASES) {
  const e = el(c), h = +e.dataset.h;
  if (e.width !== Math.round(CANVAS_W * DPR) || e.height !== Math.round(h * DPR)) {
    console.error(`FAIL #${c}: bitmap ${e.width}x${e.height}, expected ${CANVAS_W*DPR}x${h*DPR}`);
    process.exit(1);
  }
}
console.log(`all bitmaps correctly sized at dpr=${DPR}`);
console.log("\nPASS");
