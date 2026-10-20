// Builds slides/LOG6309E_group_presentation.pptx  (node slides/build_deck.js, run from the repo root or slides/)
// CS109-style deck: one running toy (MiniHDFS, invented), real HDFS examples, input -> output on every result slide.
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("./apply_theme.js"); // copied from the pptx skill so it resolves our node_modules

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(__dirname, "LOG6309E_group_presentation.pptx");
const FIG = (p) => path.join(ROOT, "results", p);

// ---------------------------------------------------------------- theme
const HEX = {
  ink: "1B2A35", white: "FFFFFF", muted: "5B6770", panel: "EEF2F4",
  prep: "2A9D8F", lab: "3D5A80", nolab: "7B5EA7", trust: "B5179E", toy: "E07A2F", bad: "C8323F",
};
const THEME = {
  name: "LogBlocks", headFontFace: "Bookman Old Style", bodyFontFace: "Calibri",
  colors: { dk1: HEX.ink, lt1: HEX.white, dk2: HEX.muted, lt2: HEX.panel,
    accent1: HEX.prep, accent2: HEX.lab, accent3: HEX.nolab, accent4: HEX.trust, accent5: HEX.toy, accent6: HEX.bad,
    hlink: HEX.lab, folHlink: HEX.nolab },
};
const MONO = "Courier New";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.title = "Log-based anomaly detection: replication and extension";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const COL = { ink: C.text1, white: C.background1, muted: C.text2, panel: C.background2,
  prep: C.accent1, lab: C.accent2, nolab: C.accent3, trust: C.accent4, toy: C.accent5, bad: C.accent6 };

const THEMES = [
  { key: "prep", name: "Prepare the data", icon: fa.FaFileAlt },
  { key: "lab", name: "Detect with labels", icon: fa.FaTags },
  { key: "nolab", name: "Detect without labels", icon: fa.FaEyeSlash },
  { key: "trust", name: "Trust & explain", icon: fa.FaSearch },
];

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: COL.white },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.45, y: 0.28, w: 7.55, h: 0.55,
      fontFace: THEME.headFontFace, fontSize: 24, color: COL.ink, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.2, y: 5.25, w: 0.45, h: 0.25, fontSize: 10, color: COL.muted, align: "right" },
});
pres.defineSlideMaster({
  title: "DARK",
  background: { color: COL.ink },
  objects: [],
  slideNumber: { x: 9.2, y: 5.25, w: 0.45, h: 0.25, fontSize: 10, color: COL.panel, align: "right" },
});

// ---------------------------------------------------------------- icons
async function iconPng(Icon, hex, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Icon, { color: "#" + hex, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const ICON = {};

// ---------------------------------------------------------------- helpers
let objN = 0;
const nm = (s) => `${s}-${++objN}`;

function title(slide, text) { slide.addText(text, { placeholder: "title" }); }

function tracker(slide, themeIdx) {
  // 4 theme icons top-right; the current one filled with its color
  const d = 0.34, gap = 0.08, x0 = 9.55 - 4 * d - 3 * gap, y = 0.38;
  THEMES.forEach((t, i) => {
    const on = i === themeIdx;
    slide.addShape(pres.shapes.OVAL, { x: x0 + i * (d + gap), y, w: d, h: d, objectName: nm("tracker"),
      fill: { color: on ? COL[t.key] : COL.panel }, line: { color: on ? COL[t.key] : COL.panel } });
    slide.addImage({ data: on ? ICON[t.key + "_w"] : ICON[t.key + "_m"], x: x0 + i * (d + gap) + 0.085, y: y + 0.085,
      w: d - 0.17, h: d - 0.17, objectName: nm("tracker-icon") });
  });
}

function textBox(slide, text, o) {
  slide.addText(text, { isTextBox: true, objectName: nm(o.name || "text"), fontSize: 14, color: COL.ink,
    valign: "top", margin: 0.05, ...o });
}

function onToy(slide, header = "FIRST, ON MINIHDFS (INVENTED)", x = 0.45, y = 1.02, w = 4.35) {
  slide.addImage({ data: ICON.server_toy, x, y: y + 0.02, w: 0.22, h: 0.22, objectName: nm("toy-icon") });
  textBox(slide, header, { x: x + 0.3, y, w: w - 0.3, h: 0.28, fontSize: 12, bold: true, color: COL.toy, charSpacing: 1, margin: 0 });
}

function code(slide, lines, o) {
  // lines: string | [{text, kind}] ; kind: add (green-ish = prep), del (bad), dim (muted), hi (toy)
  const runs = [];
  lines.forEach((ln, i) => {
    const parts = Array.isArray(ln) ? ln : [{ text: ln }];
    parts.forEach((p, j) => {
      const c = p.kind === "del" ? COL.bad : p.kind === "add" ? COL.prep : p.kind === "dim" ? COL.muted
        : p.kind === "hi" ? COL.toy : COL.ink;
      runs.push({ text: p.text, options: { color: c, bold: p.kind === "hi" || p.kind === "del",
        breakLine: j === parts.length - 1 && i < lines.length - 1 } });
    });
  });
  slide.addText(runs, { isTextBox: true, objectName: nm("code"), fontFace: MONO, fontSize: o.fontSize || 11,
    x: o.x, y: o.y, w: o.w, h: o.h, valign: o.valign || "middle", margin: 0.08,
    fill: { color: o.fill || COL.panel }, line: { color: o.fill || COL.panel }, rectRadius: 0.05,
    shape: pres.shapes.ROUNDED_RECTANGLE, paraSpaceAfter: o.para || 0 });
}

function pill(slide, label, colorKey, x, y, w = 1.15, h = 0.36, fontSize = 12) {
  slide.addText(label, { isTextBox: true, objectName: nm("pill"), shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.18,
    x, y, w, h, fill: { color: COL[colorKey] }, line: { color: COL[colorKey] }, color: COL.white, bold: true,
    fontSize, align: "center", valign: "middle", margin: 0.02 });
}

function arrow(slide, x, y, w = 0.32) {
  slide.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h: 0.22, objectName: nm("arrow"),
    fill: { color: COL.muted }, line: { color: COL.muted } });
}

function ioRow(slide, y, inputLines, label, colorKey, o = {}) {
  // input box -> arrow -> pill ; o.note = small muted caption under the input
  const x = o.x || 0.45, inW = o.inW || 2.75, h = o.h || 0.62;
  code(slide, inputLines, { x, y, w: inW, h, fontSize: o.fontSize || 11 });
  arrow(slide, x + inW + 0.08, y + h / 2 - 0.11);
  pill(slide, label, colorKey, x + inW + 0.48, y + h / 2 - 0.18, o.pillW || 1.12, 0.36, o.pillFont || 12);
  if (o.note) textBox(slide, o.note, { x, y: y + h + 0.02, w: inW + 1.6, h: 0.24, fontSize: 10, color: COL.muted, italic: true, margin: 0 });
}

function upsideDown(slide, text, x, y, w, h, fontSize = 12) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, objectName: nm("answer-box"),
    fill: { color: COL.white }, line: { color: COL.muted, dashType: "dash", width: 1 } });
  slide.addText(text, { isTextBox: true, objectName: nm("answer"), x: x + 0.08, y: y + 0.05, w: w - 0.16, h: h - 0.1,
    rotate: 180, fontSize, italic: true, color: COL.muted, align: "left", valign: "middle", margin: 0 });
}

function means(slide, bullets, colorKey, box = { x: 5.05, y: 3.38, w: 4.5, h: 1.85 }, head = "What it means") {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { ...box, rectRadius: 0.08, objectName: nm("means-card"),
    fill: { color: COL.panel }, line: { color: COL.panel } });
  textBox(slide, head, { x: box.x + 0.15, y: box.y + 0.08, w: box.w - 0.3, h: 0.28, fontSize: 13, bold: true,
    color: COL[colorKey], margin: 0 });
  slide.addText(bullets.map((b, i) => ({ text: b, options: { bullet: { indent: 12 }, breakLine: i < bullets.length - 1 } })), {
    isTextBox: true, objectName: nm("means-text"), x: box.x + 0.12, y: box.y + 0.38, w: box.w - 0.24, h: box.h - 0.45,
    fontSize: box.fontSize || 12.5, color: COL.ink, valign: "top", margin: 0.02, paraSpaceAfter: 3 });
}

function caption(slide, text, x, y, w) {
  textBox(slide, text, { x, y, w, h: 0.24, fontSize: 10, color: COL.muted, italic: true, margin: 0 });
}

function figure(slide, file, ratio, box, cap) {
  // fit image into box keeping aspect ratio, centered horizontally
  let w = box.w, h = w / ratio;
  if (h > box.h) { h = box.h; w = h * ratio; }
  slide.addImage({ path: file, x: box.x + (box.w - w) / 2, y: box.y, w, h, objectName: nm("figure") });
  if (cap) caption(slide, cap, box.x, box.y + h + 0.03, box.w);
}

const AX = { fontFace: "+mn-lt" };
function barChart(slide, data, o) {
  slide.addChart(pres.charts.BAR, data, {
    x: o.x, y: o.y, w: o.w, h: o.h, objectName: nm("chart"), barDir: o.barDir || "col", barGrouping: "clustered",
    chartColors: o.colors, showLegend: data.length > 1, legendPos: "t", legendFontSize: 10, legendFontFace: "+mn-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFontFace: "+mn-lt",
    dataLabelColor: HEX.ink, dataLabelFormatCode: o.fmt || "0.000",
    catAxisLabelColor: HEX.muted, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: HEX.muted, valAxisLabelFontSize: 9, valAxisLabelFontFace: "+mn-lt",
    valAxisMinVal: o.min, valAxisMaxVal: o.max, valAxisMajorUnit: o.unit, valAxisLabelFormatCode: o.axFmt || "0.0",
    valGridLine: { color: "DDE3E7", size: 0.5 }, catGridLine: { style: "none" },
    showTitle: !!o.title, title: o.title, titleFontSize: 11, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    barGapWidthPct: 60,
  });
}

function addSlide(master, section) { return pres.addSlide({ masterName: master, sectionTitle: section }); }

// ================================================================= slides
async function build() {
  for (const t of THEMES) {
    ICON[t.key + "_w"] = await iconPng(t.icon, HEX.white);
    ICON[t.key + "_m"] = await iconPng(t.icon, HEX.muted);
    ICON[t.key + "_c"] = await iconPng(t.icon, HEX[t.key]);
  }
  ICON.server_toy = await iconPng(fa.FaServer, HEX.toy);
  ICON.server_w = await iconPng(fa.FaServer, HEX.white);
  ICON.server_bad = await iconPng(fa.FaServer, HEX.bad);
  ICON.question = await iconPng(fa.FaQuestionCircle, HEX.toy);
  ICON.lightbulb = await iconPng(fa.FaLightbulb, HEX.toy);

  // ---------- 1. Title
  pres.addSection({ title: "Warm-up" });
  let s = addSlide("DARK", "Warm-up");
  textBox(s, "The Block That Never Finished", { x: 0.6, y: 1.25, w: 8.8, h: 0.9, fontFace: THEME.headFontFace,
    fontSize: 38, color: COL.white, bold: true, name: "deck-title" });
  textBox(s, "Replicating and extending a study of log representations for anomaly detection",
    { x: 0.6, y: 2.15, w: 8.8, h: 0.45, fontSize: 18, color: COL.panel });
  textBox(s, "Wu, Li & Khomh, “On the Effectiveness of Log Representation for Log-based Anomaly Detection”, EMSE 2023",
    { x: 0.6, y: 2.75, w: 8.8, h: 0.3, fontSize: 12, italic: true, color: COL.panel });
  textBox(s, "[TEAM NAMES]  ·  LOG6309E Intelligent DevOps  ·  Group assignment", { x: 0.6, y: 4.55, w: 6.5, h: 0.3, fontSize: 13, color: COL.toy, bold: true });
  [0, 1, 2].forEach((i) => s.addImage({ data: i === 2 ? ICON.server_bad : ICON.server_w, x: 7.45 + i * 0.62, y: 3.55, w: 0.45, h: 0.45, objectName: nm("doodle") }));
  textBox(s, "the write never finished", { x: 7.3, y: 4.05, w: 2.2, h: 0.25, fontSize: 10, italic: true, color: COL.panel, align: "center" });
  s.addNotes(`Hi everyone. Our talk is about a block that never finished. [pause]
We replicated part of a 2023 EMSE paper by Wu, Li and Khomh. The paper asks a simple question: when we turn log lines into numbers, which way of doing it helps a model detect anomalies best?
We redid their first research question on two datasets, then we went further on HDFS: models that need no labels, a statistical ranking, and an explanation of the best model.
We will use one small invented system the whole way through, plus real log lines. Let's start with a puzzle.`);

  // ---------- 2. Hook puzzle (real HDFS blocks)
  s = addSlide("CONTENT", "Warm-up");
  title(s, "Puzzle: one of these blocks is broken");
  textBox(s, "Block A  ·  blk_3796613345934950050  (13 lines, real)", { x: 0.45, y: 1.0, w: 5.2, h: 0.28, fontSize: 12, bold: true, color: COL.ink, margin: 0 });
  code(s, [
    "Receiving block blk_3796… src:/10.251.74.79",
    "Receiving block blk_3796… src:/10.251.110.68",
    "Receiving block blk_3796… src:/10.251.110.68",
    "NameSystem.allocateBlock: …/part-00002",
    "PacketResponder 0 for block blk_3796… terminating",
    "Received block blk_3796… of size 67108864",
    "Received block blk_3796… of size 67108864",
    "PacketResponder 1 for block blk_3796… terminating",
    "Received block blk_3796… of size 67108864",
    "addStoredBlock: blockMap updated … (x3 lines)",
    "PacketResponder 2 for block blk_3796… terminating",
  ], { x: 0.45, y: 1.3, w: 5.2, h: 2.85, fontSize: 11, valign: "middle" });
  textBox(s, "Block B  ·  blk_-5087…  (2 lines, real)", { x: 5.95, y: 1.0, w: 3.6, h: 0.28, fontSize: 12, bold: true, color: COL.ink, margin: 0 });
  code(s, ["Receiving block blk_-5087…", "  src:/10.251.202.181", "NameSystem.allocateBlock:", "  …/part-00028"],
    { x: 5.95, y: 1.3, w: 3.6, h: 1.15, fontSize: 10, valign: "top" });
  s.addImage({ data: ICON.question, x: 5.95, y: 2.62, w: 0.42, h: 0.42, objectName: nm("q") });
  textBox(s, "Which block is broken, and how could a machine tell?", { x: 6.45, y: 2.55, w: 3.1, h: 0.6, fontSize: 14, bold: true, color: COL.toy });
  upsideDown(s, "B. It was allocated and started receiving, then silence. Every one of the 558,223 normal HDFS blocks ends with exactly 3 “PacketResponder … terminating” lines; B has 0.",
    5.95, 3.2, 3.6, 0.95, 11);
  textBox(s, "Our question: can a model learn this from 11 million lines, with labels and without them?",
    { x: 0.45, y: 4.35, w: 9.1, h: 0.4, fontSize: 15, bold: true, color: COL.ink });
  caption(s, "Raw lines from Loghub HDFS_v1 (ids shortened). 2,950 anomalous blocks look exactly like B.", 0.45, 4.8, 9.1);
  s.addNotes(`Here are two real blocks from the HDFS dataset. A block is a chunk of a file that Hadoop copies to three machines.
Block A has thirteen lines. Block B has two. [pause]
Which one is broken? Take five seconds. [pause]
(click: answer) It's B. It was allocated, it started receiving, and then nothing. In the whole dataset, every single normal block, 558 thousand of them, ends with exactly three "PacketResponder terminating" lines, one per copy. B has zero. And B is not alone: 2,950 anomalous blocks look exactly like this.
So the question of our project is: can a model learn this kind of rule on its own, from 11 million lines? With labels, and without labels?`);

  // ---------- 3. Meet the toy
  s = addSlide("CONTENT", "Warm-up");
  title(s, "Meet MiniHDFS (invented for this talk)");
  textBox(s, "A file block is copied to 3 machines. Every step writes one log line. We write each line as one letter:",
    { x: 0.45, y: 1.0, w: 9.1, h: 0.35, fontSize: 14 });
  const letters = [["R", "Receiving block (a copy starts)"], ["A", "allocateBlock (name server books it)"], ["P", "PacketResponder … terminating"],
    ["D", "Received block … (copy done)"], ["S", "addStoredBlock (copy registered)"], ["E", "writeBlock … exception"]];
  letters.forEach(([l, d], i) => {
    const x = 0.45 + (i % 3) * 3.05, y = 1.45 + Math.floor(i / 3) * 0.5;
    code(s, [[{ text: l, kind: "hi" }]], { x, y, w: 0.42, h: 0.38, fontSize: 14 });
    textBox(s, d, { x: x + 0.5, y: y + 0.04, w: 2.5, h: 0.32, fontSize: 12, color: COL.ink, margin: 0 });
  });
  // INPUT -> detector -> OUTPUT strip
  textBox(s, "INPUT: one block's lines", { x: 0.45, y: 2.6, w: 3.0, h: 0.26, fontSize: 11, bold: true, color: COL.muted, margin: 0 });
  code(s, [[{ text: "R R R A P D P D P D S S S" }]], { x: 0.45, y: 2.9, w: 3.0, h: 0.45, fontSize: 13 });
  arrow(s, 3.55, 3.01);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.95, y: 2.9, w: 1.7, h: 0.45, rectRadius: 0.08, objectName: nm("detector"),
    fill: { color: COL.ink }, line: { color: COL.ink } });
  textBox(s, "detector", { x: 3.95, y: 2.9, w: 1.7, h: 0.45, fontSize: 14, bold: true, color: COL.white, align: "center", valign: "middle" });
  arrow(s, 5.75, 3.01);
  textBox(s, "OUTPUT: label + score", { x: 6.15, y: 2.6, w: 3.4, h: 0.26, fontSize: 11, bold: true, color: COL.muted, margin: 0 });
  pill(s, "normal · 0.01", "muted", 6.15, 2.94, 1.55);
  pill(s, "anomaly · 0.99", "bad", 7.85, 2.94, 1.7);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.45, y: 3.75, w: 9.1, h: 1.0, rectRadius: 0.08, objectName: nm("bigidea"),
    fill: { color: COL.panel }, line: { color: COL.panel } });
  s.addImage({ data: ICON.lightbulb, x: 0.65, y: 3.98, w: 0.5, h: 0.5, objectName: nm("bulb") });
  textBox(s, [{ text: "Big idea: ", options: { bold: true, color: COL.toy } },
    { text: "one data point = all the log lines of ONE block (a “session”). The detector never sees a single line alone; it judges the whole block." }],
    { x: 1.3, y: 3.85, w: 8.1, h: 0.8, fontSize: 15, valign: "middle" });
  s.addNotes(`To keep things concrete, we invented a tiny system: MiniHDFS. It's not real, but it behaves like HDFS.
A block is copied to three machines, and each step writes one log line. To save space we write each line as one letter: R when a copy starts, A when the name server books the block, P when a copy's responder terminates, D when a copy is done, S when it is registered, and E for an exception.
A healthy block looks like this: three R, one A, three P, three D, three S. (click: strip) The detector takes the lines of one block and outputs a label and a score.
The big idea: one data point is one block, not one line. Keep that in mind for the whole talk.`);

  // ---------- 4. Vocabulary via analogy
  s = addSlide("CONTENT", "Warm-up");
  title(s, "Vocabulary: logs are parcel tracking");
  const vocab = [
    ["prep", "Template", "the stamp type, blanks removed", "Received block <*> of size <*> from <*>"],
    ["prep", "Session", "one parcel's full tracking history", "block #1 = R R R A P D P D P D S S S"],
    ["prep", "MCV (count vector)", "how many stamps of each type", "[R A P D S E] = [3 1 3 3 3 0]"],
    ["prep", "TF-IDF (Text)", "the words on each stamp, weighted", "P → {packetresponder, terminating, …}"],
    ["lab", "Supervised", "learns from parcels marked “lost”", "Random Forest, LSTM"],
    ["nolab", "Label-free", "never sees a “lost” tag", "Isolation Forest, DeepLog"],
  ];
  vocab.forEach(([k, term, analog, ex], i) => {
    const y = 1.0 + i * 0.66;
    pill(s, term, k, 0.45, y + 0.08, 2.05, 0.4, 12);
    textBox(s, analog, { x: 2.65, y: y + 0.1, w: 3.0, h: 0.38, fontSize: 14, margin: 0, valign: "middle" });
    code(s, [ex], { x: 5.75, y: y + 0.06, w: 3.8, h: 0.44, fontSize: 11 });
  });
  s.addNotes(`Six words and we are ready. Think of a log as parcel tracking.
A template is the stamp type: the fixed text of a line with the variable parts blanked out. A session is one parcel's whole history, for us one block.
Then we need numbers. The Message Count Vector, MCV, just counts how many stamps of each type a block has: three R, one A, three P, and so on. TF-IDF Text instead looks at the words written on each stamp.
Finally, models. Supervised models learn from blocks that someone labelled as broken. Label-free models never see that tag: Isolation Forest and DeepLog.`);

  // ---------- 5. Your turn
  s = addSlide("CONTENT", "Warm-up");
  title(s, "Your turn!");
  textBox(s, "1. Which template does this real line become?", { x: 0.45, y: 1.0, w: 9.1, h: 0.3, fontSize: 15, bold: true });
  code(s, ["Received block blk_3796613345934950050 of size 67108864 from /10.251.74.79"], { x: 0.45, y: 1.38, w: 9.1, h: 0.45, fontSize: 12 });
  upsideDown(s, "Received block <*> of size <*> from <*>", 0.45, 1.93, 4.6, 0.42, 12);
  textBox(s, "2. MiniHDFS block #4 is  R R A .  What is its MCV [R A P D S E]?  Normal or not?",
    { x: 0.45, y: 2.75, w: 9.1, h: 0.3, fontSize: 15, bold: true });
  code(s, [[{ text: "R R A", kind: "hi" }]], { x: 0.45, y: 3.13, w: 1.8, h: 0.45, fontSize: 14 });
  upsideDown(s, "[2 1 0 0 0 0]. Anomaly: no P at all, the write never finished (like block B).", 0.45, 3.68, 6.0, 0.42, 12);
  s.addImage({ data: ICON.question, x: 8.75, y: 3.9, w: 0.75, h: 0.75, objectName: nm("q") });
  s.addNotes(`Quick self-check. Two questions. [pause]
First: which template does this real line become? The block id, the size and the IP address are variables, so they become blanks. (click: answer) "Received block <*> of size <*> from <*>".
Second: MiniHDFS block number 4 is R, R, A. What is its count vector, and is it normal? [pause]
(click: answer) Two R, one A, zeros everywhere else. And it is broken, for the same reason as block B: no P at all.
If you got both, you already understand how our data looks.`);

  // ---------- 6. The toy's diary
  pres.addSection({ title: "The data" });
  s = addSlide("CONTENT", "The data");
  title(s, "MiniHDFS's diary: three blocks, three rows");
  const diary = [
    ["#1", "R R R A P D P D P D S S S", "[3 1 3 3 3 0]", "normal", "muted", "all 3 copies finished"],
    ["#2", "R A", "[1 1 0 0 0 0]", "anomaly", "bad", "stops early (like block B)"],
    ["#3", "R R A E", "[2 1 0 0 0 1]", "anomaly", "bad", "exception, never finishes"],
  ];
  textBox(s, "block", { x: 0.45, y: 1.0, w: 0.6, h: 0.25, fontSize: 11, bold: true, color: COL.muted, margin: 0 });
  textBox(s, "event sequence (input)", { x: 1.1, y: 1.0, w: 3.6, h: 0.25, fontSize: 11, bold: true, color: COL.muted, margin: 0 });
  textBox(s, "MCV [R A P D S E]", { x: 4.85, y: 1.0, w: 2.1, h: 0.25, fontSize: 11, bold: true, color: COL.muted, margin: 0 });
  textBox(s, "label", { x: 7.15, y: 1.0, w: 1.2, h: 0.25, fontSize: 11, bold: true, color: COL.muted, margin: 0 });
  diary.forEach(([id, seq, mcv, lab, k, why], i) => {
    const y = 1.32 + i * 0.85;
    textBox(s, id, { x: 0.45, y: y + 0.08, w: 0.6, h: 0.4, fontSize: 18, bold: true, color: COL.toy, margin: 0 });
    code(s, [[{ text: seq }]], { x: 1.1, y, w: 3.6, h: 0.52, fontSize: 13 });
    code(s, [[{ text: mcv, kind: i === 0 ? undefined : "del" }]], { x: 4.85, y, w: 2.1, h: 0.52, fontSize: 13 });
    pill(s, lab, k, 7.15, y + 0.08, 1.15);
    textBox(s, why, { x: 1.1, y: y + 0.55, w: 6.0, h: 0.24, fontSize: 11, italic: true, color: COL.muted, margin: 0 });
  });
  textBox(s, "Each block = one row of the dataset. The model sees the left part (sequence or counts) and must predict the label.",
    { x: 0.45, y: 4.0, w: 9.1, h: 0.55, fontSize: 15, bold: true, color: COL.ink });
  s.addNotes(`Here is MiniHDFS's diary: three blocks. Each block becomes exactly one row of the dataset.
Block 1 is healthy: thirteen events and the counts three, one, three, three, three.
Block 2 stops after A: that's our real block B. Block 3 hits an exception and never finishes.
The model gets the left part, either the sequence of events or the counts, and has to predict the label on the right. Notice that for blocks 2 and 3, the counts alone already scream "broken": the P, D and S columns are zero.`);

  // ---------- 7. Scale up (funnel)
  s = addSlide("CONTENT", "The data");
  title(s, "Scale up: 3 toy blocks → 575,061 real ones");
  const funnel = (x, name, rows, k) => {
    textBox(s, name, { x, y: 0.98, w: 4.4, h: 0.3, fontSize: 15, bold: true, color: COL[k], margin: 0 });
    rows.forEach(([n, lab, frac], i) => {
      const y = 1.38 + i * 0.66, w = 4.4 * frac;
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: Math.max(w, 0.08), h: 0.3, objectObjectName: undefined, objectName: nm("funnel-bar"),
        fill: { color: COL[k], transparency: i * 18 }, line: { color: COL[k], transparency: i * 18 } });
      textBox(s, [{ text: n + "  ", options: { bold: true } }, { text: lab, options: { color: COL.muted } }],
        { x, y: y + 0.31, w: 4.4, h: 0.28, fontSize: 12, margin: 0 });
    });
  };
  funnel(0.45, "HDFS (Hadoop file system)", [["11,175,629", "log lines", 1], ["46", "templates (Drain)", 0.55],
    ["575,061", "sessions = blocks", 0.4], ["16,838", "anomalous blocks (2.9%)", 0.12], ["402,542 / 172,519", "train / test (70/30)", 0.3]], "prep");
  funnel(5.15, "BGL (BlueGene/L supercomputer)", [["4,747,963", "log lines", 1], ["382", "templates (Drain)", 0.55],
    ["826", "sessions = 6-hour windows", 0.4], ["409", "anomalous windows (49.5%)", 0.2], ["660 / 166", "train / test (80/20)", 0.3]], "lab");
  caption(s, "Bar lengths are illustrative, not to scale. Every model uses the same 10 stratified train/test splits (seeds 0-9).", 0.45, 4.92, 9.1);
  s.addNotes(`Now scale up. MiniHDFS gave us three blocks. Real HDFS has 11 million lines, which collapse into 46 templates and 575 thousand blocks. Only 2.9 percent of blocks are anomalous, so it is a needle-in-a-haystack problem.
The second dataset, BGL, comes from a supercomputer. It has no block id, so like the paper we cut it into six-hour windows: only 826 sessions. And here almost half of the windows are anomalous, because a six-hour window almost always contains some alert.
Keep these two shapes in mind: HDFS has lots of sessions and few anomalies; BGL has few sessions and many anomalies.`);

  // ---------- 8. Roadmap
  s = addSlide("CONTENT", "The data");
  title(s, "Roadmap: four questions about MiniHDFS");
  const road = [
    ["Can we turn raw lines into R/A/P letters and counts faithfully?", "Parsing, sessions, representation  (steps 1-2)"],
    ["If someone labelled the broken blocks, which detector learns best?", "Replication of the paper's RQ1 + correlation/VIF"],
    ["Can we catch block #2 with no labels at all?", "Extension: Isolation Forest, DeepLog, Scott-Knott ranking"],
    ["Can we trust the scores, and why does the model flag #2?", "Duplicates check, SHAP explanation"],
  ];
  road.forEach(([q, what], i) => {
    const y = 1.0 + i * 1.0, t = THEMES[i];
    s.addShape(pres.shapes.OVAL, { x: 0.45, y: y + 0.05, w: 0.62, h: 0.62, objectName: nm("road-circle"), fill: { color: COL[t.key] }, line: { color: COL[t.key] } });
    s.addImage({ data: ICON[t.key + "_w"], x: 0.6, y: y + 0.2, w: 0.32, h: 0.32, objectName: nm("road-icon") });
    textBox(s, t.name, { x: 1.25, y, w: 3.0, h: 0.3, fontSize: 13, bold: true, color: COL[t.key], margin: 0 });
    textBox(s, q, { x: 1.25, y: y + 0.3, w: 8.3, h: 0.3, fontSize: 15, margin: 0 });
    textBox(s, what, { x: 1.25, y: y + 0.6, w: 8.3, h: 0.26, fontSize: 11, italic: true, color: COL.muted, margin: 0 });
  });
  s.addNotes(`Our talk follows four questions, each with its own color and icon, which you'll see in the top-right corner of every slide.
Teal: can we turn raw lines into letters and counts faithfully? Steel blue: if someone labelled the broken blocks, which detector learns best? That's the replication of the paper.
Violet: can we catch block 2 without any labels? That's our extension.
And magenta: can we trust these numbers, and why does the best model flag block 2?`);

  // ---------- 9. Parsing
  pres.addSection({ title: "Prepare the data" });
  s = addSlide("CONTENT", "Prepare the data");
  title(s, "Step 1 · Can we recover the templates?");
  tracker(s, 0);
  onToy(s);
  ioRow(s, 1.4, ["PacketResponder 1 for block", "blk_42 terminating"], "P", "prep", { pillW: 0.6 });
  ioRow(s, 2.2, ["Receiving block blk_42", "src: /10.0.0.7:49239"], "R", "prep", { pillW: 0.6, note: "regex masks ids, IPs, numbers before Drain" });
  textBox(s, "Your turn: “Received block blk_42 of size 67108864” → ?", { x: 0.45, y: 3.2, w: 4.35, h: 0.3, fontSize: 12, bold: true, margin: 0 });
  upsideDown(s, "D  (Received block <*> of size <*> …)", 0.45, 3.55, 3.2, 0.38, 11);
  barChart(s, [{ name: "no preprocessing", labels: ["HDFS", "BGL"], values: [0.998, 0.963] },
    { name: "our regex preprocessing", labels: ["HDFS", "BGL"], values: [1.0, 0.970] },
    { name: "our first (buggy) version", labels: ["HDFS", "BGL"], values: [1.0, 0.918] }],
    { x: 5.0, y: 0.95, w: 4.6, h: 2.3, min: 0.85, max: 1.05, unit: 0.05, axFmt: "0.00", colors: ["B9C4CC", HEX.prep, HEX.bad], title: "Group accuracy on Loghub-2k ground truth", fmt: "0.000" });
  means(s, ["HDFS 1.000, BGL 0.970: almost every line gets its true template",
    "Our first version re-spaced “<*>:” → BGL 0.918; found by ablation",
    "BGL: 1,444 → 382 templates. HDFS: 46 = the paper's 46 MCV dims"], "prep");
  s.addNotes(`Step one: parsing. On MiniHDFS it's easy: "PacketResponder 1 for block blk_42 terminating" becomes the letter P. On real data we use Drain, plus regular expressions that blank out block ids, IP addresses and numbers before Drain sees them.
Your turn: what does "Received block blk_42 of size 67108864" become? [pause] D.
(click: chart) We checked against hand-labelled ground truth. Guess first: how many lines end in the right template? [pause] HDFS: all of them, 1.000. BGL: 0.970.
One lesson: our first version scored only 0.918 on BGL, worse than no preprocessing. We had added a space between a wildcard and a colon, which changes the number of tokens, and Drain groups lines by token count. We found it by removing one regex at a time.
And a good sign: HDFS gives 46 templates, exactly the 46 dimensions the paper reports.`);

  // ---------- 10. Sessions / replication check
  s = addSlide("CONTENT", "Prepare the data");
  title(s, "Step 2 · Does our data match the paper's?");
  tracker(s, 0);
  onToy(s, "FIRST, ON A TOY BGL (INVENTED)");
  textBox(s, "No block id, so we cut by time:", { x: 0.45, y: 1.32, w: 4.35, h: 0.28, fontSize: 13, margin: 0 });
  ioRow(s, 1.7, ["01:10 INFO parity fixed", "04:55 FATAL TLB error"], "window 1: anomaly", "bad", { pillW: 1.55, inW: 2.5, fontSize: 10.5, pillFont: 11 });
  ioRow(s, 2.5, ["07:02 INFO generating", "10:30 INFO LOGIN ok"], "window 2: normal", "muted", { pillW: 1.55, inW: 2.5, fontSize: 10.5, pillFont: 11, note: "6-hour windows; anomalous if any line is an alert" });
  s.addTable([
    [{ text: "", options: { bold: true } }, { text: "ours", options: { bold: true } }, { text: "paper", options: { bold: true } }],
    ["HDFS sessions", "575,061", "575,061"], ["HDFS train / test", "402,542 / 172,519", "402,542 / 172,519"],
    ["BGL sessions", { text: "826", options: { bold: true, color: COL.bad } }, { text: "718", options: { bold: true, color: COL.bad } }],
    ["BGL lines / session", "5,748", "6,565"],
  ], { x: 5.05, y: 1.0, w: 4.5, colW: [1.7, 1.45, 1.35], fontSize: 12, color: COL.ink, objectName: nm("table"),
    border: { type: "solid", pt: 0.5, color: "DDE3E7" }, fill: { color: COL.white }, rowH: 0.36, margin: 0.05 });
  caption(s, "Paper: Wu et al. 2023, Table 1 and Sect. 4.", 5.05, 2.88, 4.5);
  means(s, ["HDFS: identical to the paper, split included",
    "BGL: 4,713,493 non-empty lines / 718 = 6,564.8 ≈ paper's 6,565: they likely dropped empty lines",
    "Window rule not described → we keep 826",
    "49.5% anomalous: “always anomaly” gets F1 0.66"], "prep",
    { x: 5.05, y: 3.18, w: 4.5, h: 2.05, fontSize: 12 });
  s.addNotes(`Step two: sessions. BGL has no block id, so we cut by time. In this toy, two lines between 1 and 5 am form window one; it contains a FATAL alert, so the whole window is anomalous.
(click: table) Does our data match the paper? HDFS: exactly, down to the train-test split. BGL: no. We get 826 windows, the paper reports 718.
We did some detective work. The paper says 6,565 lines per session on average. If you remove the 34 thousand lines that have an empty message, and divide by 718, you get 6,564.8. So they most likely dropped those lines, the same bug we had in our first parse. The window boundaries still differ and the paper doesn't describe them, so we keep ours and report it.
Also notice: half the BGL windows are anomalous, so even "always say anomaly" gets an F1 of 0.66. Keep that in mind when you read BGL numbers.`);

  // ---------- 11. RQ1
  pres.addSection({ title: "Detect with labels" });
  s = addSlide("CONTENT", "Detect with labels");
  title(s, "RQ1 · With labels, which detector wins?");
  tracker(s, 1);
  onToy(s);
  ioRow(s, 1.4, ["block #3 counts", "[2 1 0 0 0 1]"], "RF: anomaly", "bad", { pillW: 1.3 });
  ioRow(s, 2.2, ["block #3 as word vectors", "R → R → A → E"], "LSTM: anomaly", "bad", { pillW: 1.3,
    note: "RF reads counts (MCV); LSTM reads TF-IDF vectors in order" });
  textBox(s, "Guess first: does the deep model beat the forest?", { x: 0.45, y: 3.3, w: 4.35, h: 0.3, fontSize: 12, bold: true, margin: 0 });
  upsideDown(s, "No. On both datasets the Random Forest on counts wins, as in the paper.", 0.45, 3.68, 4.3, 0.55, 11);
  barChart(s, [{ name: "ours (mean of 10 splits)", labels: ["RF HDFS", "LSTM HDFS", "RF BGL", "LSTM BGL"], values: [0.998, 0.968, 0.921, 0.851] },
    { name: "paper", labels: ["RF HDFS", "LSTM HDFS", "RF BGL", "LSTM BGL"], values: [0.999, 0.955, 0.891, 0.871] }],
    { x: 5.0, y: 0.95, w: 4.6, h: 2.3, min: 0.7, max: 1.05, unit: 0.1, colors: [HEX.lab, "B9C4CC"], title: "F1 on the test sets", fmt: "0.000" });
  means(s, ["All 4 cells within ~0.03 F1 of the paper",
    "Random Forest on counts ≥ LSTM on both datasets",
    "BGL: ±0.03-0.05 across splits (166 test windows)"], "lab");
  s.addNotes(`Now the replication. If someone labelled the broken blocks, which detector learns best?
On the toy: the Random Forest gets block 3's counts and says anomaly. The LSTM reads the same block as a sequence of word vectors, R, R, A, E, and also says anomaly.
Guess first: does the deep model beat the forest? [pause]
(click: chart) No. On HDFS the forest gets 0.998 F1, the LSTM 0.968. On BGL, 0.921 versus 0.851. All four numbers are within about 0.03 of the paper, and the ordering is the same as theirs.
One caveat: BGL has only 166 test windows, so results move by 3 to 5 points between splits. That's why every number here is an average over ten splits.`);

  // ---------- 12. Correlation / VIF
  s = addSlide("CONTENT", "Detect with labels");
  title(s, "RQ1b · Should we drop correlated features?");
  tracker(s, 1);
  onToy(s);
  ioRow(s, 1.4, ["P, D, S move together:", "(3,3,3) or (0,0,0)"], "keep P only", "lab", { pillW: 1.3, inW: 2.75 });
  ioRow(s, 2.2, ["filtered: P's vector", "= [0 0 … 0]"], "= padding!", "bad", { pillW: 1.3, inW: 2.75,
    note: "the LSTM can no longer tell P from “nothing”" });
  s.addTable([
    [{ text: "model · data", options: { bold: true } }, { text: "#features", options: { bold: true } }, { text: "F1 all → filtered", options: { bold: true } }, { text: "Holm p", options: { bold: true } }],
    ["RF · HDFS", "47 → 28", "0.998 → 0.997", "0.014"],
    ["RF · BGL", "366 → 125", "0.921 → 0.909", "0.82"],
    ["LSTM · HDFS", "113 → 22", { text: "0.968 → 0.923", options: { bold: true, color: COL.bad } }, "0.020"],
    ["LSTM · BGL", "627 → 108", "0.851 → 0.810", "0.25"],
  ], { x: 5.05, y: 1.0, w: 4.5, colW: [1.25, 1.05, 1.45, 0.75], fontSize: 12, color: COL.ink, objectName: nm("table"),
    border: { type: "solid", pt: 0.5, color: "DDE3E7" }, fill: { color: COL.white }, rowH: 0.36, margin: 0.05 });
  caption(s, "Spearman |ρ|>0.7 + VIF≤5 on train; Wilcoxon, 10 paired splits", 5.05, 2.88, 4.5);
  means(s, ["RF: up to 2/3 of features removed, F1 ≈ unchanged",
    "LSTM HDFS −0.045 F1: 110 word columns from 45 templates (rank 39); 17 templates turn all-zero",
    "Filter count tables, not embeddings"], "lab", { x: 5.05, y: 3.2, w: 4.5, h: 2.03 });
  s.addNotes(`The assignment asked us to remove correlated and redundant features, and compare with and without.
On the toy: the counts of P, D and S always move together, three-three-three or zero-zero-zero. So the filter keeps just one of them, and the forest loses nothing.
(click: table) That's what we see: the forest drops up to two thirds of its features, and F1 barely moves.
The LSTM is a different story: minus 0.045 F1 on HDFS, significant even after correcting for multiple tests. Why? The LSTM's features are words, and those words come from only 45 templates, so most of them are perfectly redundant. The filter throws them away without asking whether they help tell templates apart. After filtering, 17 templates have a vector of all zeros, exactly the padding vector. On the toy: the LSTM can no longer tell P from nothing.
Our take-away: this filter is great for count tables, not for embeddings.`);

  // ---------- 13. Without labels
  pres.addSection({ title: "Detect without labels" });
  s = addSlide("CONTENT", "Detect without labels");
  title(s, "Ext. 1 · Catch block #2 without labels?");
  tracker(s, 2);
  onToy(s);
  ioRow(s, 1.4, ["Isolation Forest: counts", "[1 1 0 0 0 0] are rare"], "anomaly", "bad", { pillW: 1.1 });
  ioRow(s, 2.2, ["DeepLog: after R, A is a", "likely next event … end"], "normal ✗", "muted", { pillW: 1.1 });
  ioRow(s, 3.0, ["DeepLog + EOS token:", "after A, EOS is unlikely"], "anomaly ✓", "bad", { pillW: 1.1,
    note: "DeepLog learns from normal blocks only; EOS = end-of-session" });
  barChart(s, [{ name: "F1", labels: ["DeepLog", "DeepLog + EOS", "Isolation Forest"], values: [0.573, 0.640, 0.654] },
    { name: "AUC", labels: ["DeepLog", "DeepLog + EOS", "Isolation Forest"], values: [0.839, 0.961, 0.958] }],
    { x: 5.0, y: 0.95, w: 4.6, h: 2.3, min: 0, max: 1.2, unit: 0.2, colors: ["B9A6D6", HEX.nolab], title: "HDFS, mean of 10 splits", fmt: "0.000" });
  means(s, ["2,950 real broken blocks are just “R, A, silence”: without EOS, DeepLog never checks them",
    "EOS: AUC 0.84 → 0.96; F1 still below the literature's ~0.9 (100k blocks, CPU)",
    "EOS added after inspecting split 0 (diagnosis, not tuning)"], "nolab");
  s.addNotes(`Now the hard part: no labels. Can we still catch block 2?
Isolation Forest looks at the counts. One-one-zero-zero is a rare pattern, so it gets isolated quickly: anomaly.
DeepLog learns from normal blocks only, and predicts the next event. After R, is A a likely next event? Yes. And then... the block ends. There is nothing left to predict, so nothing looks wrong. DeepLog says normal. [pause]
The fix is tiny: add an end-of-session token, EOS. Now, after A, the model expects another R or a P, and sees EOS instead: anomaly.
(click: chart) This matters a lot on real data: 2,950 broken blocks are exactly "R, A, silence". With EOS, AUC goes from 0.84 to 0.96.
Honesty corner: we're still below the 0.9 F1 reported in the literature; we trained on a sample on a CPU. And we added EOS after looking at the first split, so we report the original too.`);

  // ---------- 14. Ranking
  s = addSlide("CONTENT", "Detect without labels");
  title(s, "Ext. 2 · Guess: how much are labels worth?");
  tracker(s, 2);
  onToy(s, "GUESS FIRST");
  textBox(s, "F1 gap between the best supervised and the best label-free model on HDFS?", { x: 0.45, y: 1.4, w: 4.35, h: 0.6, fontSize: 14, bold: true, margin: 0 });
  ["0.05", "0.15", "0.35"].forEach((g, i) => pill(s, g, "toy", 0.45 + i * 1.35, 2.1, 1.15, 0.42, 14));
  upsideDown(s, "≈ 0.34  (RF 0.998 vs Isolation Forest 0.654)", 0.45, 2.75, 4.0, 0.42, 11);
  s.addTable([
    [{ text: "rank", options: { bold: true } }, { text: "by F1", options: { bold: true } }, { text: "by AUC", options: { bold: true } }],
    ["1", "RF 0.998", "RF 1.000"], ["2", "LSTM 0.968", "LSTM 0.9996"],
    ["3", "Isolation F. 0.654", "DeepLog+EOS 0.961"], ["4", "DeepLog+EOS 0.640", "Isolation F. 0.958"],
  ], { x: 0.45, y: 3.35, w: 4.35, colW: [0.55, 1.85, 1.95], fontSize: 11, color: COL.ink, objectName: nm("table"),
    border: { type: "solid", pt: 0.5, color: "DDE3E7" }, fill: { color: COL.white }, rowH: 0.3, margin: 0.04 });
  figure(s, path.join(__dirname, "fig_ranking_f1.png"), 2.065, { x: 5.0, y: 0.95, w: 4.6, h: 2.2 }, "F1 over 10 splits, HDFS. Scott-Knott ESD (R ScottKnottESD).");
  means(s, ["Labels buy ~0.34 F1: supervised models rank 1-2 on both metrics",
    "Label-free models swap places by metric: no clear winner",
    "DeepLog+EOS is less stable (F1 std 0.078 vs 0.020)"], "nolab", { x: 5.05, y: 3.45, w: 4.5, h: 1.78 });
  s.addNotes(`Now we rank all four models. Guess first: how much F1 do we lose without labels? 5 points? 15? 35? [pause]
(click: answer) About 34 points: 0.998 for the forest, 0.654 for the best label-free model.
(click: figure) We used the Scott-Knott ESD test, the same family of test the paper uses, over ten resampled splits. Supervised models are ranks one and two on both F1 and AUC.
The two label-free models are close and swap places depending on the metric: Isolation Forest wins on F1, DeepLog with EOS wins on AUC. So we say: no clear winner. Isolation Forest is much more stable across splits, though.`);

  // ---------- 15. Trust
  pres.addSection({ title: "Trust & explain" });
  s = addSlide("CONTENT", "Trust & explain");
  title(s, "Can we trust F1 = 0.998?");
  tracker(s, 3);
  onToy(s);
  ioRow(s, 1.4, ["test block:", "R R R A P D P D P D S S S"], "seen in train", "trust", { pillW: 1.3, inW: 2.75 });
  textBox(s, "Like an exam with the homework questions: the model may just remember.", { x: 0.45, y: 2.2, w: 4.35, h: 0.55, fontSize: 13, italic: true, color: COL.muted, margin: 0 });
  const stat = (x, y, big, small, k) => {
    textBox(s, big, { x, y, w: 2.2, h: 0.55, fontSize: 30, bold: true, color: COL[k], margin: 0 });
    textBox(s, small, { x, y: y + 0.55, w: 2.2, h: 0.5, fontSize: 11, color: COL.muted, margin: 0 });
  };
  stat(5.05, 1.0, "18,375", "distinct sequences for 575,061 blocks", "trust");
  stat(7.35, 1.0, "97.8%", "of test blocks repeat a training sequence", "trust");
  stat(5.05, 2.15, "0.989", "RF F1 on the 3,789 test blocks with a NEW sequence", "lab");
  stat(7.35, 2.15, "0.41", "Isolation Forest F1 with a label-free threshold (vs 0.654)", "nolab");
  ioRow(s, 3.0, ["same block, label-free", "threshold (‘auto’)"], "F1 0.65→0.41", "nolab", { pillW: 1.3, inW: 2.75 });
  means(s, ["Duplicates inflate supervised scores a little, but cannot explain the 0.34 gap (new sequences: 0.989)",
    "Isolation Forest's F1 relies on contamination 0.03 ≈ the true 2.9% anomaly rate; AUC (0.96) does not"], "trust",
    { x: 5.05, y: 3.45, w: 4.5, h: 1.78 });
  s.addNotes(`Before celebrating 0.998, let's be skeptical. HDFS has only 18 thousand distinct event sequences for 575 thousand blocks. And 97.8 percent of the test blocks repeat a sequence that's already in the training set. It's like an exam that reuses the homework questions. [pause]
So we re-scored the forest only on test blocks with a brand-new sequence: still 0.989. Counts generalise: a new order of the same events has the same counts. Duplicates inflate the score a little, but they don't explain the big gap.
Second check: Isolation Forest needs a threshold. We used the library default of 3 percent, which happens to be close to the real anomaly rate. With a truly label-free threshold, its F1 drops to 0.41. AUC doesn't depend on the threshold and stays at 0.96. So: trust AUC more for label-free models.`);

  // ---------- 16. SHAP
  s = addSlide("CONTENT", "Trust & explain");
  title(s, "Ext. 3 · Why does the forest flag block #2?");
  tracker(s, 3);
  onToy(s, "SHAP ON ONE REAL BLOCK LIKE #2");
  code(s, [[{ text: "start: 0.03 (base rate)", kind: "dim" }], [{ text: "P count = 0      +0.90", kind: "del" }],
    [{ text: "delete count     +0.07" }], [{ text: "final score:     1.00", kind: "hi" }]], { x: 0.45, y: 1.4, w: 2.75, h: 1.3, fontSize: 12, para: 2 });
  arrow(s, 3.28, 1.94);
  pill(s, "anomaly", "bad", 3.68, 1.87, 1.1);
  caption(s, "Numbers from the real waterfall plot, seed 0 (shap_waterfall_vif_seed0.png).", 0.45, 2.75, 4.35);
  textBox(s, [{ text: "Normal blocks: ", options: { bold: true } }, { text: "558,223 of 558,223 have exactly 3 P lines." , options: {} }],
    { x: 0.45, y: 3.15, w: 4.35, h: 0.3, fontSize: 13, margin: 0 });
  textBox(s, [{ text: "Anomalous blocks: ", options: { bold: true } }, { text: "6,181 (36.7%) have 0 P lines; none has exactly 2.", options: {} }],
    { x: 0.45, y: 3.5, w: 4.35, h: 0.5, fontSize: 13, margin: 0 });
  // mean |SHAP| from results/ext/shap/shap_importance_vif_seed0.csv (1,000 + 1,000 test blocks); reversed so the largest is on top
  barChart(s, [{ name: "mean |SHAP|", labels: ["Redundant addStoredBlock", "NameSystem.delete → invalidSet",
    "addStoredBlock: not in any file", "Starting thread to transfer block", "Unexpected error deleting block", "P: PacketResponder … terminating"],
    values: [0.0285, 0.0288, 0.0369, 0.0798, 0.1441, 0.1446] }],
    { x: 5.0, y: 0.95, w: 4.6, h: 2.35, barDir: "bar", min: 0, max: 0.18, unit: 0.05, axFmt: "0.00", colors: [HEX.trust],
      title: "Mean |SHAP|, RF on HDFS (seed 0, 2,000 test blocks)", fmt: "0.000" });
  means(s, ["Two tied top signals: the P count and “Unexpected error deleting block”",
    "The forest rediscovered HDFS's 3-copy write protocol",
    "Explained on the 28-feature corr/VIF model, so correlated P/D/S don't split the credit"], "trust",
    { x: 5.05, y: 3.45, w: 4.5, h: 1.78 });
  s.addNotes(`Last question: why does the forest flag block 2? We used SHAP, which tells us how much each feature pushed the score up or down.
Here is one real test block that looks like our block 2: the model starts at the base rate, 3 percent. The P count is zero: plus 0.9. A small push from the delete lines, and we end at 1.0, anomaly.
(click: chart) Across 2,000 test blocks, two signals are tied at the top: the P count, and the error line "Unexpected error trying to delete block". Then re-replication: "Starting thread to transfer block".
And this matches the data: every one of the 558 thousand normal blocks has exactly three P lines, and more than a third of the anomalous blocks have zero. The forest rediscovered HDFS's three-copy write protocol from counts alone.
We explained the filtered model so that correlated features like P, D and S don't split the credit.`);

  // ---------- 17. Recap
  pres.addSection({ title: "Wrap-up" });
  s = addSlide("CONTENT", "Wrap-up");
  title(s, "Recap: MiniHDFS vs the real thing");
  const recap = [
    ["Lines → letters and counts", "HDFS GA 1.000, BGL 0.970; HDFS matches the paper, BGL 826 vs 718 windows"],
    ["Forest beats LSTM", "RF 0.998 / 0.921 F1 ≥ LSTM 0.968 / 0.851; VIF fine for RF, bad for LSTM"],
    ["Block #2 needs an EOS token", "Label-free: F1 ≈ 0.65, AUC ≈ 0.96; no clear winner"],
    ["“No P” = broken", "SHAP: missing P lines drive the forest; 97.8% repeated sequences"],
  ];
  recap.forEach(([toy, real], i) => {
    const x = 0.45 + (i % 2) * 4.6, y = 1.0 + Math.floor(i / 2) * 2.05, t = THEMES[i];
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.45, h: 1.85, rectRadius: 0.08, objectName: nm("recap-card"),
      fill: { color: COL.panel }, line: { color: COL.panel } });
    s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: y + 0.15, w: 0.45, h: 0.45, objectName: nm("recap-circle"), fill: { color: COL[t.key] }, line: { color: COL[t.key] } });
    s.addImage({ data: ICON[t.key + "_w"], x: x + 0.26, y: y + 0.26, w: 0.23, h: 0.23, objectName: nm("recap-icon") });
    textBox(s, t.name, { x: x + 0.7, y: y + 0.2, w: 3.6, h: 0.3, fontSize: 13, bold: true, color: COL[t.key], margin: 0 });
    textBox(s, [{ text: "Toy: ", options: { bold: true, color: COL.toy } }, { text: toy }], { x: x + 0.2, y: y + 0.72, w: 4.1, h: 0.35, fontSize: 13, margin: 0 });
    textBox(s, [{ text: "Real: ", options: { bold: true } }, { text: real }], { x: x + 0.2, y: y + 1.07, w: 4.1, h: 0.7, fontSize: 12, margin: 0 });
  });
  s.addNotes(`Let's recap with MiniHDFS. Turning lines into letters and counts works: near-perfect parsing, and HDFS matches the paper exactly.
With labels, the forest on counts beats the LSTM, and the correlation filter is fine for the forest but hurts the LSTM.
Without labels, block 2 needs an EOS (end-of-session) token, and the two label-free models end up close, around 0.65 F1.
And the reason it all works: no P lines means a broken block, which is exactly what SHAP shows.`);

  // ---------- 18. Confidence
  s = addSlide("CONTENT", "Wrap-up");
  title(s, "How confident should we be?");
  const caveats = [
    [0, "Drain ran on all data before the split (as in the paper); BGL windows differ (826 vs 718)"],
    [1, "Random session split + 97.8% repeated sequences make HDFS easier (not discussed by Wu et al.)"],
    [1, "LSTM epochs and class weights were fixed by us; the paper does not report them"],
    [2, "Isolation Forest threshold ≈ true anomaly rate; DeepLog trained on 100k blocks on CPU, EOS added after split 0"],
    [3, "Each seed changes both the split and the model initialisation; extension on HDFS only"],
  ];
  caveats.forEach(([ti, txt], i) => {
    const y = 1.0 + i * 0.8, t = THEMES[ti];
    s.addShape(pres.shapes.OVAL, { x: 0.45, y: y + 0.05, w: 0.45, h: 0.45, objectName: nm("cav-circle"), fill: { color: COL[t.key] }, line: { color: COL[t.key] } });
    s.addImage({ data: ICON[t.key + "_w"], x: 0.56, y: y + 0.16, w: 0.23, h: 0.23, objectName: nm("cav-icon") });
    textBox(s, txt, { x: 1.1, y: y + 0.02, w: 8.45, h: 0.55, fontSize: 14, valign: "middle", margin: 0 });
  });
  s.addNotes(`How confident should you be in all this? Five caveats, each tagged with the part it affects.
Parsing ran on all the data before splitting, as in the paper, and our BGL windows don't match theirs.
The biggest one, which the original paper does not discuss: HDFS is split randomly, and almost every test block repeats a training sequence. That makes HDFS look easier than a real deployment would be.
We fixed some LSTM settings ourselves. Isolation Forest's threshold borrows from the true anomaly rate, and DeepLog was trained on a sample. Finally, every seed changes both the split and the model, and the extension uses HDFS only.`);

  // ---------- 19. One sentence
  s = addSlide("DARK", "Wrap-up");
  textBox(s, "On HDFS, the loudest anomaly is a line that never shows up.", { x: 0.8, y: 1.6, w: 8.4, h: 1.3,
    fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: COL.white, valign: "middle" });
  textBox(s, "Count before you read: a forest on template counts beats sequence models, and finds the missing lines.",
    { x: 0.8, y: 3.05, w: 8.4, h: 0.7, fontSize: 18, color: COL.toy });
  s.addNotes(`If you remember one sentence: on HDFS, the loudest anomaly is a line that never shows up.
Count before you read. A simple forest on template counts beats the sequence models, and it does so by noticing the lines that are missing.`);

  // ---------- 20. Your turn + Q&A
  s = addSlide("CONTENT", "Wrap-up");
  title(s, "Your turn: what if MiniHDFS changed?");
  textBox(s, "MiniHDFS always writes 3 copies. Suppose each file chooses 1 to 5 copies.\nWould counting (MCV + Random Forest) still win?",
    { x: 0.45, y: 1.0, w: 9.1, h: 0.85, fontSize: 17, bold: true });
  const hints = ["“exactly 3 P lines” stops being a fixed rule", "DeepLog does not care how many copies there are",
    "Could we divide each count by the number of copies?"];
  hints.forEach((h, i) => {
    const x = 0.45 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.1, w: 2.9, h: 1.05, rectRadius: 0.15, objectName: nm("hint"),
      fill: { color: COL.panel }, line: { color: COL.toy, width: 1 } });
    textBox(s, [{ text: "Hint " + (i + 1) + ": ", options: { bold: true, color: COL.toy } }, { text: h }],
      { x: x + 0.12, y: 2.15, w: 2.66, h: 0.95, fontSize: 13, valign: "middle", margin: 0 });
  });
  textBox(s, "Thank you!  Questions?", { x: 0.45, y: 3.65, w: 9.1, h: 0.75, fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: COL.ink });
  textBox(s, "Code, results and notes: git tags step1-final … ext-final", { x: 0.45, y: 4.45, w: 9.1, h: 0.3, fontSize: 12, color: COL.muted, italic: true });
  s.addNotes(`To finish, a question for you. MiniHDFS always writes three copies. What if each file could choose between one and five copies? Would counting still win? [pause]
Hint one: "exactly three P lines" would no longer be a fixed rule. Hint two: DeepLog doesn't care how many copies there are; it only looks at what comes next. Hint three: maybe we could divide each count by the number of copies.
We'd love to hear what you think. Thank you, and we're happy to take questions.`);

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
