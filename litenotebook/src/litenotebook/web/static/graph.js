/* LiteNotebook knowledge graph: Global and Local views over graph memory.
 *
 * Canvas drawing with a small force-directed layout (Barnes–Hut repulsion,
 * springs, a gentle pull to the centre), pan / zoom / drag, an inspector and
 * a structured list view for keyboard and screen-reader use. Data comes from
 * /memory/graph/*.json (bounded on the server). All text goes through
 * textContent — never innerHTML. Only harmless view preferences are kept in
 * localStorage (no labels, no content).
 */
(function () {
  "use strict";
  const root = document.getElementById("kg");
  const canvas = document.getElementById("kg-canvas");
  if (!root || !canvas) return;
  const $ = (id) => document.getElementById(id);
  const nb = root.dataset.notebookId;
  const base = "/notebooks/" + nb + "/memory/graph";
  const stage = $("kg-stage"), statusEl = $("kg-status"), tip = $("kg-tip");
  const inspector = $("kg-inspector"), inspectorBody = $("kg-inspector-body");
  const listEl = $("kg-list"), listBox = $("kg-list-box"), listSummary = $("kg-list-summary");
  const searchInput = $("kg-q"), hitsEl = $("kg-hits");
  const ctx = canvas.getContext("2d");
  const TYPE_NAMES = {
    concept: "Concept", note: "Note", source: "Source", conversation: "Conversation",
    correction: "Correction", question: "Question", answer: "Answer", evidence: "Evidence",
    annotation: "Annotation", quiz: "Quiz",
  };
  const FLAG_NAMES = {
    needs_review: "needs review", seen: "seen in quizzes", strong_recently: "strong recently",
    useful: "marked useful", dead_end: "dead end", corrected: "corrected", superseded: "superseded by a correction",
    has_correction: "has a correction", source_deleted: "source deleted",
  };
  const PLURALS = {
    concept: "Concepts", note: "Notes", source: "Sources", conversation: "Conversations",
    correction: "Corrections", question: "Questions", answer: "Answers", evidence: "Evidence",
    annotation: "Annotations", quiz: "Quizzes",
  };
  const PREF_KEY = "litenotebook.graph.v1";

  let mode = root.dataset.mode === "local" ? "local" : "global";
  let center = root.dataset.center || "";
  let depth = Math.min(3, Math.max(1, Number(root.dataset.depth) || 1));
  let data = null;                 // last payload
  let nodes = [], edges = [], byId = new Map(), adj = new Map();
  let selected = null, hovered = null, matches = null;
  let view = { k: 1, x: 0, y: 0 };   // screen = world * k + (x, y)
  let alpha = 0, frame = 0, dirty = true;
  let autoFit = true;  // fit again when the layout settles, until the user moves the view
  let explicit = false;  // the user chose the selection (the local centre alone does not dim)
  let colors = {};
  // prefers-reduced-motion: the layout settles out of sight, in short chunks, and
  // only the settled graph is drawn (no animated settling, no springing on drag).
  const motionQuery = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  const reducedMotion = () => !!(motionQuery && motionQuery.matches);
  let quiet = false, quietToken = 0;
  let animatedFrames = 0, quietLayouts = 0;  // read by automated checks

  // --- preferences (view settings only) -------------------------------------------

  function loadPrefs() {
    try { return JSON.parse(localStorage.getItem(PREF_KEY) || "{}") || {}; } catch (e) { return {}; }
  }
  function savePrefs() {
    try {
      localStorage.setItem(PREF_KEY, JSON.stringify({
        types: types(), orphans: $("kg-orphans").checked, min: minDegree(),
        labels: $("kg-labels").checked, depth: depth, concepts: $("kg-concepts").value,
      }));
    } catch (e) { /* private mode: preferences are optional */ }
  }
  (function applyPrefs() {
    const p = loadPrefs();
    if (Array.isArray(p.types) && p.types.length) {
      document.querySelectorAll("#kg-types input").forEach((i) => { i.checked = p.types.indexOf(i.value) >= 0; });
    }
    if (typeof p.orphans === "boolean") $("kg-orphans").checked = p.orphans;
    if (typeof p.labels === "boolean") $("kg-labels").checked = p.labels;
    if (["auto", "0", "2", "3"].indexOf(p.concepts) >= 0) $("kg-concepts").value = p.concepts;
    if (Number.isFinite(p.min)) $("kg-min").value = String(Math.max(0, Math.min(50, p.min)));
    if (!root.dataset.center && [1, 2, 3].indexOf(p.depth) >= 0) depth = p.depth;
    document.querySelectorAll("input[name=kg-depth]").forEach((i) => { i.checked = Number(i.value) === depth; });
  })();

  function types() { return Array.from(document.querySelectorAll("#kg-types input:checked")).map((i) => i.value); }
  function minDegree() { return Math.max(0, Math.min(50, Number($("kg-min").value) || 0)); }

  function readColors() {
    const css = getComputedStyle(document.documentElement);
    const v = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);
    colors = { bg: v("--surface", "#111"), text: v("--text", "#ddd"), muted: v("--muted", "#999"),
               edge: v("--g-edge", "#3a4540"), focus: v("--focus", "#a8e6bd"), warn: v("--warn", "#d9b45f"),
               select: v("--g-select", "#ffd27a"), danger: v("--danger", "#e07a7a") };
    Object.keys(TYPE_NAMES).forEach((t) => { colors[t] = v("--g-" + t, "#888"); });
  }

  // --- data -------------------------------------------------------------------------

  function say(text) { statusEl.textContent = text; }
  const plural = (n, word) => n.toLocaleString() + " " + word + (n === 1 ? "" : "s");

  let loadToken = 0, pending = false;
  async function load() {
    const token = ++loadToken;  // a newer request wins; late answers are ignored
    pending = true;
    say("Loading graph…");
    let url;
    if (mode === "local") {
      if (!center) center = pickCenter();
      if (!center) { mode = "global"; syncModeButtons(); return load(); }
      url = base + "/local.json?" + new URLSearchParams({ node: center, depth: String(depth), types: types().join(",") });
    } else {
      url = base + "/view.json?" + new URLSearchParams({
        types: types().join(","), min_degree: String(minDegree()),
        hide_orphans: $("kg-orphans").checked ? "1" : "0",
        concept_min: $("kg-concepts").value,
      });
      const limit = new URLSearchParams(window.location.search).get("max_nodes");
      if (limit && /^\d{1,4}$/.test(limit)) url += "&max_nodes=" + limit;  // server caps it
    }
    let payload;
    try {
      const r = await fetch(url, { credentials: "same-origin" });
      payload = await r.json();
      if (token !== loadToken) return;
      if (!r.ok) { pending = false; say(payload.error || "Could not load the graph"); return; }
    } catch (e) { if (token === loadToken) { pending = false; say("Could not load the graph"); } return; }
    pending = false;
    setData(payload);
  }

  function pickCenter() {
    if (selected) return selected.id;
    if (!nodes.length) return "";
    const concept = nodes.filter((n) => n.type === "concept").sort((a, b) => b.degree - a.degree)[0];
    return (concept || nodes[0]).id;
  }

  function setData(payload) {
    const keep = new Map(nodes.map((n) => [n.id, n]));
    data = payload;
    nodes = payload.nodes.map((n, i) => {
      const old = keep.get(n.id);
      const node = Object.assign({}, n, { r: radius(n.degree), index: i });
      if (old) { node.x = old.x; node.y = old.y; node.vx = 0; node.vy = 0; node.fx = old.fx; node.fy = old.fy; }
      return node;
    });
    byId = new Map(nodes.map((n) => [n.id, n]));
    edges = payload.edges.map((e) => ({ s: byId.get(e.source), t: byId.get(e.target), e: e }))
      .filter((l) => l.s && l.t);
    adj = new Map(nodes.map((n) => [n.id, new Set()]));
    edges.forEach((l) => { adj.get(l.s.id).add(l.t.id); adj.get(l.t.id).add(l.s.id); });
    seed();
    if (selected) selected = byId.get(selected.id) || null;
    if (!selected && mode === "local") selected = byId.get(payload.center) || null;
    explicit = false;
    hovered = null;
    describe();
    buildList();
    applySearch();
    autoFit = true;
    warm();
    fit();
    if (selected) inspect(selected.id);
    else if (mode === "global") showHint();
  }

  function describe() {
    if (!data) return;
    const shown = nodes.length, total = data.nodes_total;
    let text;
    if (!shown) text = "Nothing to show with these filters — tick more node types or lower the minimum connections.";
    else if (mode === "local") {
      const c = byId.get(data.center);
      text = "Local graph of “" + (c ? c.label : "") + "”, depth " + data.depth + ": " +
        plural(shown, "node") + ", " + plural(edges.length, "link") + ".";
      if (data.truncated) text += " Showing " + shown + " of " + total.toLocaleString() + " reachable nodes (limit " + data.limits.nodes + ").";
    } else {
      text = data.truncated
        ? "Showing the most connected " + shown.toLocaleString() + " of " + total.toLocaleString() + " nodes."
        : plural(shown, "node") + ", " + plural(edges.length, "link") + ".";
    }
    if (mode === "global" && data.concepts_hidden) {
      text += " " + plural(data.concepts_hidden, "concept") + " mentioned by only one item " +
        (data.concepts_hidden === 1 ? "is" : "are") + " hidden (Filters → Concepts).";
    }
    if (data.edges_truncated) text += " Showing the strongest " + edges.length.toLocaleString() + " of " + data.edges_total.toLocaleString() + " links.";
    if (data.complete_load === false) text += " This notebook's memory is very large; only part of it was read.";
    if (selected) text += " Selected " + TYPE_NAMES[selected.type] + " “" + selected.label + "”.";
    say(text);
    listSummary.textContent = "List view of this graph (" + plural(shown, "node") + ")";
  }

  function radius(degree) { return Math.max(4.5, Math.min(15, 4.5 + 2.1 * Math.sqrt(Math.min(degree, 60)))); }

  // --- layout -----------------------------------------------------------------------

  function seed() {  // deterministic: a sunflower spiral in the server's order
    let i = 0;
    nodes.forEach((n) => {
      if (n.x !== undefined) return;
      const r = 12 * Math.sqrt(0.5 + i), a = i * 2.399963229728653;
      n.x = r * Math.cos(a); n.y = r * Math.sin(a); n.vx = 0; n.vy = 0; i += 1;
    });
    if (mode === "local" && data && byId.get(data.center)) {
      const c = byId.get(data.center);
      if (c.fx === undefined || c.fx === null) { c.x = 0; c.y = 0; }
    }
    alpha = 1;
  }

  function quadtree() {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    nodes.forEach((n) => { x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x); y1 = Math.max(y1, n.y); });
    const size = Math.max(x1 - x0, y1 - y0, 1) + 1;
    const rootQ = { x: x0, y: y0, s: size, n: 0, cx: 0, cy: 0, kids: null, leaf: null };
    function insert(q, p, depthQ) {
      q.cx = (q.cx * q.n + p.x) / (q.n + 1); q.cy = (q.cy * q.n + p.y) / (q.n + 1); q.n += 1;
      if (!q.kids && !q.leaf) { q.leaf = p; return; }
      if (depthQ > 24) return;  // coincident points
      if (!q.kids) {
        const h = q.s / 2;
        q.kids = [0, 1, 2, 3].map((k) => ({ x: q.x + (k & 1) * h, y: q.y + (k >> 1) * h, s: h, n: 0, cx: 0, cy: 0, kids: null, leaf: null }));
        const old = q.leaf; q.leaf = null; place(q, old, depthQ);
      }
      place(q, p, depthQ);
    }
    function place(q, p, depthQ) {
      const h = q.s / 2, k = (p.x >= q.x + h ? 1 : 0) + (p.y >= q.y + h ? 2 : 0);
      insert(q.kids[k], p, depthQ + 1);
    }
    nodes.forEach((n) => insert(rootQ, n, 0));
    return rootQ;
  }

  const REPULSE = 260, THETA2 = 0.81, LINK = 46, GRAVITY = 0.06, DECAY = 0.0228, FRICTION = 0.58;

  function tick() {
    if (!nodes.length) return;
    const q = quadtree();
    nodes.forEach((n) => {
      let fx = 0, fy = 0;
      const stack = [q];
      while (stack.length) {
        const c = stack.pop();
        if (!c.n || (c.leaf === n && !c.kids)) continue;
        let dx = c.cx - n.x, dy = c.cy - n.y, d2 = dx * dx + dy * dy;
        if (c.kids && (c.s * c.s) / (d2 || 1) > THETA2) { c.kids.forEach((k) => stack.push(k)); continue; }
        if (d2 < 1) { dx = (n.index % 7) - 3 || 1; dy = (n.index % 5) - 2 || 1; d2 = dx * dx + dy * dy; }
        if (d2 > 640000) continue;
        const f = (REPULSE * c.n * alpha) / d2;
        fx -= dx * f; fy -= dy * f;
      }
      n.vx += fx; n.vy += fy;
      n.vx -= n.x * GRAVITY * alpha; n.vy -= n.y * GRAVITY * alpha;
    });
    edges.forEach((l) => {
      const a = l.s, b = l.t;
      let dx = b.x + b.vx - a.x - a.vx, dy = b.y + b.vy - a.y - a.vy;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const want = LINK + a.r + b.r;
      const strength = 0.7 / Math.min(adj.get(a.id).size, adj.get(b.id).size);
      const f = ((d - want) / d) * alpha * strength;
      dx *= f; dy *= f;
      const wa = adj.get(b.id).size / (adj.get(a.id).size + adj.get(b.id).size);
      b.vx -= dx * (1 - wa); b.vy -= dy * (1 - wa);
      a.vx += dx * wa; a.vy += dy * wa;
    });
    nodes.forEach((n) => {
      if (n.fx !== undefined && n.fx !== null) { n.x = n.fx; n.y = n.fy; n.vx = 0; n.vy = 0; return; }
      n.vx *= FRICTION; n.vy *= FRICTION;
      n.x += n.vx; n.y += n.vy;
    });
    alpha += (0 - alpha) * DECAY;
    if (alpha < 0.004) alpha = 0;
  }

  function warm() {  // settle most of the layout before the first frame
    if (reducedMotion()) { settleQuietly(); return; }
    const started = performance.now();
    let n = 0;
    while (alpha > 0.03 && performance.now() - started < 450 && n < 300) { tick(); n += 1; }
    dirty = true; loop();
  }

  const QUIET_CHUNK_MS = 12, QUIET_MAX_TICKS = 600;
  function settleQuietly() {  // run the whole layout without drawing intermediate frames
    const token = ++quietToken;
    let ticks = 0;
    quiet = true;
    function chunk() {
      if (token !== quietToken) return;  // a newer layout replaced this one
      const started = performance.now();
      while (alpha > 0 && ticks < QUIET_MAX_TICKS && performance.now() - started < QUIET_CHUNK_MS) { tick(); ticks += 1; }
      if (alpha > 0 && ticks < QUIET_MAX_TICKS) { setTimeout(chunk, 0); return; }  // yield to input
      alpha = 0; quiet = false; quietLayouts += 1;
      if (autoFit) { autoFit = false; fit(); }
      redraw();
    }
    chunk();
  }

  function loop() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      if (quiet) return;  // settling out of sight; the settled graph is drawn once
      if (alpha > 0) {
        tick(); dirty = true; animatedFrames += 1;
        if (alpha === 0 && autoFit) { autoFit = false; fit(); }
      }
      if (dirty) { draw(); dirty = false; }
      if (alpha > 0) loop();
    });
  }
  function redraw() { dirty = true; loop(); }
  function reheat(a) { alpha = Math.max(alpha, a); loop(); }

  // --- drawing ----------------------------------------------------------------------

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    const w = stage.clientWidth, h = stage.clientHeight;
    canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio);
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    redraw();
  }

  function shape(n, x, y, r) {
    ctx.beginPath();
    switch (n.type) {
      case "note": ctx.rect(x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7); break;
      case "source": ctx.rect(x - r * 0.7, y - r, r * 1.4, r * 2); break;
      case "conversation": ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r * 1.15, y); ctx.lineTo(x, y + r * 1.15); ctx.lineTo(x - r * 1.15, y); ctx.closePath(); break;
      case "correction": ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 1.1, y + r * 0.85); ctx.lineTo(x - r * 1.1, y + r * 0.85); ctx.closePath(); break;
      case "question": ctx.moveTo(x, y + r * 1.15); ctx.lineTo(x + r * 1.1, y - r * 0.8); ctx.lineTo(x - r * 1.1, y - r * 0.8); ctx.closePath(); break;
      case "answer": for (let i = 0; i < 6; i += 1) { const a = Math.PI / 3 * i; ctx[i ? "lineTo" : "moveTo"](x + r * Math.cos(a), y + r * Math.sin(a)); } ctx.closePath(); break;
      case "evidence": ctx.rect(x - r * 0.6, y - r * 0.6, r * 1.2, r * 1.2); break;
      case "annotation": ctx.arc(x, y, r * 0.75, 0, Math.PI * 2); break;
      case "quiz": for (let i = 0; i < 10; i += 1) { const a = Math.PI / 5 * i - Math.PI / 2, rr = i % 2 ? r * 0.5 : r * 1.15; ctx[i ? "lineTo" : "moveTo"](x + rr * Math.cos(a), y + rr * Math.sin(a)); } ctx.closePath(); break;
      default: ctx.arc(x, y, r, 0, Math.PI * 2);
    }
  }

  function focusSet() {
    const f = hovered || (selected && !(mode === "local" && data && selected.id === data.center && !explicit) ? selected : null);
    if (!f) return null;
    const set = new Set(adj.get(f.id) || []);
    set.add(f.id);
    return set;
  }

  function draw() {
    const ratio = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(ratio * view.k, 0, 0, ratio * view.k, ratio * view.x, ratio * view.y);
    const near = focusSet();
    const dimmed = (n) => (near && !near.has(n.id)) || (matches && !matches.has(n.id));
    ctx.lineWidth = 1 / view.k;
    edges.forEach((l) => {
      const on = near && near.has(l.s.id) && near.has(l.t.id) && (l.s === (selected || hovered) || l.t === (selected || hovered));
      const faded = (near && !on) || (matches && !(matches.has(l.s.id) && matches.has(l.t.id)));
      ctx.globalAlpha = faded ? 0.07 : on ? 0.95 : 0.45;
      ctx.strokeStyle = on ? colors.focus : colors.edge;
      ctx.lineWidth = (on ? 1.8 : 1 + Math.min(2, Math.log2(l.e.weight || 1) * 0.4)) / view.k;
      const superseded = l.e.type === "corrects" || l.s.flags.indexOf("superseded") >= 0 || l.t.flags.indexOf("superseded") >= 0;
      ctx.setLineDash(superseded ? [4 / view.k, 3 / view.k] : []);
      ctx.beginPath(); ctx.moveTo(l.s.x, l.s.y); ctx.lineTo(l.t.x, l.t.y); ctx.stroke();
    });
    ctx.setLineDash([]);
    nodes.forEach((n) => {
      const faded = dimmed(n);
      const superseded = n.flags.indexOf("superseded") >= 0 || n.flags.indexOf("dead_end") >= 0;
      ctx.globalAlpha = faded ? 0.18 : superseded ? 0.55 : 1;
      shape(n, n.x, n.y, n.r);
      ctx.fillStyle = colors[n.type] || colors.muted;
      ctx.fill();
      ctx.lineWidth = 1.2 / view.k;
      ctx.strokeStyle = colors.bg;
      if (superseded) { ctx.setLineDash([3 / view.k, 2 / view.k]); ctx.strokeStyle = colors.muted; }
      ctx.stroke();
      ctx.setLineDash([]);
      if (n.flags.indexOf("needs_review") >= 0) {
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r + 3.2, 0, Math.PI * 2);
        ctx.strokeStyle = colors.warn; ctx.lineWidth = 1.6 / view.k; ctx.stroke();
      }
      if (n === selected || n === hovered || (data && n.id === data.center && mode === "local")) {
        ctx.globalAlpha = 1;
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r + 5.5, 0, Math.PI * 2);
        ctx.strokeStyle = n === selected ? colors.select : colors.focus;
        ctx.lineWidth = (n === selected ? 2.4 : 1.4) / view.k; ctx.stroke();
      }
    });
    if ($("kg-labels").checked) drawLabels(near, dimmed);
    ctx.globalAlpha = 1;
  }

  function drawLabels(near, dimmed) {
    // Greedy placement: most important first; a label that would overlap one
    // already drawn is skipped (zoom in or hover to read it).
    const fontPx = 11.5, k = Math.max(view.k, 0.35);
    ctx.font = (fontPx / k) + "px system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    const focus = selected || hovered;
    const rank = (n) => (n === focus ? 0 : (near && near.has(n.id)) || (matches && matches.has(n.id)) ? 1 : 2);
    const order = nodes.filter((n) => !dimmed(n) || rank(n) < 2)
      .sort((a, b) => rank(a) - rank(b) || b.degree - a.degree || a.index - b.index);
    const placed = [];
    const lineH = (fontPx + 2) / k;
    order.forEach((n) => {
      const text = n.label.length > 34 ? n.label.slice(0, 33) + "…" : n.label;
      const w = ctx.measureText(text).width, x0 = n.x - w / 2, y0 = n.y + n.r + 3 / k;
      const box = [x0 - 2 / k, y0, x0 + w + 2 / k, y0 + lineH];
      if (rank(n) === 2 && placed.some((b) => box[0] < b[2] && b[0] < box[2] && box[1] < b[3] && b[1] < box[3])) return;
      placed.push(box);
      ctx.globalAlpha = dimmed(n) ? 0.35 : 1;
      ctx.fillStyle = rank(n) < 2 ? colors.text : colors.muted;
      ctx.fillText(text, n.x, y0);
    });
  }

  // --- view: zoom, pan, fit --------------------------------------------------------

  function toWorld(px, py) { return [(px - view.x) / view.k, (py - view.y) / view.k]; }
  function zoomAt(factor, px, py) {
    autoFit = false;
    const k = Math.min(6, Math.max(0.12, view.k * factor));
    const [wx, wy] = toWorld(px, py);
    view.k = k; view.x = px - wx * k; view.y = py - wy * k;
    redraw();
  }
  function zoomCenter(factor) { zoomAt(factor, stage.clientWidth / 2, stage.clientHeight / 2); }
  function fit() {
    if (!nodes.length) { view = { k: 1, x: stage.clientWidth / 2, y: stage.clientHeight / 2 }; redraw(); return; }
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    nodes.forEach((n) => { x0 = Math.min(x0, n.x - n.r); y0 = Math.min(y0, n.y - n.r); x1 = Math.max(x1, n.x + n.r); y1 = Math.max(y1, n.y + n.r + 14); });
    const w = stage.clientWidth, h = stage.clientHeight, pad = 28;
    const labelRoom = $("kg-labels").checked ? Math.min(80, w / 8) : 0;  // labels are centred under nodes
    const k = Math.min(2.2, Math.max(0.12, Math.min((w - 2 * (pad + labelRoom)) / Math.max(x1 - x0, 1), (h - 2 * pad) / Math.max(y1 - y0, 1))));
    view = { k: k, x: w / 2 - ((x0 + x1) / 2) * k, y: h / 2 - ((y0 + y1) / 2) * k };
    redraw();
  }
  function centerOn(n) {
    view.x = stage.clientWidth / 2 - n.x * view.k; view.y = stage.clientHeight / 2 - n.y * view.k;
    if (view.k < 0.9) zoomCenter(1.2 / view.k);
    redraw();
  }

  // --- pointer ----------------------------------------------------------------------

  function nodeAt(px, py) {
    const [x, y] = toWorld(px, py);
    let best = null, bestD = Infinity;
    nodes.forEach((n) => {
      const d = Math.hypot(n.x - x, n.y - y);
      if (d < n.r + 6 / view.k && d < bestD) { best = n; bestD = d; }
    });
    return best;
  }
  function local(ev) { const r = canvas.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; }

  let drag = null;
  canvas.addEventListener("pointerdown", (ev) => {
    const [px, py] = local(ev);
    const n = nodeAt(px, py);
    canvas.setPointerCapture(ev.pointerId);
    drag = { n: n, px: px, py: py, vx: view.x, vy: view.y, moved: false };
    stage.focus({ preventScroll: true });
  });
  canvas.addEventListener("pointermove", (ev) => {
    const [px, py] = local(ev);
    if (drag) {
      if (Math.hypot(px - drag.px, py - drag.py) > 3) drag.moved = true;
      if (!drag.moved) return;
      autoFit = false;
      if (drag.n) {
        const [wx, wy] = toWorld(px, py);
        drag.n.fx = wx; drag.n.fy = wy;
        if (reducedMotion()) { drag.n.x = wx; drag.n.y = wy; redraw(); }  // only the dragged node moves
        else reheat(0.25);
      } else {
        view.x = drag.vx + px - drag.px; view.y = drag.vy + py - drag.py; redraw();
      }
      return;
    }
    const n = nodeAt(px, py);
    if (n !== hovered) {
      hovered = n;
      canvas.style.cursor = n ? "pointer" : "grab";
      if (n) { tip.textContent = TYPE_NAMES[n.type] + ": " + n.label + " · " + plural(n.degree, "connection"); tip.hidden = false; }
      else tip.hidden = true;
      redraw();
    }
  });
  canvas.addEventListener("pointerup", (ev) => {
    if (!drag) return;
    const d = drag; drag = null;
    if (!d.moved) select(d.n, true);
  });
  canvas.addEventListener("pointerleave", () => { if (!drag && hovered) { hovered = null; tip.hidden = true; redraw(); } });
  canvas.addEventListener("dblclick", (ev) => {
    const [px, py] = local(ev);
    const n = nodeAt(px, py);
    if (n) focusLocal(n.id);
  });
  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const [px, py] = local(ev);
    zoomAt(Math.exp(-ev.deltaY * 0.0015), px, py);
  }, { passive: false });

  // --- keyboard ----------------------------------------------------------------------

  stage.addEventListener("keydown", (ev) => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const k = ev.key;
    if (k.startsWith("Arrow")) { ev.preventDefault(); step(k); }
    else if (k === "Enter") {
      if (!selected) return;
      ev.preventDefault();
      const first = inspectorBody.querySelector("a.kg-primary");
      if (first) window.location.href = first.getAttribute("href");
    } else if (k === "Escape") { if (selected || matches) { ev.preventDefault(); clearSelection(); } }
    else if (k === "+" || k === "=") { ev.preventDefault(); zoomCenter(1.25); }
    else if (k === "-" || k === "_") { ev.preventDefault(); zoomCenter(0.8); }
    else if (k === "0") { ev.preventDefault(); fit(); }
    else if (k === "l" || k === "L") { if (selected) { ev.preventDefault(); focusLocal(selected.id); } }
  });

  function step(key) {
    if (!nodes.length) return;
    if (!selected) {
      const start = (data && byId.get(data.center)) || nodes.slice().sort((a, b) => b.degree - a.degree)[0];
      select(start, false); centerOn(start); return;
    }
    const dir = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[key];
    let best = null, bestScore = -Infinity;
    const pool = adj.get(selected.id).size ? Array.from(adj.get(selected.id)).map((id) => byId.get(id)) : nodes.filter((n) => n !== selected);
    pool.forEach((n) => {
      const dx = n.x - selected.x, dy = n.y - selected.y, d = Math.hypot(dx, dy) || 1;
      const cos = (dx * dir[0] + dy * dir[1]) / d;
      if (cos <= 0.2) return;
      const score = cos * 2 - d / 400;
      if (score > bestScore) { bestScore = score; best = n; }
    });
    if (best) { select(best, false); centerOn(best); }
  }

  // --- selection and inspector ------------------------------------------------------

  function select(n, fromPointer) {
    selected = n || null;
    explicit = !!n;
    describe();
    redraw();
    markList();
    if (selected) inspect(selected.id);
    else showHint();
    if (fromPointer && n) tip.hidden = true;
  }
  function clearSelection() {
    selected = null; matches = null; searchInput.value = ""; clear(hitsEl);
    describe(); redraw(); markList(); showHint();
  }
  function showHint() {
    clear(inspectorBody);
    const p = document.createElement("p"); p.className = "muted";
    p.textContent = "Select a node to see what it is, what it connects to, and to open it. Drag to move nodes (they stay pinned until Reset layout); scroll or use + / − to zoom; double-click a node for its local graph.";
    inspectorBody.appendChild(p);
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function el(tag, cls, text, parent) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  let inspectToken = 0;
  async function inspect(id) {
    const token = ++inspectToken;
    let d;
    try {
      const r = await fetch(base + "/node.json?" + new URLSearchParams({ id: id }), { credentials: "same-origin" });
      d = await r.json();
      if (!r.ok) { clear(inspectorBody); el("p", "notice-warn", d.error || "Not found", inspectorBody); return; }
    } catch (e) { return; }
    if (token !== inspectToken) return;
    clear(inspectorBody);
    const head = el("p", "mem-head", null, inspectorBody);
    el("span", "tag", d.type_label, head);
    d.flags.forEach((f) => { if (FLAG_NAMES[f]) el("span", "tag " + (f === "needs_review" || f === "superseded" || f === "dead_end" || f === "source_deleted" ? "tag-warn" : "tag-muted"), FLAG_NAMES[f], head); });
    el("h4", "kg-node-title", d.label, inspectorBody);
    if (d.snippet && d.snippet !== d.label) el("blockquote", "ev-text", d.snippet, inspectorBody);
    const facts = [];
    if (d.meta.pages) facts.push(plural(d.meta.pages, "page"));
    if (d.meta.page) facts.push("page " + d.meta.page);
    if (d.meta.latest_score) facts.push("latest score " + d.meta.latest_score);
    if (d.meta.study) facts.push("quizzed " + plural(d.meta.study.times_seen, "time"));
    if (d.meta.outcomes) Object.keys(d.meta.outcomes).forEach((o) => facts.push(d.meta.outcomes[o] + " " + (FLAG_NAMES[o] || o)));
    if (d.flags.indexOf("superseded") >= 0) facts.push("this answer was corrected — prefer the correction");
    if (facts.length) el("p", "muted", facts.join(" · "), inspectorBody);
    const acts = el("div", "actions", null, inspectorBody);
    d.actions.forEach((a, i) => {
      const link = el("a", "btn btn-sm" + (i === 0 ? " kg-primary" : " btn-ghost"), a.label, acts);
      link.setAttribute("href", a.href);
    });
    const localBtn = el("button", "btn btn-sm btn-ghost", mode === "local" && data && data.center === d.id ? "Local graph (shown)" : "Show local graph", acts);
    localBtn.type = "button";
    localBtn.addEventListener("click", () => focusLocal(d.id));
    const groups = Object.keys(d.connections);
    el("h4", "kg-sub", groups.length ? "Connections" : "No connections", inspectorBody);
    groups.forEach((t) => {
      const g = d.connections[t];
      el("p", "kg-group", PLURALS[t] + " · " + g.count, inspectorBody);
      const ul = el("ul", "mem-edges", null, inspectorBody);
      g.items.forEach((item) => {
        const li = el("li", null, null, ul);
        el("span", "mono rel", item.direction === "out" ? item.relation + " →" : "← " + item.relation, li);
        li.appendChild(document.createTextNode(" "));
        const b = el("button", "fb", item.label, li);
        b.type = "button";
        b.addEventListener("click", () => goTo(item.id));
      });
      if (g.count > g.items.length) el("p", "muted", "…and " + (g.count - g.items.length) + " more", inspectorBody);
    });
  }

  function goTo(id) {  // select a node, loading its local graph if it is not shown
    const n = byId.get(id);
    if (n) { select(n, false); centerOn(n); stage.focus({ preventScroll: true }); return; }
    focusLocal(id);
  }

  function focusLocal(id) {
    mode = "local"; center = id;
    selected = { id: id };
    syncModeButtons();
    syncUrl();
    load();
  }

  // --- search -----------------------------------------------------------------------

  let searchTimer = 0;
  function applySearch() {
    const q = searchInput.value.trim().toLocaleLowerCase();
    matches = q ? new Set(nodes.filter((n) => n.label.toLocaleLowerCase().indexOf(q) >= 0).map((n) => n.id)) : null;
    redraw();
  }
  searchInput.addEventListener("input", () => {
    applySearch();
    clearTimeout(searchTimer);
    searchTimer = setTimeout(findRemote, 180);
  });
  async function findRemote() {
    const q = searchInput.value.trim();
    clear(hitsEl);
    if (!q) return;
    let res;
    try {
      const r = await fetch(base + "/find.json?" + new URLSearchParams({ q: q }), { credentials: "same-origin" });
      res = (await r.json()).results || [];
    } catch (e) { return; }
    if (searchInput.value.trim() !== q) return;
    if (!res.length) { el("li", "muted", "No match", hitsEl); return; }
    res = res.map((n, i) => [n, i]).sort((a, b) => (byId.has(b[0].id) - byId.has(a[0].id)) || a[1] - b[1]).map((x) => x[0]);
    res.slice(0, 12).forEach((n) => {
      const li = el("li", null, null, hitsEl);
      const b = el("button", "fb", n.label + " — " + TYPE_NAMES[n.type] + (byId.has(n.id) ? "" : " (not in this view)"), li);
      b.type = "button";
      b.addEventListener("click", () => { clear(hitsEl); goTo(n.id); });
    });
  }
  $("kg-search").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const first = hitsEl.querySelector("button");
    if (first) first.click();
    else if (matches && matches.size) goTo(Array.from(matches)[0]);
  });
  searchInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && searchInput.value) { ev.preventDefault(); clearSelection(); }
    else if (ev.key === "ArrowDown") { const b = hitsEl.querySelector("button"); if (b) { ev.preventDefault(); b.focus(); } }
  });
  hitsEl.addEventListener("keydown", (ev) => {
    if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp") return;
    const buttons = Array.from(hitsEl.querySelectorAll("button"));
    const i = buttons.indexOf(document.activeElement);
    if (i < 0) return;
    ev.preventDefault();
    if (ev.key === "ArrowUp" && i === 0) { searchInput.focus(); return; }
    buttons[Math.max(0, Math.min(buttons.length - 1, i + (ev.key === "ArrowDown" ? 1 : -1)))].focus();
  });

  // --- accessible list view ----------------------------------------------------------

  function buildList() {
    if (!listBox.open) { clear(listEl); return; }
    clear(listEl);
    const sorted = nodes.slice().sort((a, b) => a.type.localeCompare(b.type) || b.degree - a.degree || a.label.localeCompare(b.label));
    const rels = new Map(nodes.map((n) => [n.id, []]));
    edges.forEach((l) => {
      rels.get(l.s.id).push([l.e.label + " →", l.t]);
      rels.get(l.t.id).push(["← " + l.e.label, l.s]);
    });
    sorted.forEach((n) => {
      const li = el("li", "kg-list-node", null, listEl);
      li.dataset.id = n.id;
      const b = el("button", "fb", TYPE_NAMES[n.type] + ": " + n.label + " (" + plural(n.degree, "connection") + ")", li);
      b.type = "button";
      b.addEventListener("click", () => goTo(n.id));
      const items = rels.get(n.id);
      if (items.length) {
        const ul = el("ul", null, null, li);
        items.slice(0, 25).forEach(([rel, other]) => {
          const r = el("li", null, null, ul);
          el("span", "mono rel", rel, r);
          r.appendChild(document.createTextNode(" "));
          const ob = el("button", "fb", TYPE_NAMES[other.type] + ": " + other.label, r);
          ob.type = "button";
          ob.addEventListener("click", () => goTo(other.id));
        });
        if (items.length > 25) el("li", "muted", "…and " + (items.length - 25) + " more", ul);
      }
    });
    markList();
  }
  function markList() {
    listEl.querySelectorAll(".kg-list-node").forEach((li) => {
      const on = selected && li.dataset.id === selected.id;
      li.classList.toggle("is-selected", !!on);
      const b = li.querySelector("button");
      if (b) b.setAttribute("aria-current", on ? "true" : "false");
    });
  }
  listBox.addEventListener("toggle", buildList);

  // --- controls ---------------------------------------------------------------------

  function syncModeButtons() {
    $("kg-mode-global").setAttribute("aria-pressed", mode === "global" ? "true" : "false");
    $("kg-mode-local").setAttribute("aria-pressed", mode === "local" ? "true" : "false");
    $("kg-depth").hidden = mode !== "local";
  }
  function syncUrl() {
    const url = new URL(window.location.href);
    url.searchParams.set("mode", mode);
    if (mode === "local" && center) url.searchParams.set("node", center); else url.searchParams.delete("node");
    if (mode === "local") url.searchParams.set("depth", String(depth)); else url.searchParams.delete("depth");
    history.replaceState(null, "", url);
  }
  $("kg-mode-global").addEventListener("click", () => { mode = "global"; syncModeButtons(); syncUrl(); load(); });
  $("kg-mode-local").addEventListener("click", () => {
    center = selected ? selected.id : (center || pickCenter());
    mode = "local"; syncModeButtons(); syncUrl(); load();
  });
  document.querySelectorAll("input[name=kg-depth]").forEach((i) => i.addEventListener("change", () => {
    depth = Number(i.value); savePrefs(); syncUrl(); load();
  }));
  document.querySelectorAll("#kg-types input").forEach((i) => i.addEventListener("change", () => { savePrefs(); load(); }));
  $("kg-orphans").addEventListener("change", () => { savePrefs(); load(); });
  $("kg-min").addEventListener("change", () => { savePrefs(); load(); });
  $("kg-concepts").addEventListener("change", () => { savePrefs(); load(); });
  $("kg-labels").addEventListener("change", () => { savePrefs(); redraw(); });
  $("kg-zoom-in").addEventListener("click", () => zoomCenter(1.25));
  $("kg-zoom-out").addEventListener("click", () => zoomCenter(0.8));
  $("kg-fit").addEventListener("click", fit);
  $("kg-reset").addEventListener("click", () => {
    nodes.forEach((n) => { n.fx = null; n.fy = null; n.x = undefined; });
    autoFit = true;
    seed(); warm(); fit();
    say("Layout reset: pinned nodes were released.");
  });

  if (motionQuery) {
    const onMotion = () => { if (reducedMotion() && alpha > 0 && !quiet) settleQuietly(); };
    if (motionQuery.addEventListener) motionQuery.addEventListener("change", onMotion);
    else if (motionQuery.addListener) motionQuery.addListener(onMotion);
  }
  window.addEventListener("resize", () => { resize(); });
  if (window.ResizeObserver) new ResizeObserver(() => resize()).observe(stage);
  readColors();
  resize();
  syncModeButtons();
  showHint();
  load();
  window.LNGraph = {  // for automated checks only (read-only view of the state)
    state: () => ({ mode: mode, pending: pending, center: data && data.center, depth: depth, nodes: nodes.length, edges: edges.length,
                    selected: selected && selected.id, k: view.k, alpha: alpha, truncated: data && data.truncated,
                    matches: matches ? matches.size : null, reducedMotion: reducedMotion(), quiet: quiet,
                    animatedFrames: animatedFrames, quietLayouts: quietLayouts }),
    screenOf: (id) => { const n = byId.get(id); return n ? [n.x * view.k + view.x, n.y * view.k + view.y] : null; },
    pinned: () => nodes.filter((n) => n.fx !== undefined && n.fx !== null).map((n) => n.id),
  };
})();
