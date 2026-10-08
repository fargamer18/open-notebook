/* LiteNotebook memory graph: a bounded local subgraph drawn as plain SVG.
 * Deterministic radial layout (centre, depth-1 ring, depth-2 ring); all text
 * goes through textContent — never innerHTML.
 */
(function () {
  "use strict";
  const root = document.getElementById("graph");
  const svg = document.getElementById("graph-svg");
  if (!root || !svg) return;
  const NS = "http://www.w3.org/2000/svg";
  const nb = root.dataset.notebookId;
  const statusEl = document.getElementById("graph-status");
  const hits = document.getElementById("graph-hits");
  const W = 900, H = 620, CX = W / 2, CY = H / 2, R1 = 170, R2 = 285;
  let center = Number(root.dataset.center);
  let depth = Number(root.dataset.depth) || 1;
  let selected = null;
  let scale = 1;  // view zoom; the layout itself never changes
  function applyView() {
    const w = W / scale, h = H / scale;
    svg.setAttribute("viewBox", (CX - w / 2) + " " + (CY - h / 2) + " " + w + " " + h);
  }
  applyView();
  const zoomBy = (f) => { scale = Math.min(3, Math.max(0.6, scale * f)); applyView(); };
  const zin = document.getElementById("graph-zoom-in"), zout = document.getElementById("graph-zoom-out");
  const zreset = document.getElementById("graph-zoom-reset");
  if (zin) zin.addEventListener("click", () => zoomBy(1.25));
  if (zout) zout.addEventListener("click", () => zoomBy(0.8));
  if (zreset) zreset.addEventListener("click", () => { scale = 1; applyView(); });
  const activate = (ev, fn) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); fn(); } };

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function types() {
    return Array.from(document.querySelectorAll("#graph-types input:checked")).map((i) => i.value);
  }

  function inspect(path) {
    if (window.htmx) window.htmx.ajax("GET", path, { target: "#graph-inspector", swap: "innerHTML" });
  }

  function layout(data) {
    const pos = {};
    const byDepth = { 0: [], 1: [], 2: [] };
    data.nodes.forEach((n) => byDepth[n.depth].push(n));
    byDepth[0].forEach((n) => { pos[n.id] = [CX, CY, 0]; });
    byDepth[1].forEach((n, i) => {
      const a = (2 * Math.PI * i) / Math.max(1, byDepth[1].length) - Math.PI / 2;
      pos[n.id] = [CX + R1 * Math.cos(a), CY + R1 * Math.sin(a), a];
    });
    const parentAngle = (n) => {
      const e = data.edges.find((e) => (e.from === n.id && pos[e.to] && pos[e.to][0] !== CX) ||
                                       (e.to === n.id && pos[e.from] && pos[e.from][0] !== CX));
      if (!e) return 10;
      const p = pos[e.from === n.id ? e.to : e.from];
      return p ? p[2] : 10;
    };
    byDepth[2].sort((a, b) => parentAngle(a) - parentAngle(b) || a.id - b.id).forEach((n, i) => {
      const a = (2 * Math.PI * i) / Math.max(1, byDepth[2].length) - Math.PI / 2;
      pos[n.id] = [CX + R2 * Math.cos(a), CY + R2 * Math.sin(a), a];
    });
    return pos;
  }

  function render(data) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const pos = layout(data);
    const edges = el("g", { class: "g-edges" }, svg);
    data.edges.forEach((e) => {
      const a = pos[e.from], b = pos[e.to];
      if (!a || !b) return;
      const g = el("g", { class: "g-edge rel-" + e.relation, tabindex: "0", role: "button",
                          "aria-label": "Link: " + e.relation.replace(/_/g, " ") }, edges);
      el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: "g-line" }, g);
      el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: "g-hit" }, g);
      const t = el("title", {}, g); t.textContent = e.relation;
      const open = () => inspect("/notebooks/" + nb + "/memory/edges/" + e.id);
      g.addEventListener("click", open);
      g.addEventListener("keydown", (ev) => activate(ev, open));
    });
    const nodes = el("g", { class: "g-nodes" }, svg);
    data.nodes.forEach((n) => {
      const p = pos[n.id];
      const g = el("g", { class: "g-node t-" + n.type + (n.id === center ? " is-center" : "") +
                          (n.id === selected ? " is-selected" : ""),
                          transform: "translate(" + p[0] + "," + p[1] + ")", tabindex: "0", role: "button",
                          "aria-label": (n.id === center ? "Centre: " : "") + n.type.replace(/_/g, " ") + ": " + n.label,
                          "aria-pressed": n.id === selected ? "true" : "false" }, nodes);
      el("circle", { r: n.id === center ? 13 : (n.depth === 1 ? 9 : 7) }, g);
      const label = el("text", { y: n.id === center ? 28 : 21, "text-anchor": "middle" }, g);
      label.textContent = n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label;
      const tip = el("title", {}, g);
      tip.textContent = n.type.replace("_", " ") + ": " + n.label + " (" + n.provenance.replace("_", " ") + ")";
      const choose = () => {
        if (selected === n.id && n.id !== center) { recenter(n.id); return; }
        selected = n.id;
        inspect("/notebooks/" + nb + "/memory/nodes/" + n.id);
        svg.querySelectorAll(".g-node.is-selected").forEach((x) => { x.classList.remove("is-selected"); x.setAttribute("aria-pressed", "false"); });
        g.classList.add("is-selected");
        g.setAttribute("aria-pressed", "true");
        selectedLabel = n.label;
        describe();
      };
      g.addEventListener("click", choose);
      g.addEventListener("keydown", (ev) => activate(ev, choose));
    });
    const c = data.nodes.find((n) => n.id === center);
    centerLabel = c ? c.label : "";
    const plural = (n, word) => n + " " + word + (n === 1 ? "" : "s");
    summary = plural(data.nodes.length, "node") + ", " + plural(data.edges.length, "link") + " · depth " +
      data.depth + (data.truncated ? " · limited to " + data.limits.nodes + " nodes / " +
      data.limits.edges + " links" : "");
    if (!data.nodes.length) summary = "Nothing to show with these filters — tick more node types";
    describe();
  }

  let centerLabel = "", selectedLabel = "", summary = "";
  function describe() {
    statusEl.textContent = (centerLabel ? "Centred on “" + centerLabel + "”. " : "") +
      (selectedLabel && selectedLabel !== centerLabel ? "Selected “" + selectedLabel + "”. " : "") +
      summary + ". Select a node to inspect it; select it again to centre on it.";
  }

  async function load() {
    statusEl.textContent = "Loading graph…";
    const q = new URLSearchParams({ node: String(center), depth: String(depth), types: types().join(",") });
    try {
      const r = await fetch("/notebooks/" + nb + "/memory/graph.json?" + q.toString(), { credentials: "same-origin" });
      const data = await r.json();
      if (!r.ok) { statusEl.textContent = data.error || "Could not load the graph"; return; }
      render(data);
    } catch (e) {
      statusEl.textContent = "Could not load the graph";
    }
  }

  function recenter(id) {
    center = id;
    selected = id;
    const url = new URL(window.location.href);
    url.searchParams.set("node", String(id));
    url.searchParams.set("depth", String(depth));
    history.replaceState(null, "", url);
    inspect("/notebooks/" + nb + "/memory/nodes/" + id);
    load();
  }

  document.querySelectorAll("input[name=depth]").forEach((i) => i.addEventListener("change", () => {
    depth = Number(i.value); load();
  }));
  document.querySelectorAll("#graph-types input").forEach((i) => i.addEventListener("change", load));
  document.getElementById("graph-search").addEventListener("submit", async (event) => {
    event.preventDefault();
    const q = document.getElementById("graph-q").value.trim();
    while (hits.firstChild) hits.removeChild(hits.firstChild);
    if (!q) return;
    const r = await fetch("/notebooks/" + nb + "/memory/graph/search?" + new URLSearchParams({ q: q }),
                          { credentials: "same-origin" });
    const data = await r.json();
    if (!data.results || !data.results.length) {
      const li = document.createElement("li"); li.className = "muted"; li.textContent = "No match."; hits.appendChild(li);
      return;
    }
    data.results.forEach((n) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button"; b.className = "fb";
      b.textContent = n.label + " (" + n.type.replace("_", " ") + ")";
      b.addEventListener("click", () => { while (hits.firstChild) hits.removeChild(hits.firstChild); recenter(n.id); });
      li.appendChild(b); hits.appendChild(li);
    });
  });
  inspect("/notebooks/" + nb + "/memory/nodes/" + center);
  load();
})();
