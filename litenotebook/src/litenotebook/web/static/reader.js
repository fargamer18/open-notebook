/* LiteNotebook PDF reader.
 * Pages are server-rendered PNGs; a transparent, positioned text layer (from
 * PyMuPDF word boxes) makes text selectable; highlights are overlays in
 * page-relative percentages. All text goes through textContent — never
 * innerHTML. Nothing here touches the embedding model.
 */
(function () {
  "use strict";
  const R = document.getElementById("reader");
  if (!R) return;
  const LN = window.LN || {
    store: { get: (k, d) => d, set: () => {} },
    notify: () => {}, dialogOpen: () => !!document.querySelector("dialog[open]"),
  };
  const nb = R.dataset.notebookId, sid = R.dataset.sourceId;
  const pages = Number(R.dataset.pages);
  const base = "/notebooks/" + nb + "/sources/" + sid;
  const $ = (id) => document.getElementById(id);
  const viewport = $("r-viewport"), box = $("r-pagebox"), img = $("r-img");
  const hl = $("r-hl"), textLayer = $("r-text"), status = $("r-status");
  const pageInput = $("r-page"), zoomLabel = $("r-zoom"), tools = $("r-sel");
  const fitBtn = $("r-fit"), backBtn = $("r-back");
  const SCALES = [1, 1.5, 2, 2.5, 3];
  const SIDE_MIN = 280, SIDE_MAX = 760;
  let page = Math.min(Math.max(1, Number(R.dataset.page) || 1), pages);
  let zoom = Number(R.dataset.zoom) || 1;           // 1 = fit width
  let focus = { kind: R.dataset.focusKind, id: Number(R.dataset.focusId) || 0 };
  let cited = null;   // {page, focus}: where "Back to cited passage" returns
  let ptsWidth = null, words = [], saveTimer = null, selection = null, loadToken = 0;

  function say(text, kind) { status.textContent = text || ""; status.dataset.kind = kind || ""; }
  function typing(el) {
    return el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
  }
  function cssWidth() { return Math.max(200, (viewport.clientWidth - 24) * zoom); }
  function renderScale() {
    if (!ptsWidth) return 1.5;
    const want = (cssWidth() * (window.devicePixelRatio || 1)) / ptsWidth;
    return SCALES.reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a));
  }
  function pageUrl(p) { return base + "/pages/" + p + "/render?scale=" + renderScale(); }
  function post(url, fields) {
    const body = new URLSearchParams();
    for (const k in fields) body.set(k, fields[k]);
    return fetch(url, { method: "POST", body: body, credentials: "same-origin" });
  }

  function place(el, b) {
    el.style.left = (b[0] * 100) + "%";
    el.style.top = (b[1] * 100) + "%";
    el.style.width = ((b[2] - b[0]) * 100) + "%";
    el.style.height = ((b[3] - b[1]) * 100) + "%";
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function reveal(el) {  // centre an overlay inside the page view only
    const r = el.getBoundingClientRect(), v = viewport.getBoundingClientRect();
    viewport.scrollTop += (r.top - v.top) - v.height / 2 + r.height / 2;
  }

  // The header can wrap onto two rows; size the reader to what is left.
  function fitHeight() {
    const head = document.querySelector(".reader-head");
    if (head) document.documentElement.style.setProperty("--reader-head", head.offsetHeight + 6 + "px");
  }
  fitHeight();

  async function loadText(p, token) {
    clear(textLayer);
    words = [];
    try {
      const r = await fetch(base + "/pages/" + p + "/text", { credentials: "same-origin" });
      const data = await r.json();
      if (token !== loadToken) return;
      words = data.words || [];
      const h = box.clientHeight || 1;
      words.forEach((w, i) => {
        const span = document.createElement("span");
        span.textContent = w.t + " ";
        span.dataset.i = String(i);
        span.dataset.l = w.l;
        place(span, w.b);
        span.style.fontSize = Math.max(4, (w.b[3] - w.b[1]) * h * 0.82) + "px";
        textLayer.appendChild(span);
      });
      if (!data.has_text) {
        say("No selectable text on this page" + (data.method === "ocr" ? " (OCR text exists but has no positions)" : "") +
            ". You can still use “Comment on page”.", "info");
      }
    } catch (e) { /* the page image still works without the text layer */ }
  }

  function drawRects(rects, cls, data) {
    const made = [];
    (rects || []).forEach((b) => {
      const d = document.createElement("div");
      d.className = "hl " + cls;
      place(d, b);
      if (data && data.title) d.title = data.title;
      if (data && data.id) d.dataset.annotation = String(data.id);
      hl.appendChild(d);
      made.push(d);
    });
    return made;
  }

  function emphasiseListItem() {
    document.querySelectorAll(".ann.is-focused").forEach((n) => n.classList.remove("is-focused"));
    if (focus.kind !== "annotation") return;
    const item = $("ann-" + focus.id);
    if (item) { item.classList.add("is-focused"); item.scrollIntoView({ block: "nearest" }); }
  }

  async function loadAnnotations(p, token) {
    const r = await fetch(base + "/annotations.json?page=" + p, { credentials: "same-origin" });
    const data = await r.json();
    if (token !== loadToken) return;
    hl.querySelectorAll(".hl-user").forEach((n) => n.remove());
    (data.annotations || []).forEach((a) => {
      const made = drawRects(a.rects, "hl-user hl-" + a.color, { id: a.id, title: a.comment || a.text });
      if (focus.kind === "annotation" && focus.id === a.id && made.length) {
        made.forEach((d) => d.classList.add("is-focused"));
        reveal(made[0]);
        say("Annotation on page " + p, "ok");
      }
    });
    emphasiseListItem();
  }

  function updateBack() {
    backBtn.hidden = !cited || cited.page === page;
  }

  async function applyFocus(p, token) {
    if (!focus.kind || focus.kind === "annotation") return;
    const url = focus.kind === "citation"
      ? "/notebooks/" + nb + "/reader/citations/" + focus.id
      : base + "/chunks/" + focus.id + "/highlight";
    try {
      const r = await fetch(url, { credentials: "same-origin" });
      const t = await r.json();
      if (token !== loadToken) return;
      if (!r.ok) { say(t.error || "Passage unavailable", "warn"); return; }
      if (!t.live) { say(t.reason || "Original source has been deleted", "warn"); return; }
      if (t.page !== p) { goTo(t.page, focus); return; }
      cited = { page: t.page, focus: { kind: focus.kind, id: focus.id } };
      updateBack();
      hl.querySelectorAll(".hl-cite").forEach((n) => n.remove());
      if (t.rects && t.rects.length) {
        const made = drawRects(t.rects, "hl-cite", { title: "Cited passage" });
        reveal(made[0]);
        say((focus.kind === "citation" ? "Cited passage" : "Passage") + " highlighted on page " + p, "ok");
      } else {
        say((t.reason || "Highlight unavailable for this passage") + " — it is on page " + p + ".", "warn");
      }
    } catch (e) { say("Could not locate the passage", "warn"); }
  }

  function preload(p) {
    [p - 1, p + 1].forEach((n) => { if (n >= 1 && n <= pages) { const i = new Image(); i.src = pageUrl(n); } });
  }

  function layout() {
    box.style.width = cssWidth() + "px";
    zoomLabel.textContent = Math.round(zoom * 100) + "%" + (zoom === 1 ? " · fit" : "");
    fitBtn.setAttribute("aria-pressed", zoom === 1 ? "true" : "false");
  }

  function show(p) {
    page = Math.min(Math.max(1, p), pages);
    pageInput.value = String(page);
    const token = ++loadToken;
    layout();
    clear(hl);
    hideTools();
    updateBack();
    say("Rendering page " + page + "…", "busy");
    img.alt = "Page " + page + " of " + pages;
    img.onload = async () => {
      if (token !== loadToken) return;
      if (!ptsWidth) ptsWidth = img.naturalWidth / 1.5;
      say("", "");
      await loadText(page, token);
      await loadAnnotations(page, token);
      await applyFocus(page, token);
      preload(page);
    };
    img.onerror = () => { if (token === loadToken) say("Render failed — page " + page + " is unavailable. Try another page or reload.", "error"); };
    img.src = pageUrl(page);
    scheduleSave();
  }

  function goTo(p, newFocus) {
    focus = newFocus || { kind: "", id: 0 };
    show(p);
    viewport.scrollTop = 0;
  }

  function setZoom(z) {
    zoom = Math.min(4, Math.max(0.25, Math.round(z * 100) / 100));
    show(page);
  }

  function scheduleSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => post(base + "/reading-state", { page: page, zoom: zoom }), 1200);
  }

  // --- selection → highlight / annotate / add to note ---------------------------

  function hideTools() {
    tools.hidden = true;
    $("r-comment-box").hidden = true;
    $("r-sel-error").hidden = true;
    clear($("r-sel-note"));
    selection = null;
  }

  function currentSelection() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
    const range = sel.getRangeAt(0);
    if (!textLayer.contains(range.commonAncestorContainer)) return null;
    const chosen = Array.from(textLayer.children).filter((s) => range.intersectsNode(s));
    if (!chosen.length) return null;
    const lines = {};
    chosen.forEach((s) => {
      const b = words[Number(s.dataset.i)].b;
      const l = lines[s.dataset.l];
      lines[s.dataset.l] = l ? [Math.min(l[0], b[0]), Math.min(l[1], b[1]), Math.max(l[2], b[2]), Math.max(l[3], b[3])] : b.slice();
    });
    const text = chosen.map((s) => words[Number(s.dataset.i)].t).join(" ");
    return { text: text.slice(0, 5000), rects: Object.values(lines), anchor: range.getBoundingClientRect() };
  }

  document.addEventListener("mouseup", (event) => {
    if (tools.contains(event.target)) return;
    const s = currentSelection();
    if (!s) { if (!tools.contains(document.activeElement)) hideTools(); return; }
    selection = s;
    tools.hidden = false;
    tools.style.left = Math.min(window.innerWidth - 340, Math.max(8, s.anchor.left)) + "px";
    tools.style.top = Math.min(window.innerHeight - 60, s.anchor.bottom + 8) + "px";
  });

  async function save(kind, color, comment) {
    if (!selection && kind !== "comment") return null;
    const r = await post(base + "/annotations", {
      page: page, kind: kind, color: color || "yellow", comment: comment || "",
      text: selection ? selection.text : "", rects: JSON.stringify(selection ? selection.rects : []),
    });
    const data = await r.json();
    if (!r.ok) { const e = $("r-sel-error"); e.textContent = data.error || "Not saved"; e.hidden = false; tools.hidden = false; return null; }
    if (window.htmx) window.htmx.trigger(document.body, "annotations-changed");
    loadAnnotations(page, loadToken);
    LN.notify("success", kind === "comment" ? "Annotation created" : "Highlight created (" + (color || "yellow") + ")");
    return data;
  }

  tools.querySelectorAll("[data-color]").forEach((b) => b.addEventListener("click", async () => {
    if (await save("highlight", b.dataset.color)) { window.getSelection().removeAllRanges(); hideTools(); viewport.focus(); }
  }));
  $("r-annotate").addEventListener("click", () => { $("r-comment-box").hidden = false; $("r-comment").focus(); });
  $("r-comment-save").addEventListener("click", async () => {
    const text = $("r-comment").value.trim();
    if (!text) { $("r-comment").focus(); return; }
    if (await save("comment", "yellow", text)) { $("r-comment").value = ""; hideTools(); viewport.focus(); }
  });
  $("r-to-note").addEventListener("click", async () => {
    const a = await save("highlight", "yellow");
    if (a && window.htmx) {
      window.htmx.ajax("GET", "/notebooks/" + nb + "/notes/add-form?kind=annotation&id=" + a.id,
                       { target: "#r-sel-note", swap: "innerHTML" });
      tools.hidden = false;
    }
  });
  $("r-page-comment").addEventListener("click", () => {
    selection = null;
    tools.hidden = false;
    tools.style.left = "40%"; tools.style.top = "120px";
    $("r-comment-box").hidden = false;
    $("r-comment").focus();
  });

  // --- controls -----------------------------------------------------------------

  $("r-prev").addEventListener("click", () => goTo(page - 1));
  $("r-next").addEventListener("click", () => goTo(page + 1));
  pageInput.addEventListener("change", () => goTo(Number(pageInput.value) || page));
  pageInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); goTo(Number(pageInput.value) || page); } });
  $("r-zoom-in").addEventListener("click", () => setZoom(zoom + 0.15));
  $("r-zoom-out").addEventListener("click", () => setZoom(zoom - 0.15));
  fitBtn.addEventListener("click", () => setZoom(1));
  backBtn.addEventListener("click", () => { if (cited) goTo(cited.page, cited.focus); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !tools.hidden) { hideTools(); viewport.focus(); return; }
    if (LN.dialogOpen() || typing(document.activeElement)) return;
    const t = e.target;
    if (t && t.closest && t.closest("[role=tablist], [role=separator]")) return;  // own arrow keys
    if (e.altKey || e.metaKey) return;
    if (!e.ctrlKey && (e.key === "PageDown" || e.key === "ArrowRight")) { e.preventDefault(); goTo(page + 1); }
    else if (!e.ctrlKey && (e.key === "PageUp" || e.key === "ArrowLeft")) { e.preventDefault(); goTo(page - 1); }
    else if (e.ctrlKey && (e.key === "+" || e.key === "=")) { e.preventDefault(); setZoom(zoom + 0.15); }
    else if (e.ctrlKey && e.key === "-") { e.preventDefault(); setZoom(zoom - 0.15); }
  });

  // in-place navigation from annotation list / evidence "Open in PDF"
  document.addEventListener("click", (e) => {
    const a = e.target.closest && e.target.closest("[data-reader-page]");
    if (!a || a.dataset.readerSource !== sid) return;
    e.preventDefault();
    const f = a.dataset.readerAnnotation ? { kind: "annotation", id: Number(a.dataset.readerAnnotation) }
      : a.dataset.readerCitation ? { kind: "citation", id: Number(a.dataset.readerCitation) }
      : a.dataset.readerChunk ? { kind: "chunk", id: Number(a.dataset.readerChunk) } : null;
    goTo(Number(a.dataset.readerPage), f);
    viewport.focus();
  });
  document.body.addEventListener("annotations-changed", () => loadAnnotations(page, loadToken));

  // --- side panel: tabs (remembered per browser) --------------------------------

  const tabs = Array.from(document.querySelectorAll(".reader-tabs [role=tab]"));
  const panel = $("reader-panel");
  function selectTab(tab, load) {
    tabs.forEach((x) => {
      const on = x === tab;
      x.setAttribute("aria-selected", on ? "true" : "false");
      x.tabIndex = on ? 0 : -1;
      x.classList.toggle("is-on", on);
    });
    panel.setAttribute("aria-labelledby", tab.id);
    LN.store.set("reader:tab", tab.dataset.panel);
    if (load && window.htmx) window.htmx.ajax("GET", tab.getAttribute("hx-get"), { target: "#reader-panel", swap: "innerHTML" });
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => selectTab(tab, false));  // htmx loads it
    tab.addEventListener("keydown", (e) => {
      let next = null;
      if (e.key === "ArrowRight") next = tabs[(i + 1) % tabs.length];
      else if (e.key === "ArrowLeft") next = tabs[(i - 1 + tabs.length) % tabs.length];
      else if (e.key === "Home") next = tabs[0];
      else if (e.key === "End") next = tabs[tabs.length - 1];
      if (next) { e.preventDefault(); next.focus(); selectTab(next, true); }
    });
  });
  const startTab = tabs.find((t) => t.dataset.panel === LN.store.get("reader:tab", "annotations")) || tabs[0];
  setTimeout(() => selectTab(startTab, true), 0);  // after htmx has initialised
  document.addEventListener("htmx:afterSwap", (e) => { if (e.target === panel) emphasiseListItem(); });

  // --- side panel: resizable split (width remembered per browser) ---------------

  const handle = $("r-resize");
  function setSide(w, persist) {
    w = Math.round(Math.min(SIDE_MAX, Math.max(SIDE_MIN, w)));
    R.style.setProperty("--side", w + "px");
    handle.setAttribute("aria-valuenow", String(w));
    if (persist) LN.store.set("reader:side", w);
  }
  const savedWidth = Number(LN.store.get("reader:side", 0)) ||
    (() => { try { return Number(localStorage.getItem("ln-reader-side")) || 0; } catch (e) { return 0; } })();
  if (savedWidth) setSide(savedWidth, false);
  handle.addEventListener("pointerdown", (e) => {
    handle.setPointerCapture(e.pointerId);
    const move = (ev) => setSide(R.getBoundingClientRect().right - ev.clientX, false);
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      setSide(parseInt(getComputedStyle(R).getPropertyValue("--side"), 10), true);
      show(page);
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
  });
  handle.addEventListener("keydown", (e) => {
    const now = parseInt(getComputedStyle(R).getPropertyValue("--side"), 10) || 400;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      setSide(now + (e.key === "ArrowLeft" ? 32 : -32), true);
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => show(page), 250);
    }
  });
  let resizeTimer = null;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { fitHeight(); show(page); }, 200); });

  show(page);
})();
