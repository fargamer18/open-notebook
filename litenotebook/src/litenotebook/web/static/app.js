/* LiteNotebook shell: toasts, confirmation dialog, command palette, global
 * shortcuts and small shared helpers (window.LN).
 *
 * All text is set with textContent — never innerHTML. Nothing here calls the
 * LLM or the embedding model: the palette reads /palette.json (SQLite only).
 * localStorage holds harmless UI preferences only and may be unavailable.
 */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);

  // --- helpers -------------------------------------------------------------

  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem("ln:" + key); return v === null ? fallback : JSON.parse(v); }
      catch (e) { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem("ln:" + key, JSON.stringify(value)); } catch (e) { /* ignore */ }
    },
  };

  function isTyping(el) {
    return !!el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
  }

  function dialogOpen() { return !!document.querySelector("dialog[open]"); }

  function post(url, fields) {
    const body = new URLSearchParams();
    for (const k in fields) body.set(k, fields[k]);
    return fetch(url, { method: "POST", body: body, credentials: "same-origin" });
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  // --- toasts ----------------------------------------------------------------

  const TOAST_MS = { success: 4000, info: 5000, warn: 8000, error: 10000 };

  function arm(toast) {
    const kind = (toast.className.match(/toast-(\w+)/) || [])[1] || "info";
    if (!toast.querySelector(".toast-close")) {
      const close = document.createElement("button");
      close.type = "button";
      close.className = "toast-close";
      close.setAttribute("aria-label", "Dismiss notification");
      close.textContent = "×";
      close.addEventListener("click", () => toast.remove());
      toast.appendChild(close);
    }
    let timer = null;
    const start = () => { timer = setTimeout(() => toast.remove(), TOAST_MS[kind] || 5000); };
    const stop = () => clearTimeout(timer);
    toast.addEventListener("mouseenter", stop);
    toast.addEventListener("mouseleave", start);
    toast.addEventListener("focusin", stop);
    toast.addEventListener("focusout", start);
    start();
  }

  function notify(kind, message) {
    const box = $("toasts");
    if (!box || !message) return;
    kind = ["success", "info", "warn", "error"].includes(kind) ? kind : "info";
    const toast = document.createElement("div");
    toast.className = "toast toast-" + kind;
    if (kind === "error") toast.setAttribute("role", "alert");
    const text = document.createElement("span");
    text.className = "toast-text";
    text.textContent = String(message).slice(0, 300);
    toast.appendChild(text);
    box.appendChild(toast);
    while (box.children.length > 4) box.removeChild(box.firstChild);
    arm(toast);
  }

  document.addEventListener("htmx:beforeSwap", (event) => {
    const xhr = event.detail && event.detail.xhr;
    const raw = xhr && xhr.getResponseHeader("X-LN-Notify");
    if (!raw) return;
    try { const n = JSON.parse(raw); notify(n.kind, n.message); } catch (e) { /* ignore */ }
  });
  document.addEventListener("htmx:sendError", () => notify("error", "Could not reach LiteNotebook — is the server running?"));
  document.addEventListener("htmx:responseError", (event) => {
    const status = event.detail && event.detail.xhr ? event.detail.xhr.status : 0;
    if (status >= 500) notify("error", "Something went wrong (" + status + "). Details are in the server log.");
    else if (status === 404) notify("warn", "That item no longer exists.");
    else if (status === 403) notify("error", "That request was not allowed.");
  });

  // --- dialogs (native <dialog>: focus trap, Escape, inert page) -------------

  function openDialog(dialog, from) {
    if (!dialog || dialog.open) return;
    dialog.lnReturnFocus = from || document.activeElement;  // per dialog
    dialog.showModal();
  }

  document.addEventListener("click", (event) => {
    const closer = event.target.closest("[data-dialog-close]");
    if (closer) { const d = closer.closest("dialog"); if (d) d.close(); return; }
    // a click on the backdrop (outside the dialog box) closes it
    if (event.target.tagName === "DIALOG" && event.target.open) {
      const r = event.target.getBoundingClientRect();
      const inside = event.clientX >= r.left && event.clientX <= r.right && event.clientY >= r.top && event.clientY <= r.bottom;
      if (!inside) event.target.close();
    }
  });

  document.addEventListener("close", (event) => {
    if (event.target.tagName !== "DIALOG") return;
    const back = event.target.lnReturnFocus;
    event.target.lnReturnFocus = null;
    if (dialogOpen()) return;  // another dialog took over (palette → shortcuts)
    if (back && back.isConnected && back !== document.body && !back.closest("dialog")) back.focus();
    else { const main = $("main"); if (main) main.focus(); }
  }, true);

  // --- destructive confirmation --------------------------------------------

  const confirmDialog = $("confirm-dialog");
  let pending = null;

  document.addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-confirm-action]");
    if (!trigger || !confirmDialog) return;
    event.preventDefault();
    const d = trigger.dataset;
    pending = { trigger: trigger, action: d.confirmAction, target: d.confirmTarget || "", swap: d.confirmSwap || "outerHTML" };
    $("confirm-dialog-title").textContent = d.confirmTitle || "Delete this item?";
    $("confirm-dialog-item").textContent = d.confirmItem || "";
    $("confirm-dialog-item").hidden = !d.confirmItem;
    $("confirm-dialog-body").textContent = d.confirmBody || "This cannot be undone.";
    $("confirm-dialog-submit").textContent = d.confirmLabel || "Delete";
    $("confirm-dialog-value").value = d.confirmValue || "";
    $("confirm-dialog-error").hidden = true;
    $("confirm-dialog-form").action = d.confirmAction;
    openDialog(confirmDialog, trigger);
  });

  if (confirmDialog) {
    $("confirm-dialog-form").addEventListener("submit", (event) => {
      if (!pending) { event.preventDefault(); return; }
      const p = pending;
      if (!p.target || !window.htmx) return; // plain form post; the server redirects
      event.preventDefault();
      const submit = $("confirm-dialog-submit");
      submit.disabled = true;
      window.htmx.ajax("POST", p.action, {
        target: p.target, swap: p.swap, values: { confirm: $("confirm-dialog-value").value },
      }).then(() => {
        submit.disabled = false;
        pending = null;
        confirmDialog.close();
      }, () => {
        submit.disabled = false;
        const e = $("confirm-dialog-error");
        e.textContent = "Could not delete — check that LiteNotebook is still running.";
        e.hidden = false;
      });
    });
    confirmDialog.addEventListener("close", () => { pending = null; $("confirm-dialog-submit").disabled = false; });
  }

  // --- command palette --------------------------------------------------------

  const palette = $("palette");
  const pq = $("palette-q"), plist = $("palette-list"), pcount = $("palette-count");
  let commands = [], shown = [], active = 0;

  function matches(cmd, tokens) {
    const hay = (cmd.label + " " + (cmd.keywords || "") + " " + cmd.group).toLowerCase();
    return tokens.every((t) => hay.includes(t));
  }

  function rank(cmd, query) {
    const label = cmd.label.toLowerCase();
    if (!query) return 0;
    if (label.startsWith(query)) return 0;
    if (label.split(/[\s:·]+/).some((w) => w.startsWith(query))) return 1;
    if (label.includes(query)) return 2;
    return 3;
  }

  function filterCommands(query) {
    const q = query.trim().toLowerCase();
    const tokens = q.split(/\s+/).filter(Boolean);
    return commands
      .map((c, i) => ({ c: c, i: i }))
      .filter((x) => matches(x.c, tokens))
      .sort((a, b) => rank(a.c, q) - rank(b.c, q) || a.i - b.i)
      .slice(0, 60)
      .map((x) => x.c);
  }

  function renderPalette() {
    shown = filterCommands(pq.value);
    active = Math.min(active, Math.max(0, shown.length - 1));
    clear(plist);
    let group = null;
    shown.forEach((cmd, i) => {
      if (cmd.group !== group) {
        group = cmd.group;
        const h = document.createElement("li");
        h.className = "palette-group";
        h.setAttribute("role", "presentation");
        h.textContent = group;
        plist.appendChild(h);
      }
      const li = document.createElement("li");
      li.id = "palette-opt-" + i;
      li.className = "palette-opt" + (i === active ? " is-active" : "");
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", i === active ? "true" : "false");
      li.textContent = cmd.label;
      li.addEventListener("mousedown", (e) => e.preventDefault());
      li.addEventListener("click", () => run(cmd));
      plist.appendChild(li);
    });
    if (!shown.length) {
      const li = document.createElement("li");
      li.className = "palette-empty";
      li.setAttribute("role", "presentation");
      li.textContent = commands.length ? "No matching command." : "Loading…";
      plist.appendChild(li);
    }
    pq.setAttribute("aria-activedescendant", shown.length ? "palette-opt-" + active : "");
    pcount.textContent = shown.length + " command" + (shown.length === 1 ? "" : "s");
    const current = $("palette-opt-" + active);
    if (current) current.scrollIntoView({ block: "nearest" });
  }

  function run(cmd) {
    palette.close();
    if (cmd.action === "shortcuts") { openDialog($("shortcuts")); return; }
    if (cmd.url) window.location.href = cmd.url;
  }

  async function openPalette(from) {
    if (!palette) return;
    if (dialogOpen() && !palette.open) return;
    pq.value = "";
    active = 0;
    commands = [];
    openDialog(palette, from);
    pq.focus();
    renderPalette();
    const params = new URLSearchParams();
    if (document.body.dataset.notebookId) params.set("nb", document.body.dataset.notebookId);
    if (document.body.dataset.sourceId) params.set("source", document.body.dataset.sourceId);
    try {
      const r = await fetch("/palette.json?" + params.toString(), { credentials: "same-origin" });
      commands = (await r.json()).commands || [];
    } catch (e) {
      commands = [];
      notify("error", "Could not load commands.");
    }
    if (palette.open) renderPalette();
  }

  if (palette) {
    pq.addEventListener("input", () => { active = 0; renderPalette(); });
    pq.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { e.preventDefault(); active = Math.min(active + 1, shown.length - 1); renderPalette(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); active = Math.max(active - 1, 0); renderPalette(); }
      else if (e.key === "Home" && e.ctrlKey) { e.preventDefault(); active = 0; renderPalette(); }
      else if (e.key === "End" && e.ctrlKey) { e.preventDefault(); active = Math.max(0, shown.length - 1); renderPalette(); }
      else if (e.key === "Enter") { e.preventDefault(); if (shown[active]) run(shown[active]); }
    });
  }

  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-palette-open]")) openPalette(event.target.closest("button"));
    else if (event.target.closest("[data-shortcuts-open]")) openDialog($("shortcuts"), event.target.closest("button"));
    else if (event.target.closest("[data-history-back]")) {
      const a = event.target.closest("[data-history-back]");
      if (window.history.length > 1 && document.referrer.startsWith(window.location.origin)) {
        event.preventDefault();
        window.history.back();
      } else if (a.tagName !== "A") { window.location.href = "/"; }
    }
  });

  // --- global keyboard shortcuts --------------------------------------------

  function visible(el) { return !!el && !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length); }

  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && (e.key === "k" || e.key === "K")) {
      e.preventDefault();
      if (palette && palette.open) palette.close(); else openPalette();
      return;
    }
    if (e.defaultPrevented || dialogOpen() || isTyping(document.activeElement)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "?") { e.preventDefault(); openDialog($("shortcuts")); }
    else if (e.key === "/") {
      const field = Array.from(document.querySelectorAll("[data-slash-focus]")).find(visible);
      if (field) { e.preventDefault(); field.focus(); if (field.select) field.select(); }
    } else if (e.key === "n") {
      const field = Array.from(document.querySelectorAll("[data-n-focus]")).find(visible);
      if (field) { e.preventDefault(); field.focus(); }
    }
  });

  // --- result lists: Up/Down between results, Enter opens --------------------

  document.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    if (dialogOpen() || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    const el = document.activeElement;
    const list = document.querySelector("[data-result-nav]");
    if (!list || !el) return;
    const links = Array.from(list.querySelectorAll("[data-result]")).filter(visible);
    if (!links.length) return;
    const fromSearch = el.matches && el.matches("[data-result-input]");
    const index = links.indexOf(el);
    if (!fromSearch && index < 0) return;
    e.preventDefault();
    let next;
    if (fromSearch) next = e.key === "ArrowDown" ? 0 : -1;
    else next = index + (e.key === "ArrowDown" ? 1 : -1);
    if (next < 0) { const input = document.querySelector("[data-result-input]"); if (input) input.focus(); return; }
    links[Math.min(next, links.length - 1)].focus();
  });

  // --- copy text --------------------------------------------------------------

  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; }
    catch (e) {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.className = "sr-only";
      document.body.appendChild(area);
      area.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
      area.remove();
      return ok;
    }
  }

  document.addEventListener("click", async (event) => {
    const b = event.target.closest("[data-copy-target]");
    if (!b) return;
    const source = $(b.dataset.copyTarget) || (b.closest("article") && b.closest("article").querySelector(b.dataset.copyTarget));
    if (!source) return;
    const text = "value" in source && source.tagName === "TEXTAREA" ? source.value : source.textContent;
    const ok = await copyText((text || "").trim());
    notify(ok ? "success" : "error", ok ? (b.dataset.copyDone || "Copied to clipboard") : "Copy failed — select the text and copy it manually.");
  });

  // --- collapsible side panels (state persisted per browser) ---------------

  function applyCollapse(button, collapsed) {
    const panel = $(button.getAttribute("aria-controls"));
    button.setAttribute("aria-expanded", collapsed ? "false" : "true");
    const holder = button.closest("[data-collapse-root]");
    if (holder) holder.classList.toggle("is-collapsed", collapsed);
    if (panel) panel.hidden = collapsed;
  }

  function setupCollapsers(root) {
    (root || document).querySelectorAll("[data-collapse-key]").forEach((b) => {
      if (b.dataset.collapseReady) return;
      b.dataset.collapseReady = "1";
      applyCollapse(b, !!store.get("collapsed:" + b.dataset.collapseKey, false));
      b.addEventListener("click", () => {
        const collapsed = b.getAttribute("aria-expanded") === "true";
        store.set("collapsed:" + b.dataset.collapseKey, collapsed);
        applyCollapse(b, collapsed);
      });
    });
  }

  // --- long synchronous forms (export, import, verify, restore) -------------

  document.addEventListener("submit", (event) => {
    const form = event.target;
    if (!form.matches || !form.matches("[data-busy-form]") || event.defaultPrevented) return;
    const button = form.querySelector("button[type=submit]");
    const label = form.querySelector(".busy-label");
    if (label) label.hidden = false;
    if (button) {
      // Defer: disabling synchronously would drop the button from the submission.
      setTimeout(() => { button.disabled = true; }, 0);
      // A download leaves the page in place: re-enable after a while.
      setTimeout(() => { button.disabled = false; if (label) label.hidden = true; }, 15000);
    }
  });
  window.addEventListener("pageshow", () => {
    document.querySelectorAll("[data-busy-form] button[type=submit]").forEach((b) => { b.disabled = false; });
    document.querySelectorAll("[data-busy-form] .busy-label").forEach((l) => { l.hidden = true; });
  });

  // --- boot -----------------------------------------------------------------

  function boot() {
    document.querySelectorAll("#toasts [data-toast]").forEach(arm);
    setupCollapsers(document);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
  document.addEventListener("htmx:load", (event) => setupCollapsers(event.target));

  window.LN = { store: store, isTyping: isTyping, dialogOpen: dialogOpen, notify: notify, post: post, clear: clear, copyText: copyText, openDialog: openDialog };
})();
