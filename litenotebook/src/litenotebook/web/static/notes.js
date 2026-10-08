/* LiteNotebook notes: debounced autosave with visible state.
 * Never sends a request per keystroke; never discards text on failure.
 */
(function () {
  "use strict";
  function setup(form) {
  if (!form || form.dataset.ready) return;
  form.dataset.ready = "1";
  const url = form.dataset.autosaveUrl;
  const title = form.elements.title;
  const content = form.elements.content;
  const revision = form.elements.revision;
  const state = form.querySelector("#save-state") || document.getElementById("save-state");
  const updated = document.getElementById("note-updated");
  const conflictBox = document.getElementById("note-conflict");
  const DEBOUNCE_MS = 800;
  let timer = null, saving = false, dirty = false, failed = false, conflict = false, retryMs = 2000;
  let explicit = false, leaving = false;
  const toast = (kind, text) => { if (window.LN) window.LN.notify(kind, text); };

  function show(kind, text) {
    state.dataset.state = kind;
    state.textContent = text;
  }

  function schedule(delay) {
    if (conflict) return;
    clearTimeout(timer);
    timer = setTimeout(save, delay);
  }

  async function save() {
    if (!saving && !conflict && !dirty && explicit) { explicit = false; show("saved", "Saved"); toast("success", "Note saved"); return; }
    if (saving || conflict || !dirty) return;
    saving = true;
    dirty = false;
    show("saving", "Saving…");
    const body = new URLSearchParams();
    body.set("title", title.value);
    body.set("content", content.value);
    body.set("revision", revision.value);
    try {
      const r = await fetch(url, { method: "POST", body: body, credentials: "same-origin" });
      let data = {};
      try { data = await r.json(); } catch (e) { /* not JSON */ }
      if (r.ok && data.ok) {
        revision.value = data.revision;
        if (updated && data.saved_at) updated.textContent = data.saved_at.slice(0, 19).replace("T", " ");
        failed = false;
        retryMs = 2000;
        show(dirty ? "dirty" : "saved", dirty ? "Unsaved changes" : "Saved");
        if (explicit) { explicit = false; toast("success", "Note saved"); }
      } else if (r.status === 409) {
        conflict = true;
        dirty = true;
        show("error", "Not saved — changed elsewhere");
        if (conflictBox) conflictBox.hidden = false;
      } else {
        dirty = true;
        failed = r.status >= 500 || r.status === 0;
        show("error", "Not saved: " + (data.error || ("HTTP " + r.status)));
        if (explicit) { explicit = false; toast("error", "Note not saved: " + (data.error || ("HTTP " + r.status))); }
        if (failed) { schedule(retryMs); retryMs = Math.min(retryMs * 2, 30000); }
      }
    } catch (e) {
      dirty = true;
      failed = true;
      show("error", "Not saved (connection problem) — retrying…");
      schedule(retryMs);
      retryMs = Math.min(retryMs * 2, 30000);
    } finally {
      saving = false;
      if (dirty && !conflict && !failed) schedule(DEBOUNCE_MS);
    }
  }

  function edited() {
    dirty = true;
    if (!conflict) show("dirty", "Unsaved changes");
    schedule(DEBOUNCE_MS);
  }

  title.addEventListener("input", edited);
  content.addEventListener("input", edited);
  form.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "s") {
      event.preventDefault();
      clearTimeout(timer);
      explicit = true;
      save();
    }
  });
  // With JavaScript, "Save now" saves in place (the plain POST is the fallback).
  form.addEventListener("submit", (event) => {
    if (!url) { dirty = false; return; }
    event.preventDefault();
    clearTimeout(timer);
    explicit = true;
    save();
  });
  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-discard-reload]")) leaving = true;  // user chose to discard
  });
  window.addEventListener("beforeunload", (event) => {
    if ((dirty || saving) && form.isConnected && !leaving) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  }

  function scan() { setup(document.getElementById("note-editor")); }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scan);
  } else {
    scan();
  }
  document.addEventListener("htmx:load", scan);
})();
