/* LiteNotebook chat: streams answers over POST + server-sent events.
 *
 * Model text is untrusted: streamed deltas are only ever added as text nodes.
 * When an answer finishes, the server-rendered message (escaped text with
 * application-built citation links) replaces the streamed one via htmx.
 */
(function () {
  "use strict";

  function setup(chat) {
  if (!chat || chat.dataset.ready) return;
  chat.dataset.ready = "1";
  const inReader = chat.dataset.inReader === "1";
  const form = document.getElementById("composer");
  const log = document.getElementById("chat-log");
  const textarea = document.getElementById("question");
  const sendBtn = document.getElementById("send");
  const stopBtn = document.getElementById("stop");
  const statusEl = document.getElementById("composer-status");
  const errorEl = document.getElementById("composer-error");
  const chatUrl = chat.dataset.chatUrl;
  const latestBtn = document.getElementById("to-latest");
  let active = null; // {controller, messageId, el}

  const nearBottom = () => log.scrollHeight - log.scrollTop - log.clientHeight < 80;
  const scrollIfFollowing = (follow) => { if (follow) log.scrollTop = log.scrollHeight; updateLatest(); };
  function updateLatest() { if (latestBtn) latestBtn.hidden = nearBottom(); }
  log.addEventListener("scroll", updateLatest, { passive: true });
  if (latestBtn) latestBtn.addEventListener("click", () => { log.scrollTop = log.scrollHeight; updateLatest(); textarea.focus(); });

  // Grows with its content up to a limit, then scrolls.
  function grow() {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight + 2, 224) + "px";
  }
  textarea.addEventListener("input", grow);
  textarea.addEventListener("input", () => { if (!errorEl.hidden && textarea.value.trim() && !active) showError(""); });

  function showError(text) {
    errorEl.textContent = text || "";
    errorEl.hidden = !text;
  }

  function setBusy(busy, label) {
    sendBtn.disabled = busy;
    textarea.readOnly = busy;
    stopBtn.hidden = !busy;
    stopBtn.disabled = false;
    log.setAttribute("aria-busy", busy ? "true" : "false");
    statusEl.textContent = label || "";
  }

  function messageElement(role, text) {
    const article = document.createElement("article");
    article.className = "msg msg-" + role + (role === "assistant" ? " msg-generating" : " msg-complete");
    const head = document.createElement("header");
    head.className = "msg-head";
    const who = document.createElement("span");
    who.className = "msg-role";
    who.textContent = role === "user" ? "You" : "Assistant";
    head.appendChild(who);
    article.appendChild(head);
    const body = document.createElement("div");
    body.className = "msg-body";
    body.textContent = text || "";
    article.appendChild(body);
    return article;
  }

  function replaceWithServerMessage(el, messageId) {
    if (!messageId || !window.htmx) return Promise.resolve();
    el.id = "msg-" + messageId;
    return window.htmx.ajax("GET", chatUrl + "/messages/" + messageId, {
      target: el, swap: "outerHTML",
    });
  }

  function parseEvents(buffer, onEvent) {
    // Returns the unconsumed tail. Events are separated by a blank line.
    let index;
    while ((index = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trim());
      }
      if (data.length) {
        try { onEvent(event, JSON.parse(data.join("\n"))); } catch (e) { /* ignore */ }
      }
    }
    return buffer;
  }

  async function stream(url, body, assistantEl, userEl) {
    const controller = new AbortController();
    const answer = assistantEl.querySelector(".msg-body");
    active = { controller: controller, messageId: null, el: assistantEl };
    setBusy(true, "Retrieving evidence…");
    showError("");
    let finished = false;
    try {
      const response = await fetch(url, {
        method: "POST",
        body: body,
        signal: controller.signal,
        headers: { "Accept": "text/event-stream" },
        credentials: "same-origin",
      });
      const type = response.headers.get("content-type") || "";
      if (!response.ok || !type.startsWith("text/event-stream")) {
        let message = "Request failed (" + response.status + ")";
        try { message = (await response.json()).error || message; } catch (e) { /* not JSON */ }
        showError(message);
        if (userEl && userEl.dataset.transient) userEl.remove();
        if (assistantEl.dataset.transient) assistantEl.remove();
        else if (assistantEl.dataset.messageId) replaceWithServerMessage(assistantEl, assistantEl.dataset.messageId);
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = parseEvents(buffer, (event, data) => {
          if (event === "start") {
            active.messageId = data.message_id;
            assistantEl.id = "msg-" + data.message_id;
            assistantEl.dataset.messageId = data.message_id;
            delete assistantEl.dataset.transient;
            if (userEl) delete userEl.dataset.transient;
            if (data.created_conversation) onConversationCreated(data);
          } else if (event === "status") {
            if (data.stage === "generating") {
              const parts = [];
              if (data.sources) parts.push(data.sources + " passage" + (data.sources === 1 ? "" : "s"));
              if (data.memories) parts.push(data.memories + " memor" + (data.memories === 1 ? "y" : "ies"));
              statusEl.textContent = "Generating from " + (parts.join(" + ") || "the notebook") + "…";
            } else if (data.stage === "loading_model") {
              statusEl.textContent = "Loading embedding model… (first question after a pause)";
            } else {
              statusEl.textContent = "Retrieving evidence…";
            }
          } else if (event === "delta") {
            const follow = nearBottom();
            answer.appendChild(document.createTextNode(data.t));
            scrollIfFollowing(follow);
          } else if (event === "done" || event === "error") {
            finished = true;
            if (event === "error") showError(data.message);
          }
        });
      }
    } catch (err) {
      if (err.name !== "AbortError") showError("Connection lost: " + err.message);
    } finally {
      const messageId = active && active.messageId;
      const follow = nearBottom();  // never yank a reader back down
      const focusWasHere = !document.activeElement || document.activeElement === document.body ||
        form.contains(document.activeElement);
      active = null;
      setBusy(false, "");
      if (messageId) await replaceWithServerMessage(assistantEl, messageId);
      if (!finished && !messageId && assistantEl.dataset.transient) assistantEl.remove();
      scrollIfFollowing(follow);
      if (finished && errorEl.hidden) statusEl.textContent = "Answer ready.";
      if (focusWasHere) textarea.focus();
    }
  }

  function onConversationCreated(data) {
    form.elements.conversation_id.value = data.conversation_id;
    const url = chatUrl + "/" + data.conversation_id;
    if (!inReader) history.replaceState(null, "", url);
    const title = document.getElementById("chat-title");
    if (title) title.textContent = data.title;
    const sidebar = document.getElementById("conversations");
    if (sidebar && window.htmx) {
      window.htmx.ajax("GET", chatUrl + "/sidebar?current=" + data.conversation_id, {
        target: "#conversations", swap: "innerHTML",
      });
    }
    if (window.htmx) {
      window.htmx.ajax("GET", url + "/header", { target: "#chat-header", swap: "innerHTML" });
    }
  }

  function send() {
    if (active) return;
    const question = textarea.value.trim();
    if (!question) { textarea.value = ""; showError("Type a question first."); textarea.focus(); return; }
    const empty = document.getElementById("chat-empty");
    if (empty) empty.remove();
    const userEl = messageElement("user", question);
    const assistantEl = messageElement("assistant", "");
    userEl.dataset.transient = "1";
    assistantEl.dataset.transient = "1";
    log.appendChild(userEl);
    log.appendChild(assistantEl);
    scrollIfFollowing(true);
    const body = new URLSearchParams();
    body.set("question", question);
    body.set("conversation_id", form.elements.conversation_id.value);
    textarea.value = "";
    grow();
    stream(chat.dataset.sendUrl, body, assistantEl, userEl).then(() => {
      if (userEl.isConnected && userEl.dataset.transient) textarea.value = question;
    });
  }

  form.addEventListener("submit", (event) => { event.preventDefault(); send(); });

  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send();
    }
  });

  stopBtn.addEventListener("click", async () => {
    if (!active) return;
    const current = active;
    stopBtn.disabled = true;
    statusEl.textContent = "Stopping…";
    let stopped = false;
    if (current.messageId) {
      try {
        const r = await fetch(chatUrl + "/messages/" + current.messageId + "/stop", {
          method: "POST", credentials: "same-origin",
        });
        stopped = r.status === 204;
      } catch (e) { /* fall through to abort */ }
    }
    // The server finishes the stream itself after a stop; abort if it does not.
    setTimeout(() => { if (active === current) current.controller.abort(); }, stopped ? 3000 : 0);
  });

  log.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-retry-url]");
    if (!button || active) return;
    const article = button.closest("article.msg");
    const fresh = messageElement("assistant", "");
    fresh.id = article.id;
    fresh.dataset.messageId = article.dataset.messageId;
    article.replaceWith(fresh);
    stream(button.dataset.retryUrl, new URLSearchParams(), fresh, null);
  });

  scrollIfFollowing(true);
  grow();
  }

  function scan(root) {
    const chat = root && root.id === "chat" ? root : document.getElementById("chat");
    if (chat) setup(chat);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => scan(document));
  } else {
    scan(document);
  }
  document.addEventListener("htmx:load", (event) => scan(event.target));
})();
