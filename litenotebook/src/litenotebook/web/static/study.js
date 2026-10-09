// Study tab: focus after answering, live announcements of generation stages,
// and number keys for choosing an option. Everything works without it.
(function () {
  "use strict";

  function focusTarget() {
    const target = document.querySelector("[data-autofocus]");
    if (!target) return;
    // After answering, the feedback gets focus (screen readers read the verdict);
    // on a new question, the first option does.
    if (location.hash === "#feedback" || target.id === "feedback" || target.tagName !== "INPUT") {
      target.focus({ preventScroll: false });
    } else {
      target.focus({ preventScroll: true });
    }
  }

  function announce(text) {
    const live = document.getElementById("quiz-announce");
    if (live && text && live.textContent !== text) live.textContent = text;
  }

  document.addEventListener("htmx:afterSwap", (event) => {
    const status = event.target && event.target.id === "quiz-status" ? event.target : document.getElementById("quiz-status");
    if (!status) return;
    const stage = document.getElementById("quiz-stage");
    if (stage) announce(stage.textContent.trim());
  });

  document.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const form = document.getElementById("answer-form");
    if (!form) return;
    const tag = (event.target.tagName || "").toLowerCase();
    if (tag === "textarea" || (tag === "input" && event.target.type === "text")) return;
    if (!/^[1-4]$/.test(event.key)) return;
    const inputs = form.querySelectorAll("input[name=choice]");
    const input = inputs[Number(event.key) - 1];
    if (!input) return;
    event.preventDefault();
    if (input.type === "radio") input.checked = true;
    else input.checked = !input.checked;
    input.focus();
  });

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", focusTarget);
  else focusTarget();
})();
