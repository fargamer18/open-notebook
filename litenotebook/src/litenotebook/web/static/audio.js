// Audio Overview page: chapters seek the native player, the transcript
// follows playback, and "Copy transcript". Everything works without it.
(function () {
  "use strict";
  const player = document.getElementById("audio-player");
  let times = [];
  try {
    const raw = document.getElementById("turn-times");
    if (raw) times = JSON.parse(raw.textContent || "[]");
  } catch (e) { times = []; }

  function sectionFor(turnId) {
    const turn = document.getElementById("turn-" + turnId);
    return turn ? turn.closest(".transcript-section") : null;
  }

  document.addEventListener("click", (event) => {
    const chapter = event.target.closest("[data-seek]");
    if (chapter && player) {
      player.currentTime = Number(chapter.dataset.seek) || 0;
      player.play().catch(() => {});
      const section = document.getElementById("sec-" + chapter.dataset.section);
      if (section) section.scrollIntoView({ block: "start", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      return;
    }
    const copy = event.target.closest("[data-copy-transcript]");
    if (copy) {
      const text = (document.getElementById("transcript-text") || {}).value || "";
      const status = document.getElementById("copy-status");
      const done = (ok) => { if (status) status.textContent = ok ? "Transcript copied" : "Copy failed: use Download transcript"; };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(() => done(true), () => done(false));
      else done(false);
    }
  });

  if (player && times.length) {
    let current = null;
    player.addEventListener("timeupdate", () => {
      const t = player.currentTime;
      const hit = times.find((x) => t >= x[1] && t < x[2] + 0.35);
      const id = hit ? hit[0] : null;
      if (id === current) return;
      if (current !== null) {
        const old = document.getElementById("turn-" + current);
        if (old) old.classList.remove("is-speaking");
      }
      current = id;
      if (id === null) return;
      const turn = document.getElementById("turn-" + id);
      if (turn) turn.classList.add("is-speaking");
      const section = sectionFor(id);
      document.querySelectorAll(".chapter").forEach((b) => {
        const on = section && ("sec-" + b.dataset.section) === section.id;
        if (on) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current");
      });
    });
  }
})();
