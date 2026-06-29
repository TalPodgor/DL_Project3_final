/* ===========================================================================
   Project page interactions: mobile nav, scroll-reveal, before/after slider,
   board switcher, copy-bibtex.
   =========================================================================== */
(function () {
  "use strict";

  /* ---------- mobile nav toggle ---------- */
  var nav = document.getElementById("nav");
  var navToggle = document.getElementById("navToggle");
  if (navToggle) {
    navToggle.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    nav.querySelectorAll(".nav-links a").forEach(function (a) {
      a.addEventListener("click", function () { nav.classList.remove("open"); });
    });
  }

  /* ---------- scroll reveal ---------- */
  var reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    reveals.forEach(function (el) { io.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add("in"); });
  }

  /* ---------- before/after slider ---------- */
  var ba = document.getElementById("ba");
  if (ba) {
    var before = document.getElementById("baBefore");
    var after  = document.getElementById("baAfter");
    var dragging = false;

    function setPos(clientX) {
      var rect = ba.getBoundingClientRect();
      var x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
      var pct = (x / rect.width) * 100;
      ba.style.setProperty("--pos", pct + "%");
    }
    function fromEvent(ev) {
      var cx = ev.touches ? ev.touches[0].clientX : ev.clientX;
      setPos(cx);
    }
    function start(ev) { dragging = true; fromEvent(ev); ev.preventDefault(); }
    function move(ev)  { if (dragging) fromEvent(ev); }
    function end()     { dragging = false; }

    ba.addEventListener("pointerdown", start);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", end);
    // graceful fallback for browsers without pointer events
    ba.addEventListener("mousedown", start);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", end);
    ba.addEventListener("touchstart", start, { passive: false });
    window.addEventListener("touchmove", move, { passive: false });
    window.addEventListener("touchend", end);

    // keyboard accessibility
    ba.setAttribute("tabindex", "0");
    ba.addEventListener("keydown", function (ev) {
      var cur = parseFloat(getComputedStyle(ba).getPropertyValue("--pos")) || 50;
      if (ev.key === "ArrowLeft")  { ba.style.setProperty("--pos", Math.max(0, cur - 4) + "%"); ev.preventDefault(); }
      if (ev.key === "ArrowRight") { ba.style.setProperty("--pos", Math.min(100, cur + 4) + "%"); ev.preventDefault(); }
    });

    /* board switcher */
    var chips = document.querySelectorAll(".chip[data-board]");
    chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        chips.forEach(function (c) { c.classList.remove("active"); });
        chip.classList.add("active");
        var b = chip.getAttribute("data-board");
        before.src = "assets/ba_" + b + "_synth.jpg";
        after.src  = "assets/ba_" + b + "_real.jpg";
        ba.style.setProperty("--pos", "50%");
      });
    });
  }

  /* ---------- copy bibtex ---------- */
  var copyBtn = document.getElementById("copyBib");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var text = document.getElementById("bibtex").innerText;
      var done = function () {
        var old = copyBtn.textContent;
        copyBtn.textContent = "Copied ✓";
        setTimeout(function () { copyBtn.textContent = old; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(ta); done();
      }
    });
  }
})();
