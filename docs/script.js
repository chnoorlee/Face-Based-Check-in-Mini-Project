/* 课堂自动人脸签到系统 · 宣传页交互（无任何外部依赖） */
(function () {
  "use strict";

  // ---------- 滚动显现动画 ----------
  var revealEls = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add("visible"); });
  }

  // ---------- 复制按钮 ----------
  document.querySelectorAll(".copy-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy") || "";
      var done = function () {
        btn.textContent = "已复制";
        btn.classList.add("copied");
        setTimeout(function () {
          btn.textContent = "复制";
          btn.classList.remove("copied");
        }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done); });
      } else {
        fallbackCopy(text, done);
      }
    });
  });

  function fallbackCopy(text, done) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); done(); } catch (e) { /* 忽略复制失败 */ }
    document.body.removeChild(ta);
  }

  // ---------- 数字滚动 ----------
  function countUp(el, target, duration) {
    var start = null;
    function tick(ts) {
      if (start === null) start = ts;
      var p = Math.min((ts - start) / duration, 1);
      el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  var statObserver = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      entry.target.querySelectorAll("[data-count]").forEach(function (el) {
        countUp(el, parseInt(el.getAttribute("data-count"), 10), 1100);
      });
      statObserver.unobserve(entry.target);
    });
  }, { threshold: 0.5 });
  var statsEl = document.querySelector(".stats");
  if (statsEl) statObserver.observe(statsEl);
})();
