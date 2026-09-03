// Right-rail TOC: highlight the number of the currently visible lesson.
// (Smooth scrolling itself is handled by CSS `scroll-behavior: smooth`.)

document.addEventListener("DOMContentLoaded", function () {
  // --- Copy buttons for code blocks ---
  document.querySelectorAll("pre").forEach(function (pre) {
    var code = pre.querySelector("code");
    if (!code) return;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.setAttribute("aria-label", "Copy code");
    btn.addEventListener("click", function () {
      var text = code.textContent;
      var done = function () {
        btn.textContent = "Copied!";
        btn.classList.add("copied");
        setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done); });
      } else {
        fallbackCopy(text, done);
      }
    });
    pre.appendChild(btn);
  });

  function fallbackCopy(text, cb) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(ta);
    cb();
  }

  var links = Array.prototype.slice.call(document.querySelectorAll("#toc a"));
  if (!links.length) return;

  var sections = links
    .map(function (a) {
      return document.querySelector(a.getAttribute("href"));
    })
    .filter(Boolean);

  function setActive(id) {
    links.forEach(function (a) {
      var li = a.parentElement;
      li.classList.toggle("active", a.getAttribute("href") === "#" + id);
    });
  }

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.setAttribute("data-on-screen", "");
        } else {
          entry.target.removeAttribute("data-on-screen");
        }
      });
      updateFromView();
    },
    { rootMargin: "0px 0px -40% 0px", threshold: 0 }
  );

  // The active lesson is the last one whose heading is still on screen
  // (bottom 40% of the viewport is ignored, so a lesson stops counting
  // as active once its heading has scrolled well past).
  function updateFromView() {
    var current = null;
    for (var i = 0; i < sections.length; i++) {
      if (sections[i].hasAttribute("data-on-screen")) {
        current = sections[i];
      }
    }
    if (current) setActive(current.id);
    // If nothing qualifies (long gap between lessons), keep the last highlight.
  }

  sections.forEach(function (s) {
    observer.observe(s);
  });
});
