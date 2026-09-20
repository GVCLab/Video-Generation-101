// Material for MkDocs swaps page content without a reload when
// navigation.instant is on, so MathJax has to be re-run per navigation via the
// document$ observable rather than once on DOMContentLoaded.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*",
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code",
                   "annotation", "annotation-xml", "mjx-container"],
    processHtmlClass: "arithmatex",
  },
};

// Serialize navigation typesetting after initial startup; clearing while startup
// is active can append a second rendered container to each equation.
document$.subscribe(() => {
  if (!window.MathJax?.startup?.promise) return;
  MathJax.startup.promise = MathJax.startup.promise.then(() => {
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  });
});
