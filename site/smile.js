// Animate the illustrative SVG's existing draws; axes and quote markers stay fixed.
(async () => {
  const image = document.querySelector("[data-smile-plot]");
  if (!image) return;
  try {
    const response = await fetch(image.src);
    if (!response.ok) return;
    const source = new DOMParser().parseFromString(await response.text(), "image/svg+xml");
    if (source.querySelector("parsererror")) return;
    const svg = source.documentElement;
    const curves = [...svg.querySelectorAll(".draw")];
    const draws = curves.map(curve => curve.getAttribute("points").trim().split(/\s+/).map(pair => pair.split(",").map(Number)));
    if (draws.length < 2 || !draws.every(points => points.length === draws[0].length)) return;
    svg.dataset.smilePlot = "";
    image.replaceWith(svg);
    await document.fonts.ready;
    svg.querySelectorAll(".axis-label").forEach(group => {
      const bounds = group.querySelector("text").getBBox();
      const capsule = group.querySelector("rect");
      capsule.setAttribute("x", bounds.x - 7);
      capsule.setAttribute("y", bounds.y - 4);
      capsule.setAttribute("width", bounds.width + 14);
      capsule.setAttribute("height", bounds.height + 8);
      capsule.setAttribute("rx", (bounds.height + 8) / 2);
    });

    const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
    let inView = true;
    let frameId = 0;
    let lastFrame = 0;
    let time = 0;

    function draw() {
      curves.forEach((curve, index) => {
        const phase = (time + index * 0.37) / 9;
        const cycle = Math.floor(phase);
        const weight = (1 - Math.cos(Math.PI * (phase - cycle))) / 2;
        const from = draws[(index + cycle) % draws.length];
        const to = draws[(index + cycle + 1) % draws.length];
        // Convex interpolation keeps the curves inside the original draw envelope.
        curve.setAttribute("points", from.map(([x, y], point) => `${x},${(y + (to[point][1] - y) * weight).toFixed(2)}`).join(" "));
      });
    }

    function canAnimate() {
      return !reducedMotion.matches && inView && document.visibilityState === "visible";
    }

    function frame(timestamp) {
      frameId = 0;
      if (!canAnimate()) return;
      if (!lastFrame) lastFrame = timestamp;
      if (timestamp - lastFrame >= 1000 / 30) {
        time += Math.min((timestamp - lastFrame) / 1000, 0.1);
        lastFrame = timestamp;
        draw();
      }
      frameId = requestAnimationFrame(frame);
    }

    function sync() {
      if (frameId) cancelAnimationFrame(frameId);
      frameId = 0;
      lastFrame = 0;
      if (canAnimate()) frameId = requestAnimationFrame(frame);
    }

    reducedMotion.addEventListener("change", sync);
    document.addEventListener("visibilitychange", sync);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(([entry]) => {
        inView = entry.isIntersecting && entry.intersectionRatio > 0.05;
        sync();
      }, {threshold: 0.05}).observe(svg);
    }
    sync();
  } catch {
    // Retain the original image if fetching/inlining is unavailable (e.g. file://).
  }
})();
