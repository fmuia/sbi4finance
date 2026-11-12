// Decorative model illustrations only: no market data or inference results.
(() => {
  const canvas = document.querySelector("[data-hero-canvas]");
  if (!canvas) return;
  const context = canvas.getContext("2d");
  if (!context) return;
  const hero = canvas.closest(".hero");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  let inView = true;
  let frameId = 0;
  let lastFrame = 0;
  let time = 0;
  let width = 0;
  let height = 0;
  const colors = {
    grid: ["#52729a", "#7290b3"],
    pathMain: ["#087f7d", "#70e0ce"],
    pathA: ["#315aa3", "#72aaf7"],
    pathB: ["#7160ae", "#a99af5"],
    point: ["#067a74", "#9ee9dd"],
    smileMain: ["#a15b20", "#e3b785"],
    smileDraw: ["#3555aa", "#8db0ff"],
    quote: ["#263c61", "#dce9ff"],
    surfaceA: ["#16878b", "#6ed4ce"],
    surfaceB: ["#6957aa", "#9c9cf2"],
    surfaceGrid: ["#366a9d", "#75a6dd"],
    varianceA: ["#168488", "#7ad5d5"],
    varianceB: ["#405da0", "#88a8f2"],
  };
  let themeBlend = document.documentElement.dataset.theme === "dark" ? 1 : 0;
  let targetTheme = themeBlend;
  const palette = {};
  function updatePalette() {
    for (const [name, [light, dark]] of Object.entries(colors)) {
      palette[name] = "#" + [1, 3, 5].map(offset => {
        const start = parseInt(light.slice(offset, offset + 2), 16);
        const end = parseInt(dark.slice(offset, offset + 2), 16);
        return Math.round(start + (end - start) * themeBlend).toString(16).padStart(2, "0");
      }).join("");
    }
  }
  updatePalette();

  // Two independent Brownian paths per curve let the field morph smoothly.
  // cos² + sin² = 1 preserves the Brownian covariance at every animation time.
  let seed = 41026;
  function uniform() {
    seed = (1664525 * seed + 1013904223) >>> 0;
    return (seed + 1) / 4294967297;
  }
  function normal() {
    return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  }
  const steps = 240;
  const paths = Array.from({length: 26}, (_, index) => {
    const a = [0];
    const b = [0];
    for (let step = 1; step <= steps; step++) {
      a.push(a[step - 1] + normal() / Math.sqrt(steps));
      b.push(b[step - 1] + normal() / Math.sqrt(steps));
    }
    return {a, b, phase: index * 0.61};
  });

  function stroke(points, color, opacity, lineWidth = 1) {
    context.beginPath();
    points.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
    context.strokeStyle = color;
    context.globalAlpha = opacity;
    context.lineWidth = lineWidth;
    context.stroke();
  }

  function grid() {
    context.strokeStyle = palette.grid;
    context.globalAlpha = 0.07;
    context.lineWidth = 0.7;
    context.beginPath();
    const spacing = width < 640 ? 64 : 88;
    for (let x = 0; x <= width; x += spacing) {
      context.moveTo(x, 0);
      context.lineTo(x, height);
    }
    for (let y = 0; y <= height; y += spacing) {
      context.moveTo(0, y);
      context.lineTo(width, y);
    }
    context.stroke();
  }

  function trajectories(t) {
    const count = width < 640 ? 14 : paths.length;
    for (let index = 0; index < count; index++) {
      const {a, b, phase} = paths[index];
      const angle = t * 0.075 + phase;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      const points = a.map((value, step) => {
        const maturity = step / steps;
        const price = Math.exp(-0.06 * maturity + 0.4 * (cosine * value + sine * b[step]));
        return [width * (-0.06 + maturity * 1.14), height * (0.69 - (price - 1) * 0.42)];
      });
      const featured = index % 9 === 0;
      stroke(points, featured ? palette.pathMain : index % 2 ? palette.pathA : palette.pathB, featured ? 0.46 : 0.16, featured ? 1.5 : 0.9);
      if (featured) {
        const progress = ((t * 0.075 + index * 0.19) % 1) * steps;
        const head = Math.floor(progress);
        const fraction = progress - head;
        const [x0, y0] = points[head];
        const [x1, y1] = points[head + 1];
        const x = x0 + (x1 - x0) * fraction;
        const y = y0 + (y1 - y0) * fraction;
        // A short highlight follows the path, fading toward the back of the dot.
        const tailStart = Math.max(0, progress - 12);
        const tail = [...points.slice(Math.ceil(tailStart), head + 1), [x, y]];
        context.save();
        context.lineCap = "round";
        for (let segment = 1; segment < tail.length; segment++) {
          const fade = segment / (tail.length - 1);
          stroke([tail[segment - 1], tail[segment]], palette.point, 0.62 * fade * fade, 1.2 + fade);
        }
        context.restore();
        context.globalAlpha = 0.72;
        context.fillStyle = palette.point;
        context.beginPath();
        context.arc(x, y, 2.5, 0, Math.PI * 2);
        context.fill();
      }
    }
  }

  function smile(u, maturity, t, draw = 0) {
    const k = (u - 0.53) * 2;
    const skew = -0.06 + Math.sin(t * 0.12 + maturity) * 0.025;
    const uncertainty = Math.abs(k) * Math.abs(k) * draw * 0.006;
    return 0.22 + maturity * 0.025 + (0.22 + Math.sin(t * 0.1) * 0.03) * k * k + skew * k + uncertainty;
  }

  function smileFan(t) {
    const points = 160;
    for (let draw = -8; draw <= 8; draw++) {
      const curve = Array.from({length: points + 1}, (_, index) => {
        // Extend beyond the visible field so the fan has no exposed left edge.
        const u = -0.55 + (index / points) * 1.8;
        return [width * (0.54 + u * 0.53), height * (0.44 - smile(u, 0.4, t, draw) * 0.58)];
      });
      stroke(curve, draw === 0 ? palette.smileMain : palette.smileDraw, draw === 0 ? 0.67 : 0.21, draw === 0 ? 1.7 : 1);
    }
    // Quote-like markers illustrate where the fan of curves is constrained.
    for (let index = 0; index < 7; index++) {
      const u = 0.32 + index * 0.065;
      const x = width * (0.54 + u * 0.53);
      const y = height * (0.44 - smile(u, 0.4, t) * 0.58);
      context.globalAlpha = 0.6;
      context.fillStyle = palette.quote;
      context.beginPath();
      context.arc(x, y, 2, 0, Math.PI * 2);
      context.fill();
    }
  }

  function surfacePoint(u, maturity, t) {
    return [
      width * (0.45 + u * 0.58 + maturity * 0.14),
      height * (1.03 - maturity * 0.28 - smile(u, maturity, t) * 0.43),
    ];
  }

  function volatilitySurface(t) {
    const rows = width < 640 ? 10 : 18;
    for (let row = 0; row <= rows; row++) {
      const maturity = row / rows;
      const points = Array.from({length: 81}, (_, index) => surfacePoint(index / 80, maturity, t));
      stroke(points, row % 3 === 0 ? palette.surfaceA : palette.surfaceB, 0.19 + maturity * 0.16, 1);
    }
    for (let strike = 0; strike <= 20; strike++) {
      const points = Array.from({length: rows + 1}, (_, index) => surfacePoint(strike / 20, index / rows, t));
      stroke(points, palette.surfaceGrid, 0.14, 0.7);
    }
  }

  function forwardVariance(t) {
    for (let draw = 0; draw < 9; draw++) {
      const color = draw === 4 ? palette.varianceA : palette.varianceB;
      const fade = context.createLinearGradient(width * 0.12, 0, width * 0.75, 0);
      fade.addColorStop(0, color);
      fade.addColorStop(0.35, `${color}b3`);
      fade.addColorStop(1, `${color}00`);
      const points = Array.from({length: 100}, (_, index) => {
        // Continue past the fade, rather than ending the curves in mid-air.
        const u = (index / 99) * 1.9;
        const level = 0.055 * Math.exp(-u * 4) + 0.02 * Math.sin(u * 5 + t * 0.15 + draw * 0.14);
        return [width * (-0.08 + u * 0.47), height * (0.12 + level + draw * 0.011)];
      });
      stroke(points, fade, draw === 4 ? 0.35 : 0.14, 1);
    }
  }

  function draw() {
    context.clearRect(0, 0, width, height);
    grid();
    forwardVariance(time);
    trajectories(time);
    volatilitySurface(time);
    smileFan(time);
    context.globalAlpha = 1;
  }

  function canAnimate() {
    return !reducedMotion.matches && inView && document.visibilityState === "visible";
  }

  function frame(timestamp) {
    frameId = 0;
    if (!canAnimate()) return;
    if (!lastFrame) lastFrame = timestamp;
    if (timestamp - lastFrame >= 1000 / 30) {
      const elapsed = Math.min((timestamp - lastFrame) / 1000, 0.1);
      time += elapsed;
      if (themeBlend !== targetTheme) {
        themeBlend += Math.sign(targetTheme - themeBlend) * Math.min(Math.abs(targetTheme - themeBlend), elapsed / 0.4);
        updatePalette();
      }
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

  function resize() {
    const rect = hero.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    const ratio = Math.min(devicePixelRatio || 1, 1.75);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }

  reducedMotion.addEventListener("change", () => {
    if (reducedMotion.matches) {
      themeBlend = targetTheme;
      updatePalette();
    }
    sync();
    draw();
  });
  window.addEventListener("themechange", () => {
    targetTheme = document.documentElement.dataset.theme === "dark" ? 1 : 0;
    if (!canAnimate()) {
      themeBlend = targetTheme;
      updatePalette();
      draw();
    }
    sync();
  });
  document.addEventListener("visibilitychange", sync);
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting && entry.intersectionRatio > 0.05;
      sync();
    }, {threshold: 0.05}).observe(hero);
  }
  if ("ResizeObserver" in window) new ResizeObserver(resize).observe(hero);
  else window.addEventListener("resize", resize);
  resize();
  sync();
})();
