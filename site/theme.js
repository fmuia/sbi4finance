// Run before the stylesheet so a saved preference is applied before first paint.
(() => {
  const root = document.documentElement;
  const system = matchMedia("(prefers-color-scheme: dark)");
  const storageKey = "sbi4finance-theme";
  let preference = null;
  let button = null;
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved === "light" || saved === "dark") preference = saved;
  } catch {
    // Theme switching still works if browser storage is unavailable.
  }

  function apply(theme) {
    root.dataset.theme = theme;
    const dark = theme === "dark";
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", dark ? "#0d1320" : "#f8f7f3");
    if (button) {
      const action = dark ? "Light mode" : "Dark mode";
      button.setAttribute("aria-label", `Switch to ${action.toLowerCase()}`);
      button.title = `Switch to ${action.toLowerCase()}`;
    }
    window.dispatchEvent(new CustomEvent("themechange", {detail: {theme}}));
  }

  apply(preference || (system.matches ? "dark" : "light"));
  system.addEventListener("change", () => {
    if (!preference) apply(system.matches ? "dark" : "light");
  });
  document.addEventListener("DOMContentLoaded", () => {
    button = document.querySelector("[data-theme-toggle]");
    if (!button) return;
    apply(root.dataset.theme);
    button.hidden = false;
    button.addEventListener("click", () => {
      preference = root.dataset.theme === "dark" ? "light" : "dark";
      try { localStorage.setItem(storageKey, preference); } catch { /* Keep the in-memory preference. */ }
      apply(preference);
    });
    requestAnimationFrame(() => requestAnimationFrame(() => root.classList.add("theme-ready")));
  });
})();
