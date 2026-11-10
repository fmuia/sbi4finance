// Keep every platform's instructions readable when JavaScript is unavailable.
document.querySelectorAll("[data-os-tabs]").forEach((group) => {
  const tabList = group.querySelector('[role="tablist"]');
  const tabs = [...group.querySelectorAll('[role="tab"]')];
  const panels = [...group.querySelectorAll('[role="tabpanel"]')];

  function activate(tab, moveFocus = false) {
    tabs.forEach((item) => {
      const selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
    });
    panels.forEach((panel) => {
      panel.hidden = panel.id !== tab.getAttribute("aria-controls");
    });
    if (moveFocus) tab.focus();
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;
      event.preventDefault();
      activate(tabs[next], true);
    });
  });

  const platform = navigator.userAgentData?.platform || navigator.platform || "";
  const os = /win/i.test(platform) ? "windows" : /mac/i.test(platform) ? "macos" : "linux";
  activate(tabs.find((tab) => tab.dataset.os === os) || tabs[0]);
  tabList.hidden = false;
});
