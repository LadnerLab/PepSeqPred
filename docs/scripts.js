(function () {
  "use strict";

  function getByPath(obj, path) {
    const tokens = path.split(".");
    let current = obj;
    for (const token of tokens) {
      if (current === null || typeof current !== "object" || !(token in current)) {
        return null;
      }
      current = current[token];
    }
    return current;
  }

  function toNumber(value) {
    if (typeof value === "number") {
      return Number.isFinite(value) ? value : null;
    }
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }

  function formatValue(raw, format) {
    if (raw === null || raw === undefined) {
      return "NA";
    }

    if (format === "int") {
      const value = toNumber(raw);
      return value === null ? "NA" : String(Math.round(value));
    }

    if (format === "date") {
      return String(raw);
    }

    const value = toNumber(raw);
    if (value === null) {
      return String(raw);
    }

    switch (format) {
      case "fixed6":
        return value.toFixed(6);
      case "fixed4":
        return value.toFixed(4);
      case "fixed3":
        return value.toFixed(3);
      case "signed6":
        return value >= 0 ? `+${value.toFixed(6)}` : value.toFixed(6);
      case "signed3":
        return value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3);
      case "percent4":
        return `${(value * 100).toFixed(4)}%`;
      case "percent2":
        return `${(value * 100).toFixed(2)}%`;
      case "sci2":
        return value.toExponential(2);
      default:
        return String(value);
    }
  }

  function hydrateBindings(snapshot) {
    const nodes = document.querySelectorAll("[data-json-path]");
    nodes.forEach((node) => {
      const path = node.getAttribute("data-json-path");
      const format = node.getAttribute("data-format") || "";
      const raw = getByPath(snapshot, path);
      node.textContent = formatValue(raw, format);
    });
  }

  async function loadSnapshot() {
    const status = document.getElementById("snapshot-status");
    try {
      const response = await fetch("data/benchmark_snapshot.json", {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const snapshot = await response.json();
      hydrateBindings(snapshot);
      if (status) {
        status.textContent = "Benchmark snapshot loaded.";
        status.classList.add("ok");
      }
    } catch (err) {
      if (status) {
        status.textContent = `Could not load benchmark snapshot: ${String(err)}`;
        status.classList.add("error");
      }
    }
  }

  document.addEventListener("DOMContentLoaded", loadSnapshot);
})();
