/** Thin Chart.js helpers. Requires vendor/chart.umd.min.js (window.Chart). */

const registry = Object.create(null);

const GRID = "rgba(244, 244, 245, 0.10)";
const TICK = "#a1a1aa";
const FONT = "'IBM Plex Sans', system-ui, sans-serif";
const MONO = "'IBM Plex Mono', ui-monospace, monospace";
const PALETTE = ["#c8ccd4", "#8fa58a", "#9aa7c4", "#c4b18a", "#a8c5c0", "#c48982", "#d4c4a8", "#b8a8c4"];

function chartLib() {
  return globalThis.Chart || null;
}

export function destroyChart(id) {
  if (registry[id]) {
    registry[id].destroy();
    delete registry[id];
  }
}

export function resizeCharts() {
  for (const c of Object.values(registry)) {
    try {
      c.resize();
    } catch {
      /* */
    }
  }
}

function baseOptions({ yLabel, suggestedMax, indexAxis, ticksCallback }) {
  const horiz = indexAxis === "y";
  const valueScale = {
    beginAtZero: true,
    suggestedMax,
    grid: { color: GRID, drawBorder: false },
    title: yLabel ? { display: true, text: yLabel, color: TICK, font: { family: FONT, size: 11 } } : undefined,
    ticks: { color: TICK, font: { family: MONO, size: 11 }, callback: ticksCallback },
  };
  const catScale = {
    grid: { display: false, drawBorder: false },
    ticks: { color: TICK, font: { family: FONT, size: 11 }, maxRotation: horiz ? 0 : 40, autoSkip: false },
  };
  return {
    indexAxis: indexAxis || "x",
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#121214",
        titleColor: "#f4f4f5",
        bodyColor: "#a1a1aa",
        titleFont: { family: FONT, size: 12 },
        bodyFont: { family: MONO, size: 11 },
      },
    },
    scales: horiz ? { x: valueScale, y: catScale } : { x: catScale, y: valueScale },
  };
}

export function barChart(canvasId, { labels, values, colors, yLabel, suggestedMax, indexAxis, ticksCallback }) {
  const Chart = chartLib();
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  destroyChart(canvasId);
  if (!Chart) {
    canvas.replaceWith(fallbackSvg(canvasId, labels, values, yLabel));
    return null;
  }
  const bg = (colors || labels.map((_, i) => PALETTE[i % PALETTE.length])).map((c) => c);
  registry[canvasId] = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: bg,
          borderWidth: 0,
          borderRadius: 4,
          maxBarThickness: indexAxis === "y" ? 18 : 48,
        },
      ],
    },
    options: baseOptions({ yLabel, suggestedMax, indexAxis, ticksCallback }),
  });
  return registry[canvasId];
}

function fallbackSvg(id, labels, values, yLabel) {
  const max = Math.max(1, ...values.map((v) => Number(v) || 0));
  const w = 640;
  const h = Math.max(180, 28 * labels.length + 40);
  const barH = 16;
  let bars = "";
  labels.forEach((lab, i) => {
    const v = Number(values[i]) || 0;
    const bw = Math.round((v / max) * 420);
    const y = 24 + i * 28;
    bars += `<text x="8" y="${y + 12}" fill="#a1a1aa" font-size="11">${escapeXml(lab)}</text>
      <rect x="140" y="${y}" width="${bw}" height="${barH}" fill="#c8ccd4" rx="3"/>
      <text x="${148 + bw}" y="${y + 12}" fill="#f4f4f5" font-size="11">${v}</text>`;
  });
  const ns = document.createElement("div");
  ns.id = id + "-fallback";
  ns.className = "chart-fallback";
  ns.innerHTML = `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" role="img" aria-label="${escapeXml(yLabel || "chart")}">${bars}</svg>`;
  return ns;
}

function escapeXml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

export function latestByModel(rows) {
  const map = new Map();
  for (const r of rows) {
    if (!r?.id) continue;
    const prev = map.get(r.id);
    if (!prev || (r.ts || "") > (prev.ts || "")) map.set(r.id, r);
  }
  return [...map.values()];
}
