/* nanolab ui — router + components. No build step, no dependencies.
   Every number comes from /api/*, the same SQLite rows the CLI reads. */

const page = document.getElementById("page");
const toastEl = document.getElementById("toast");

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const api = async (path) => {
  const r = await fetch("/api" + path);
  if (!r.ok) throw new Error(`${r.status} on ${path}`);
  return r.json();
};

const fmt = (x, d = 3) =>
  x === null || x === undefined ? "—" : Number(x).toFixed(d);

/* ── components ────────────────────────────────────────────────────── */

const modelChip = (m) => {
  if (!m) return "—";
  const i = m.lastIndexOf(":");
  if (i > 0)
    return `<span class="model">${esc(m.slice(0, i))}<span class="adapter">${esc(
      m.slice(i))}</span></span>`;
  return `<span class="model">${esc(m)}</span>`;
};

const status = (s) => `<span class="st ${esc(s)}"><i></i>${esc(s)}</span>`;

const kpi = (value, label, caption = "", badge = "") =>
  `<div class="kpi"><div class="top"><span class="l">${label}</span>${badge}</div>
   <div class="n">${value}</div><div class="c">${caption}</div></div>`;

const reward = (v, withBar = true) => {
  if (v === null || v === undefined) return '<span class="dim">—</span>';
  const pct = Math.max(0, Math.min(1, v)) * 100;
  return `<span class="rw"><span class="v">${fmt(v)}</span>${
    withBar ? `<span class="bar"><i style="width:${pct}%"></i></span>` : ""}</span>`;
};

function toast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => toastEl.classList.remove("show"), 1400);
}

window.copyCmd = (cmd) => {
  navigator.clipboard.writeText(cmd);
  toast("copied");
};

const cli = (cmd) =>
  `<button class="cli" onclick="copyCmd(this.dataset.cmd)" data-cmd="${esc(cmd)}"
    title="This copies the command — paste it in the Terminal app, or just ask Claude to run it. The UI itself is view-only."><b>$</b> ${esc(cmd)}</button>`;

const icons = {
  flask: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10 2v7L4.5 19a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 9V2M8.5 2h7"/></svg>',
  chart: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M3 3v18h18"/><rect x="7" y="10" width="3" height="7" rx=".8"/><rect x="12" y="6" width="3" height="11" rx=".8"/><rect x="17" y="13" width="3" height="4" rx=".8"/></svg>',
  wave: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M22 12h-4l-3 8L9 4l-3 8H2"/></svg>',
  server: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="2" y="3" width="20" height="7" rx="2"/><rect x="2" y="14" width="20" height="7" rx="2"/></svg>',
};

const empty = (icon, headline, body, cmd) =>
  `<div class="empty">${icons[icon] || ""}<div class="headline">${esc(headline)}</div>
   <div>${body}</div>${cmd ? `<span class="cmd">$ ${esc(cmd)}</span>` : ""}</div>`;

let gradSeq = 0;
const curveSvg = (rewards, w = 220, h = 44, endDot = true) => {
  if (!rewards || rewards.length < 2) return '<span class="dim">—</span>';
  const id = `g${gradSeq++}`;
  const lo = Math.min(...rewards, 0.1);
  const hi = Math.max(...rewards, 0.8);
  const span = hi - lo || 1;
  const y = (v) => h - 4 - ((v - lo) / span) * (h - 10);
  const dx = w / (rewards.length - 1);
  const pts = rewards.map((r, i) => [i * dx, y(r)]);
  const line = pts.map(([x, yy]) => `${x.toFixed(1)},${yy.toFixed(1)}`).join(" ");
  const area = `0,${h} ${line} ${w},${h}`;
  const [ex, ey] = pts[pts.length - 1];
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    <defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#B7F542" stop-opacity=".22"/>
      <stop offset="1" stop-color="#B7F542" stop-opacity="0"/></linearGradient></defs>
    <rect x="0" y="${y(0.8).toFixed(1)}" width="${w}"
      height="${(y(0.1) - y(0.8)).toFixed(1)}" fill="#B7F542" opacity="0.05"/>
    <polygon points="${area}" fill="url(#${id})"/>
    <polyline points="${line}" fill="none" stroke="#B7F542" stroke-width="1.5"/>
    ${endDot ? `<circle cx="${ex.toFixed(1)}" cy="${ey.toFixed(1)}" r="2.5" fill="#B7F542"/>` : ""}
  </svg>`;
};

const head = (title, sub, cmd) =>
  `<div class="page-head"><div><h1>${esc(title)}</h1>
   <div class="sub">${esc(sub)}</div></div>${cmd ? cli(cmd) : ""}</div>`;

const table = (headers, rows) =>
  `<div class="tablewrap"><table><tr>${headers
    .map((h) => `<th>${h}</th>`).join("")}</tr>${rows.join("")}</table></div>`;

const skeleton = () =>
  `<div class="page">
   <div class="skel" style="height:44px;width:40%;margin-bottom:1.3rem"></div>
   <div class="kpis">${'<div class="skel" style="height:84px"></div>'.repeat(4)}</div>
   <div class="skel" style="height:260px"></div></div>`;

/* ── pages ─────────────────────────────────────────────────────────── */

async function overview() {
  const d = await api("/overview");
  const recentEvals = d.recent_evals.length
    ? table(
        ["run", "env", "model", "status", "reward", ""],
        d.recent_evals.map(
          (e) => `<tr class="click" onclick="location.hash='#/evals/${e.id}'">
           <td class="num rank">#${e.id}</td><td class="mono">${esc(e.slug)}</td>
           <td>${modelChip(e.model)}</td><td>${status(e.status)}</td>
           <td>${reward(e.mean_reward)}</td><td class="chev">›</td></tr>`))
    : empty("chart", "Run your first evaluation",
        "Measure how a model performs on an environment, rollout by rollout.",
        "nanolab eval run gsm8k -m <model>");
  const recentTrains = d.recent_trains.length
    ? table(
        ["run", "env", "model", "status", "steps", "curve", ""],
        d.recent_trains.map(
          (t) => `<tr class="click" onclick="location.hash='#/training/${t.id}'">
           <td class="num rank">#${t.id}</td><td class="mono">${esc(t.slug ?? "?")}</td>
           <td>${modelChip(t.model)}</td><td>${status(t.status)}</td>
           <td class="num">${t.steps_completed}</td>
           <td>${curveSvg(t.rewards, 150, 34)}</td><td class="chev">›</td></tr>`))
    : empty("wave", "Run your first training run",
        "Train LoRA adapters with reinforcement learning on an environment.",
        "nanolab train configs/qwen3-0.6b-gsm8k.toml");
  return `${head("Overview", "one CLI · one SQLite file · one closed RL loop", "nanolab ui")}
    <div class="kpis">
      ${kpi(d.evals.done, "evals done", `${d.evals.active} active`)}
      ${kpi(d.best ? fmt(d.best.mean_reward) : "—", "best reward",
        d.best ? `${d.best.slug} · ${d.best.model}` : "no completed runs")}
      ${kpi(d.training.done, "train runs", `${d.adapters} adapters`)}
      ${kpi(d.environments, "environments", "installed")}
      ${kpi(d.tokens.toLocaleString(), "tokens", "ledger total")}
    </div>
    <div class="section-label"><b>01</b>recent evaluations</div>${recentEvals}
    <div class="section-label"><b>02</b>recent training</div>${recentTrains}`;
}

async function environments() {
  const envs = await api("/environments");
  const cards = envs.length
    ? `<div class="cards">${envs
        .map(
          (e) => `<div class="card">
          <div class="name">${esc(e.slug)}</div>
          <div class="meta">v${esc(e.version)} · installed ${esc(
            (e.installed_at || "").slice(0, 10))} ${
            e.importable ? "" : '<span class="badge-missing">· missing from venv</span>'}</div>
          <div class="score">${
            e.best_reward !== null
              ? `${fmt(e.best_reward)} <span class="dim" style="font-size:11px;font-family:var(--sans)">best · ${esc(e.best_model ?? "")}</span>`
              : '<span class="dim" style="font-size:12px">not evaluated yet</span>'}</div>
          </div>`).join("")}</div>`
    : empty("flask", "Install your first environment",
        "Environments are tasks plus automatic graders, in the standard verifiers format.",
        "nanolab env install primeintellect/gsm8k");
  return `${head("Environments", "tasks + automatic graders, hub-compatible",
    "nanolab env install <owner/name>")}${cards}`;
}

async function evals() {
  const runs = await api("/evals");
  const tbl = runs.length
    ? table(
        ["run", "env", "model", "status", "reward", "samples", "err", "date", ""],
        runs.map(
          (e) => `<tr class="click" onclick="location.hash='#/evals/${e.id}'">
          <td class="num rank">#${e.id}</td><td class="mono">${esc(e.env)}</td>
          <td>${modelChip(e.model)}</td><td>${status(e.status)}</td>
          <td>${reward(e.mean_reward)}</td>
          <td class="num">${e.num_examples}×${e.rollouts_per_example}</td>
          <td class="num${e.errors ? " down" : ""}">${e.errors}</td>
          <td class="num dim">${esc((e.finished_at || e.started_at || "").slice(0, 10))}</td>
          <td class="chev">›</td></tr>`)) +
      '<div class="fig">click a run for the rollout inspector</div>'
    : empty("chart", "Run your first evaluation",
        "An eval sends each task to a model and scores every answer with the environment's rubric.",
        "nanolab eval run gsm8k -m <model>");
  return `${head("Evaluations", "rollouts + rubric scoring, cached and resumable",
    "nanolab eval run <env> -m <model>")}
    <div class="kpis">
      ${kpi(runs.filter((r) => ["running", "pending"].includes(r.status)).length,
        "active evals", "pending or running")}
      ${kpi(runs.filter((r) => r.status === "done").length, "successful evals", "completed")}
      ${kpi(runs.length, "total evals", "all statuses")}
    </div>${tbl}`;
}

async function evalDetail(id) {
  const d = await api(`/evals/${id}`);
  const metrics = Object.entries(d.meta.avg_metrics || {})
    .map(([k, v]) => `<span class="chip">${esc(k)} ${fmt(v)}</span>`).join("");
  const rollouts = d.rollouts
    .map((r) => {
      const msgs = [...(r.prompt || []), ...(r.completion || [])];
      const convo = msgs
        .map((m) => `<div class="msg ${esc(m.role)}"><div class="role">${esc(m.role)}</div>
           <pre>${esc(m.content)}</pre></div>`).join("");
      const last = (r.completion || []).slice(-1)[0];
      const cls = (r.reward ?? 0) >= 0.5 ? "up" : "down";
      return `<details class="rollout"><summary>
        <span class="num rank">#${r.example}.${r.rollout}</span>
        <span class="v mono ${cls}" style="font-weight:500">${fmt(r.reward)}</span>
        <span class="preview">${esc(last ? last.content : "")}</span>
        ${r.metrics._error ? `<span class="chip failed">${esc(r.metrics._error)}</span>` : ""}
        </summary>${convo}</details>`;
    }).join("");
  return `<a class="back" href="#/evals">← Evaluations</a>
    ${head(`Eval run #${d.id}`, `${d.env} · ${d.model}`, `nanolab eval show ${d.id}`)}
    <div class="kpis">
      ${kpi(fmt(d.mean_reward), "mean reward", "", status(d.status))}
      ${kpi(`${d.num_examples}×${d.rollouts_per_example}`, "examples × rollouts",
        d.seed !== null ? `seed ${d.seed}` : "no shuffle")}
      ${kpi(d.rollouts.length, "rollouts stored",
        `${d.rollouts.filter((r) => r.metrics._error).length} errors`)}
    </div>
    ${metrics ? `<div class="metric-chips">${metrics}</div>` : ""}
    <div class="section-label"><b>01</b>rollout inspector</div>
    ${rollouts || '<div class="empty">no samples stored</div>'}`;
}

async function training() {
  const runs = await api("/training");
  const tbl = runs.length
    ? table(
        ["run", "env", "model", "status", "steps", "reward curve", "first → last", ""],
        runs.map((t) => {
          const first = t.rewards[0], last = t.rewards[t.rewards.length - 1];
          const delta = t.rewards.length ? last - first : null;
          return `<tr class="click" onclick="location.hash='#/training/${t.id}'">
            <td class="num rank">#${t.id}</td><td class="mono">${esc(t.env ?? "?")}</td>
            <td>${modelChip(t.model)}</td><td>${status(t.status)}</td>
            <td class="num">${t.steps_completed}</td><td>${curveSvg(t.rewards)}</td>
            <td class="num">${t.rewards.length
              ? `${fmt(first)} → ${fmt(last)}<br><span class="${delta >= 0 ? "up" : "down"}">${delta >= 0 ? "↑" : "↓"} ${fmt(Math.abs(delta))}</span>`
              : "—"}</td><td class="chev">›</td></tr>`;
        })) +
      '<div class="fig">shaded band = the 10–80% trainability window</div>'
    : empty("wave", "Run your first training run",
        "Training turns rewards into a LoRA adapter with GRPO — checkpointed, resumable.",
        "nanolab train configs/qwen3-0.6b-gsm8k.toml");
  return `${head("Training", "GRPO + LoRA, one synchronous loop",
    "nanolab train <config.toml> --resume")}${tbl}`;
}

async function trainingDetail(id) {
  const d = await api(`/training/${id}`);
  const rewards = d.curve.map((p) => p.reward);
  const adapters = d.adapters.length
    ? table(
        ["adapter", "step", "path", "evaluate"],
        d.adapters.map(
          (a) => `<tr><td class="num rank">#${a.id}</td><td class="num">${a.step}</td>
          <td class="mono dim">${esc(a.path)}</td>
          <td>${cli(`nanolab eval run <env> -m ${a.base_model}:${a.id}`)}</td></tr>`))
    : '<div class="empty">no checkpoints registered</div>';
  return `<a class="back" href="#/training">← Training</a>
    ${head(`Train run #${d.id}`, `${d.env ?? "?"} · ${d.model}`,
      `nanolab training show ${d.id}`)}
    <div class="kpis">
      ${kpi(d.steps_completed, "steps", "", status(d.status))}
      ${kpi(rewards.length ? fmt(rewards[rewards.length - 1]) : "—", "last reward",
        rewards.length ? `from ${fmt(rewards[0])}` : "")}
      ${kpi(d.adapters.length, "checkpoints", "in adapters/")}
    </div>
    <div class="section-label"><b>01</b>reward curve</div>
    ${curveSvg(rewards, 720, 130)}
    <div class="fig">FIG.1 — reward vs step · shaded band = trainability window</div>
    <div class="section-label"><b>02</b>checkpoints</div>${adapters}
    <div class="section-label"><b>03</b>config</div>
    <pre class="toml">${esc(d.config_toml || "")}</pre>`;
}

async function inference() {
  const deps = await api("/deployments");
  const tbl = deps.length
    ? table(
        ["id", "model", "endpoint", "pid", "status"],
        deps.map(
          (d) => `<tr><td class="num rank">#${d.id}</td>
          <td>${modelChip(`${d.base_model}:${d.adapter_id}`)}</td>
          <td class="mono dim">${esc(d.endpoint)}</td><td class="num">${d.pid ?? "—"}</td>
          <td>${status(d.status)}</td></tr>`))
    : empty("server", "No deployments yet",
        "Deployments serve a trained adapter as an OpenAI-compatible endpoint (vLLM, CUDA box) so the eval station can measure it with a base:adapter model string.",
        "nanolab deployments create <adapter-id>");
  return `${head("Inference", "serve adapters, close the loop",
    "nanolab deployments create <adapter-id>")}${tbl}`;
}

/* ── command palette ───────────────────────────────────────────────── */

let paletteOpen = false;

async function openPalette() {
  if (paletteOpen) return;
  paletteOpen = true;
  const items = [
    { label: "Overview", hash: "#/overview", kind: "page" },
    { label: "Environments", hash: "#/environments", kind: "page" },
    { label: "Evaluations", hash: "#/evals", kind: "page" },
    { label: "Training", hash: "#/training", kind: "page" },
    { label: "Inference", hash: "#/inference", kind: "page" },
  ];
  try {
    const [envs, evalsList, trains] = await Promise.all([
      api("/environments"), api("/evals"), api("/training"),
    ]);
    envs.forEach((e) => items.push({ label: e.slug, hash: "#/environments", kind: "env" }));
    evalsList.forEach((e) =>
      items.push({ label: `eval #${e.id} · ${e.env} · ${e.model} · ${fmt(e.mean_reward)}`,
        hash: `#/evals/${e.id}`, kind: "eval" }));
    trains.forEach((t) =>
      items.push({ label: `train #${t.id} · ${t.env ?? "?"} · ${t.model}`,
        hash: `#/training/${t.id}`, kind: "run" }));
  } catch {}

  const overlay = document.createElement("div");
  overlay.className = "palette-overlay";
  overlay.innerHTML = `<div class="palette">
    <input placeholder="Jump to page, environment, eval, run…" autofocus>
    <div class="results"></div></div>`;
  document.body.appendChild(overlay);
  const input = overlay.querySelector("input");
  const results = overlay.querySelector(".results");
  let sel = 0, filtered = items;

  const draw = () => {
    results.innerHTML = filtered.length
      ? filtered.slice(0, 12).map((it, i) =>
          `<div class="item${i === sel ? " sel" : ""}" data-i="${i}">
           ${esc(it.label)}<span class="kind">${it.kind}</span></div>`).join("")
      : '<div class="none">no matches</div>';
    results.querySelectorAll(".item").forEach((el) =>
      el.addEventListener("click", () => go(filtered[+el.dataset.i])));
  };
  const go = (item) => { if (item) { location.hash = item.hash; close(); } };
  const close = () => { overlay.remove(); paletteOpen = false; };

  input.addEventListener("input", () => {
    const q = input.value.toLowerCase();
    filtered = items.filter((it) => it.label.toLowerCase().includes(q));
    sel = 0; draw();
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
    else if (e.key === "ArrowDown") { sel = Math.min(sel + 1, Math.min(filtered.length, 12) - 1); draw(); }
    else if (e.key === "ArrowUp") { sel = Math.max(sel - 1, 0); draw(); }
    else if (e.key === "Enter") go(filtered[sel]);
  });
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  draw();
  input.focus();
}

document.getElementById("search-btn").addEventListener("click", openPalette);
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "k") { e.preventDefault(); openPalette(); }
});

/* ── router ────────────────────────────────────────────────────────── */

const routes = [
  [/^#?\/?$/, overview, "overview"],
  [/^#\/overview$/, overview, "overview"],
  [/^#\/environments$/, environments, "environments"],
  [/^#\/evals$/, evals, "evals"],
  [/^#\/evals\/(\d+)$/, evalDetail, "evals"],
  [/^#\/training$/, training, "training"],
  [/^#\/training\/(\d+)$/, trainingDetail, "training"],
  [/^#\/inference$/, inference, "inference"],
];

let current = null;
let lastHash = null;

async function render() {
  const hash = location.hash || "#/overview";
  const changed = hash !== lastHash;
  for (const [re, fn, nav] of routes) {
    const m = hash.match(re);
    if (m) {
      current = () => fn(...m.slice(1));
      document.querySelectorAll("nav a").forEach((a) =>
        a.classList.toggle("active", a.dataset.route === nav));
      if (changed) page.innerHTML = skeleton();
      try {
        const html = await current();
        const apply = () => {
          const openIdx = changed ? [] :
            [...document.querySelectorAll("details")].flatMap((d, i) => (d.open ? [i] : []));
          page.innerHTML = `<div class="page">${html}</div>`;
          const details = document.querySelectorAll("details");
          openIdx.forEach((i) => details[i] && (details[i].open = true));
        };
        if (changed && document.startViewTransition) document.startViewTransition(apply);
        else apply();
        lastHash = hash;
      } catch (err) {
        page.innerHTML = `<div class="empty">could not load: ${esc(err.message)}</div>`;
      }
      return;
    }
  }
  page.innerHTML = '<div class="empty">not found</div>';
}

window.addEventListener("hashchange", render);
render();
setInterval(() => {
  if (!document.hidden && !paletteOpen && current) render();
}, 5000);
