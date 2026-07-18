/* nanolab ui — hash router + renderers. No build step, no dependencies.
   Every number comes from /api/*, which reads the same SQLite the CLI reads. */

const page = document.getElementById("page");

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const api = async (path) => {
  const r = await fetch("/api" + path);
  if (!r.ok) throw new Error(`${r.status} on ${path}`);
  return r.json();
};

const fmt = (x, digits = 3) =>
  x === null || x === undefined ? "—" : Number(x).toFixed(digits);

const modelChip = (m) => {
  if (!m) return "—";
  const i = m.lastIndexOf(":");
  if (i > 0)
    return `<span class="model">${esc(m.slice(0, i))}<span class="adapter">${esc(
      m.slice(i)
    )}</span></span>`;
  return `<span class="model">${esc(m)}</span>`;
};

const chip = (status) => `<span class="chip ${esc(status)}">${esc(status)}</span>`;

const kpi = (n, label, caption = "") =>
  `<div class="kpi"><div class="n">${n}</div><div class="l">${label}</div>
   <div class="c">${caption}</div></div>`;

const cli = (cmd) =>
  `<div class="cli" title="click to copy" onclick="navigator.clipboard.writeText('${esc(
    cmd
  )}')"><b>$</b> ${esc(cmd)}</div>`;

const empty = (headline, body, cmd) =>
  `<div class="empty"><div class="headline">${esc(headline)}</div>
   <div>${body}</div>${cmd ? `<span class="cmd">$ ${esc(cmd)}</span>` : ""}</div>`;

const curveSvg = (rewards, w = 220, h = 44) => {
  if (!rewards || rewards.length < 2) return "—";
  const lo = Math.min(...rewards, 0.1);
  const hi = Math.max(...rewards, 0.8);
  const span = hi - lo || 1;
  const y = (v) => h - 4 - ((v - lo) / span) * (h - 8);
  const dx = w / (rewards.length - 1);
  const pts = rewards.map((r, i) => `${(i * dx).toFixed(1)},${y(r).toFixed(1)}`).join(" ");
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    <rect x="0" y="${y(0.8).toFixed(1)}" width="${w}"
      height="${(y(0.1) - y(0.8)).toFixed(1)}" fill="#B7F542" opacity="0.07"/>
    <polyline points="${pts}" fill="none" stroke="#B7F542" stroke-width="1.5"/></svg>`;
};

const head = (title, sub, cmd) =>
  `<div class="page-head"><div><h1>${esc(title)}</h1>
   <div class="sub">${esc(sub)}</div></div>${cmd ? cli(cmd) : ""}</div>`;

/* ── pages ─────────────────────────────────────────────────────────── */

async function overview() {
  const d = await api("/overview");
  const best = d.best
    ? `${fmt(d.best.mean_reward)} <span class="dim">· ${esc(d.best.slug)}</span>`
    : "—";
  const recentEvals = d.recent_evals.length
    ? `<table><tr><th>run</th><th>env</th><th>model</th><th>status</th><th>reward</th></tr>
       ${d.recent_evals
         .map(
           (e) => `<tr class="click" onclick="location.hash='#/evals/${e.id}'">
            <td class="num">#${e.id}</td><td class="mono">${esc(e.slug)}</td>
            <td>${modelChip(e.model)}</td><td>${chip(e.status)}</td>
            <td class="num reward">${fmt(e.mean_reward)}</td></tr>`
         )
         .join("")}</table>`
    : empty("Run your first evaluation",
        "Measure how a model performs on an environment, rollout by rollout.",
        "nanolab eval run gsm8k -m <model>");
  const recentTrains = d.recent_trains.length
    ? `<table><tr><th>run</th><th>env</th><th>model</th><th>status</th><th>steps</th><th>curve</th></tr>
       ${d.recent_trains
         .map(
           (t) => `<tr class="click" onclick="location.hash='#/training/${t.id}'">
            <td class="num">#${t.id}</td><td class="mono">${esc(t.slug ?? "?")}</td>
            <td>${modelChip(t.model)}</td><td>${chip(t.status)}</td>
            <td class="num">${t.steps_completed}</td><td>${curveSvg(t.rewards, 140, 32)}</td></tr>`
         )
         .join("")}</table>`
    : empty("Run your first training run",
        "Train LoRA adapters with reinforcement learning on an environment.",
        "nanolab train configs/qwen3-0.6b-gsm8k.toml");
  return `${head("Overview", "one CLI · one SQLite file · one closed RL loop", "nanolab ui")}
    <div class="kpis">
      ${kpi(d.evals.done, "evals done", `${d.evals.active} active`)}
      ${kpi(best, "best reward", d.best ? esc(d.best.model) : "")}
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
            (e.installed_at || "").slice(0, 10)
          )} ${e.importable ? "" : '<span class="badge-missing">· missing from venv</span>'}</div>
          <div class="score">${
            e.best_reward !== null
              ? `${fmt(e.best_reward)} <span class="dim" style="font-size:11px">best · ${esc(
                  e.best_model ?? ""
                )}</span>`
              : '<span class="dim" style="font-size:12px">not evaluated yet</span>'
          }</div></div>`
        )
        .join("")}</div>`
    : empty("Install your first environment",
        "Environments are tasks plus automatic graders, in the standard verifiers format.",
        "nanolab env install primeintellect/gsm8k");
  return `${head("Environments", "tasks + automatic graders, hub-compatible",
    "nanolab env install <owner/name>")}${cards}`;
}

async function evals() {
  const runs = await api("/evals");
  const table = runs.length
    ? `<table><tr><th>run</th><th>env</th><th>model</th><th>status</th>
       <th>reward</th><th>samples</th><th>err</th><th>date</th></tr>
       ${runs
         .map(
           (e) => `<tr class="click" onclick="location.hash='#/evals/${e.id}'">
          <td class="num">#${e.id}</td><td class="mono">${esc(e.env)}</td>
          <td>${modelChip(e.model)}</td><td>${chip(e.status)}</td>
          <td class="num reward">${fmt(e.mean_reward)}</td>
          <td class="num">${e.num_examples}×${e.rollouts_per_example}</td>
          <td class="num">${e.errors}</td>
          <td class="num">${esc((e.finished_at || e.started_at || "").slice(0, 10))}</td></tr>`
         )
         .join("")}</table>
       <div class="fig">click a run for the rollout inspector</div>`
    : empty("Run your first evaluation",
        "An eval sends each task to a model and scores every answer with the environment's rubric.",
        "nanolab eval run gsm8k -m <model>");
  const done = runs.filter((r) => r.status === "done").length;
  return `${head("Evaluations", "rollouts + rubric scoring, cached and resumable",
    "nanolab eval run <env> -m <model>")}
    <div class="kpis">
      ${kpi(runs.filter((r) => ["running", "pending"].includes(r.status)).length,
        "active evals", "pending or running")}
      ${kpi(done, "successful evals", "completed")}
      ${kpi(runs.length, "total evals", "all statuses")}
    </div>${table}`;
}

async function evalDetail(id) {
  const d = await api(`/evals/${id}`);
  const metrics = Object.entries(d.meta.avg_metrics || {})
    .map(([k, v]) => `<span class="chip">${esc(k)} ${fmt(v)}</span>`)
    .join("");
  const rollouts = d.rollouts
    .map((r) => {
      const msgs = [...(r.prompt || []), ...(r.completion || [])];
      const convo = msgs
        .map(
          (m) => `<div class="msg ${esc(m.role)}"><div class="role">${esc(m.role)}</div>
           <pre>${esc(m.content)}</pre></div>`
        )
        .join("");
      const last = (r.completion || []).slice(-1)[0];
      const cls = (r.reward ?? 0) >= 0.5 ? "up" : "down";
      return `<details class="rollout"><summary>
        <span class="num mono">#${r.example}.${r.rollout}</span>
        <span class="reward ${cls}">${fmt(r.reward)}</span>
        <span class="preview">${esc(last ? last.content : "")}</span>
        ${r.metrics._error ? `<span class="chip failed">${esc(r.metrics._error)}</span>` : ""}
        </summary>${convo}</details>`;
    })
    .join("");
  return `<a class="back" href="#/evals">← Evaluations</a>
    ${head(`Eval run #${d.id}`, `${d.env} · ${d.model}`,
      `nanolab eval show ${d.id}`)}
    <div class="kpis">
      ${kpi(fmt(d.mean_reward), "mean reward", chip(d.status))}
      ${kpi(`${d.num_examples}×${d.rollouts_per_example}`, "examples × rollouts",
        d.seed !== null ? `seed ${d.seed}` : "no shuffle")}
      ${kpi(d.rollouts.length, "rollouts stored", `${d.rollouts.filter((r) => r.metrics._error).length} errors`)}
    </div>
    ${metrics ? `<div class="metric-chips">${metrics}</div>` : ""}
    <div class="section-label"><b>01</b>rollout inspector</div>${rollouts ||
      '<div class="empty">no samples stored</div>'}`;
}

async function training() {
  const runs = await api("/training");
  const table = runs.length
    ? `<table><tr><th>run</th><th>env</th><th>model</th><th>status</th>
       <th>steps</th><th>reward curve</th><th>first → last</th></tr>
       ${runs
         .map((t) => {
           const first = t.rewards[0], last = t.rewards[t.rewards.length - 1];
           const delta = t.rewards.length ? last - first : null;
           return `<tr class="click" onclick="location.hash='#/training/${t.id}'">
            <td class="num">#${t.id}</td><td class="mono">${esc(t.env ?? "?")}</td>
            <td>${modelChip(t.model)}</td><td>${chip(t.status)}</td>
            <td class="num">${t.steps_completed}</td><td>${curveSvg(t.rewards)}</td>
            <td class="num">${t.rewards.length ? `${fmt(first)} → ${fmt(last)}<br>
              <span class="${delta >= 0 ? "up" : "down"}">${delta >= 0 ? "↑" : "↓"} ${fmt(Math.abs(delta))}</span>` : "—"}</td></tr>`;
         })
         .join("")}</table>
       <div class="fig">shaded band = the 10–80% trainability window</div>`
    : empty("Run your first training run",
        "Training turns rewards into a LoRA adapter with GRPO — checkpointed, resumable.",
        "nanolab train configs/qwen3-0.6b-gsm8k.toml");
  return `${head("Training", "GRPO + LoRA, one synchronous loop",
    "nanolab train <config.toml>")}${table}`;
}

async function trainingDetail(id) {
  const d = await api(`/training/${id}`);
  const rewards = d.curve.map((p) => p.reward);
  const adapters = d.adapters.length
    ? `<table><tr><th>adapter</th><th>step</th><th>path</th><th>evaluate</th></tr>
       ${d.adapters
         .map(
           (a) => `<tr><td class="num">#${a.id}</td><td class="num">${a.step}</td>
           <td class="mono">${esc(a.path)}</td>
           <td>${cli(`nanolab eval run <env> -m ${a.base_model}:${a.id}`)}</td></tr>`
         )
         .join("")}</table>`
    : '<div class="empty">no checkpoints registered</div>';
  return `<a class="back" href="#/training">← Training</a>
    ${head(`Train run #${d.id}`, `${d.env ?? "?"} · ${d.model}`,
      `nanolab training show ${d.id}`)}
    <div class="kpis">
      ${kpi(d.steps_completed, "steps", chip(d.status))}
      ${kpi(rewards.length ? fmt(rewards[rewards.length - 1]) : "—", "last reward",
        rewards.length ? `from ${fmt(rewards[0])}` : "")}
      ${kpi(d.adapters.length, "checkpoints", "in adapters/")}
    </div>
    <div class="section-label"><b>01</b>reward curve</div>
    ${curveSvg(rewards, 680, 120)}
    <div class="fig">FIG.1 — reward vs step; shaded band = trainability window</div>
    <div class="section-label"><b>02</b>checkpoints</div>${adapters}
    <div class="section-label"><b>03</b>config</div>
    <pre class="toml">${esc(d.config_toml || "")}</pre>`;
}

async function inference() {
  const deps = await api("/deployments");
  const table = deps.length
    ? `<table><tr><th>id</th><th>model</th><th>endpoint</th><th>pid</th><th>status</th></tr>
       ${deps
         .map(
           (d) => `<tr><td class="num">#${d.id}</td>
           <td>${modelChip(`${d.base_model}:${d.adapter_id}`)}</td>
           <td class="mono">${esc(d.endpoint)}</td><td class="num">${d.pid ?? "—"}</td>
           <td>${chip(d.status)}</td></tr>`
         )
         .join("")}</table>`
    : empty("No deployments yet",
        "Deployments serve a trained adapter as an OpenAI-compatible endpoint (vLLM, CUDA box) so the eval station can measure it with a base:adapter model string.",
        "nanolab deployments create <adapter-id>");
  return `${head("Inference", "serve adapters, close the loop",
    "nanolab deployments create <adapter-id>")}${table}`;
}

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

async function render() {
  const hash = location.hash || "#/overview";
  for (const [re, fn, nav] of routes) {
    const m = hash.match(re);
    if (m) {
      current = () => fn(...m.slice(1));
      document
        .querySelectorAll("nav a")
        .forEach((a) => a.classList.toggle("active", a.dataset.route === nav));
      try {
        const openIds = [...document.querySelectorAll("details[open]")].map((d, i) => i);
        page.innerHTML = await current();
        // restore expanded rollouts across refreshes
        const details = document.querySelectorAll("details");
        openIds.forEach((i) => details[i] && (details[i].open = true));
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
  if (!document.hidden && current) render();
}, 5000);
