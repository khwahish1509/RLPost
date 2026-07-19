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

const head = (title, sub, action = "") =>
  `<div class="page-head"><div><h1>${esc(title)}</h1>
   <div class="sub">${esc(sub)}</div></div><div>${action}</div></div>`;

async function post(path, body) {
  const r = await fetch("/api" + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `${r.status}`);
  return data;
}

const jobsStrip = async () => {
  try {
    const jobs = await api("/jobs");
    const active = jobs.filter(
      (j) => j.status === "running" || j.status === "failed").slice(0, 4);
    if (!active.length) return "";
    return `<div class="activity">${active
      .map((j) => `<div class="job ${j.status}">
        ${j.status === "running"
          ? '<span class="st running"><i></i></span>'
          : '<span class="st failed"><i></i></span>'}
        <span class="mono">${esc(j.label)}</span>
        ${j.error ? `<span class="err" title="${esc(j.error)}">${esc(j.error)}</span>` : ""}
      </div>`).join("")}</div>`;
  } catch { return ""; }
};

window.togglePanel = (id) => {
  const el = document.getElementById(id);
  if (el) el.style.display = el.style.display === "none" ? "" : "none";
};

window.submitEval = async (btn) => {
  const panel = btn.closest(".panel");
  const val = (name) => panel.querySelector(`[name=${name}]`).value;
  btn.disabled = true;
  try {
    await post("/actions/eval", {
      env: val("env"), model: val("model"), n: val("n"), r: val("r"),
    });
    toast("eval started — it will appear below");
    setTimeout(render, 600);
  } catch (err) { toast(err.message); btn.disabled = false; }
};

window.submitInstall = async (btn) => {
  const input = btn.closest(".panel").querySelector("[name=slug]");
  if (!input.value.trim()) { toast("type an environment name"); return; }
  btn.disabled = true;
  try {
    await post("/actions/install", { slug: input.value });
    toast("installing — takes a minute");
    setTimeout(render, 600);
  } catch (err) { toast(err.message); btn.disabled = false; }
};

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
  return `${head("Overview", "one CLI · one SQLite file · one closed RL loop", cli("nanolab ui"))}
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
  const [envs, strip] = await Promise.all([api("/environments"), jobsStrip()]);
  const form = `<div class="panel" id="install-form" style="display:none">
    <div class="row">
      <div class="field"><label>environment name</label>
        <input name="slug" placeholder="primeintellect/gsm8k" style="min-width:240px"></div>
      <button class="btn" onclick="submitInstall(this)">Install</button>
    </div>
    <div class="hint">Anything from the public Environments Hub works —
      browse it at <b>app.primeintellect.ai</b> and paste the name here.</div>
  </div>`;
  const cards = envs.length
    ? `<div class="cards">${envs
        .map(
          (e) => `<div class="card click" style="cursor:pointer"
            onclick="location.hash='#/env/${esc(e.env_id)}'">
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
    `<button class="btn" onclick="togglePanel('install-form')">＋ Install environment</button>`)}
    ${strip}${form}${cards}`;
}

async function evals() {
  const [runs, envs, defaults, strip] = await Promise.all([
    api("/evals"), api("/environments"), api("/defaults"), jobsStrip(),
  ]);
  const form = `<div class="panel" id="eval-form" style="display:none">
    <div class="row">
      <div class="field"><label>environment</label>
        <select name="env">${envs
          .map((e) => `<option value="${esc(e.env_id)}">${esc(e.slug)}</option>`)
          .join("")}</select></div>
      <div class="field"><label>model</label>
        <input name="model" value="${esc(defaults.model)}" placeholder="model name"></div>
      <div class="field"><label>questions</label>
        <input name="n" type="number" value="5" min="1" max="50" style="min-width:70px"></div>
      <div class="field"><label>tries each</label>
        <input name="r" type="number" value="1" min="1" max="8" style="min-width:70px"></div>
      <button class="btn" onclick="submitEval(this)">Start</button>
    </div>
    <div class="hint">${defaults.key_present
      ? `Runs against <b>${esc(defaults.base_url)}</b> using your saved key. Costs a few cents of API credit.`
      : `<span style="color:var(--warn)">No API key configured — add one to the repo's .env first.</span>`}</div>
  </div>`;
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
  return `${head("Evaluations", "measure a model on an environment — click a row to inspect",
    `<button class="btn" onclick="togglePanel('eval-form')">＋ New evaluation</button>`)}
    ${strip}${form}
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
    ${head(`Eval run #${d.id}`, `${d.env} · ${d.model}`, cli(`nanolab eval show ${d.id}`))}
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
    cli("nanolab train <config.toml> --resume"))}${tbl}`;
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
      cli(`nanolab training show ${d.id}`))}
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
    cli("nanolab deployments create <adapter-id>"))}${tbl}`;
}

/* minimal markdown renderer: headings, fenced code, inline code, bold,
   links, bullet lists, paragraphs — enough for environment READMEs */
function md(src) {
  const blocks = String(src || "").split(/```/);
  let html = "";
  blocks.forEach((block, i) => {
    if (i % 2 === 1) {
      const body = block.replace(/^[a-z]*\n/, "");
      html += `<pre><code>${esc(body)}</code></pre>`;
      return;
    }
    const lines = block.split("\n");
    let list = false, para = [];
    const flush = () => {
      if (para.length) { html += `<p>${para.join(" ")}</p>`; para = []; }
    };
    const inline = (s) =>
      esc(s)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
        .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,
          '<a href="$2" target="_blank" style="color:var(--accent)">$1</a>');
    lines.forEach((line) => {
      const h = line.match(/^(#{1,3})\s+(.*)/);
      const li = line.match(/^\s*[-*]\s+(.*)/);
      if (h) { flush(); if (list) { html += "</ul>"; list = false; }
        html += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`; }
      else if (li) { flush(); if (!list) { html += "<ul>"; list = true; }
        html += `<li>${inline(li[1])}</li>`; }
      else if (!line.trim()) { flush(); if (list) { html += "</ul>"; list = false; } }
      else para.push(inline(line));
    });
    flush(); if (list) html += "</ul>";
  });
  return html;
}

let envTab = "overview";
let envFile = 0;
window.setEnvTab = (t) => { envTab = t; render(); };
window.setEnvFile = (i) => { envFile = i; render(); };

/* minimal python syntax highlighter — strings, comments, decorators,
   keywords, numbers; multiline-safe (triple-quoted strings are one token) */
function hlPy(src) {
  const re = new RegExp(
    [
      '"""[\\s\\S]*?"""', "'''[\\s\\S]*?'''",
      '"(?:\\\\.|[^"\\\\\\n])*"', "'(?:\\\\.|[^'\\\\\\n])*'",
      "#[^\\n]*", "@\\w[\\w.]*",
      "\\b\\d+(?:\\.\\d+)?\\b",
      "\\b(?:def|class|return|if|elif|else|for|while|import|from|as|with|" +
        "try|except|finally|raise|lambda|yield|async|await|pass|break|" +
        "continue|not|and|or|in|is|None|True|False|self)\\b",
    ].join("|"),
    "g"
  );
  let out = "", last = 0, m;
  while ((m = re.exec(src))) {
    out += esc(src.slice(last, m.index));
    const t = m[0];
    const cls =
      t[0] === "#" ? "tok-com"
      : t[0] === '"' || t[0] === "'" ? "tok-str"
      : t[0] === "@" ? "tok-dec"
      : /^\d/.test(t) ? "tok-num"
      : "tok-kw";
    out += `<span class="${cls}">${esc(t)}</span>`;
    last = re.lastIndex;
  }
  return out + esc(src.slice(last));
}

async function envDetail(id) {
  const d = await api(`/environments/${id}`);
  const tabs = ["overview", "code", "leaderboard"]
    .map((t) => `<button class="tab${envTab === t ? " active" : ""}"
      onclick="setEnvTab('${t}')">${t[0].toUpperCase() + t.slice(1)}${
      t === "leaderboard" ? ` (${d.leaderboard.length})` : ""}</button>`)
    .join("");

  let body = "";
  if (envTab === "overview") {
    const best = d.leaderboard[0];
    body = `<div class="repo">
      <div class="readme">${d.readme ? md(d.readme)
        : `<p class="dim">${esc(d.summary || "No README shipped with this package.")}</p>`}</div>
      <div class="about">
        <h4>about</h4>
        <div>${esc(d.summary || "—")}</div>
        <h4>version</h4><div class="mono">v${esc(d.version || "?")}</div>
        <h4>python</h4><div class="mono">${esc(d.requires_python || "—")}</div>
        <h4>best score</h4>
        <div class="mono">${best ? `${fmt(best.mean_reward)} · ${esc(best.model)}` : "not evaluated yet"}</div>
        <h4>installed</h4><div class="mono">${esc((d.installed_at || "").slice(0, 10))}</div>
        <h4>dependencies</h4>
        ${(d.requires || []).map((r) => `<span class="dep">${esc(r)}</span>`).join("") || "—"}
      </div></div>`;
  } else if (envTab === "code") {
    if (!d.files.length) {
      body = '<div class="empty">source not found in the venv</div>';
    } else {
      const i = Math.min(envFile, d.files.length - 1);
      const f = d.files[i];
      const lineCount = f.content.split("\n").length;
      const tree = d.files
        .map((x, j) => `<div class="fitem${j === i ? " active" : ""}"
          onclick="setEnvFile(${j})"><span>${esc(x.name)}</span>
          <span class="meta">${x.content.split("\n").length}L</span></div>`)
        .join("");
      const gutter = Array.from({ length: lineCount }, (_, k) => k + 1).join("\n");
      body = `<div class="codeview">
        <div class="filetree">${tree}</div>
        <div>
          <div class="codehead">${esc(f.name)}
            <span class="meta">· ${lineCount} lines · ${(f.content.length / 1024).toFixed(1)} KB</span></div>
          <div class="codepane"><div class="gutter">${gutter}</div>
            <pre class="src">${hlPy(f.content)}</pre></div>
        </div></div>`;
    }
  } else {
    body = d.leaderboard.length
      ? table(
          ["#", "model", "reward", "samples", "date"],
          d.leaderboard.map(
            (r, i) => `<tr class="click" onclick="location.hash='#/evals/${r.id}'">
             <td class="num rank">${i + 1}</td><td>${modelChip(r.model)}</td>
             <td>${reward(r.mean_reward)}</td>
             <td class="num">${r.num_examples}×${r.rollouts_per_example}</td>
             <td class="num dim">${esc((r.finished_at || "").slice(0, 10))}</td></tr>`))
      : empty("chart", "No evaluations yet",
          "Run one from the Evaluations page and it will rank here.");
  }
  return `<a class="back" href="#/environments">← Environments</a>
    ${head(d.slug, d.summary || "environment",
      `<button class="btn" onclick="location.hash='#/evals'">Evaluate</button>`)}
    <div class="tabs">${tabs}</div>${body}`;
}

async function guide() {
  return `${head("How to use nanolab", "the 2-minute tour — no terminal needed")}
  <div class="guide">
    <p>nanolab is a lab that <b>measures</b> AI models, <b>trains</b> them, and
    <b>measures again</b> to prove they improved. Everything you see is stored in
    one database file on your computer.</p>

    <h2>The loop, in plain words</h2>
    <div class="step"><b class="n">1</b><span><b>Environments</b> are exam papers —
      sets of tasks with automatic graders. Install more with the
      <i>＋ Install environment</i> button on the Environments page.</span></div>
    <div class="step"><b class="n">2</b><span><b>Evaluations</b> make a model take an
      exam and score every answer. Click <i>＋ New evaluation</i> on the Evaluations
      page, pick an environment, click <b>Start</b>. The run appears in the table in
      seconds; click it to read every question, answer, and score.</span></div>
    <div class="step"><b class="n">3</b><span><b>Training</b> makes a model practice an
      exam thousands of times, nudging it toward answers that score higher. This
      needs a free GPU from Google —
      <a href="https://colab.research.google.com/github/khwahish1509/RLPost/blob/main/notebooks/train_gsm8k_colab.ipynb"
      target="_blank" style="color:var(--accent)">open the one-click notebook</a>,
      press <b>Runtime → Run all</b>, and come back in ~4 hours. Results land here
      automatically once the artifacts are copied in.</span></div>
    <div class="step"><b class="n">4</b><span><b>Inference</b> serves a trained model
      so it can be measured again — the before/after comparison is the whole
      point.</span></div>

    <h2>What you can do right now, with clicks only</h2>
    <p>· Run <b>＋ New evaluation</b> (Evaluations page) — costs a few cents of API credit<br>
       · <b>＋ Install environment</b> (Environments page) — free<br>
       · Explore any past run — click rows, expand conversations, press
       <code>⌘K</code> to jump anywhere<br>
       · Watch a live run — pages refresh themselves every 5 seconds</p>

    <h2>What the <code>$ …</code> chips are</h2>
    <p class="muted">Every action also exists as a typed command (for people who use
    terminals). The chips show that command and copy it on click. You never need
    them — the buttons do the same thing.</p>

    <h2>When something needs more than a click</h2>
    <p class="muted">Ask Claude in the chat — "evaluate grok on gsm8k",
    "install the wordle environment", "why did this run fail" — and it happens
    in the same lab this app is showing you.</p>
  </div>`;
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
  [/^#\/env\/([\w.-]+)$/, envDetail, "environments"],
  [/^#\/evals$/, evals, "evals"],
  [/^#\/evals\/(\d+)$/, evalDetail, "evals"],
  [/^#\/training$/, training, "training"],
  [/^#\/training\/(\d+)$/, trainingDetail, "training"],
  [/^#\/inference$/, inference, "inference"],
  [/^#\/guide$/, guide, "guide"],
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
