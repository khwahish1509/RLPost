/* nanolab — platform UI (v0.3).
   One vanilla file: router + rooms, all data from /api.
   Rules: the UI never computes a metric — every number is read from the same
   SQLite rows the CLI reads, and every number can open its receipts. */

"use strict";

/* ---------------- plumbing ---------------- */

const $ = (s) => document.querySelector(s);
const page = $("#page");

async function api(path) {
  const r = await fetch("/api" + path);
  if (!r.ok) throw new Error((await r.text()).slice(0, 200) || r.status);
  return r.json();
}
async function post(path, body) {
  const r = await fetch("/api" + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `${r.status}`);
  return data;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function toast(msg, ms = 3200) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._h);
  toast._h = setTimeout(() => t.classList.remove("show"), ms);
}
const fnum = (v, d = 3) => (v == null ? "—" : Number(v).toFixed(d));
const when = (iso) => (iso ? iso.replace("T", " ").slice(0, 16) : "—");
const modelChip = (m) => `<span class="chip">${esc(m)}</span>`;

/* footnote mark: every number is a claim; the mark opens its raw rows */
const foot = (evalId) =>
  evalId == null ? "" :
  `<button class="foot" title="Click to see the raw runs behind this number" onclick="openReceipts(${evalId});event.stopPropagation()">°</button>`;

function figWrap(inner, no, text, margin) {
  return `<figure class="fig">${inner}
    <figcaption><span class="fig-no">${esc(no)}</span>
      <span class="fig-text">${text}</span>
      ${margin ? `<span class="fig-margin">${esc(margin)}</span>` : ""}
    </figcaption></figure>`;
}

/* reward-vs-step chart, dark axes. points: [{x, y}] */
function curveSvg(points, { w = 920, h = 250, yMax = 1.05, ticks = [] } = {}) {
  if (!points.length) return `<div class="empty"><div class="why">No data points yet.</div></div>`;
  const padL = 46, padR = 14, padT = 16, padB = 30;
  const xMax = Math.max(...points.map((p) => p.x), 1);
  const X = (x) => padL + ((w - padL - padR) * x) / xMax;
  const Y = (y) => padT + (h - padT - padB) * (1 - Math.min(y, yMax) / yMax);
  const path = points.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" ");
  const area = `${path} L${X(points[points.length - 1].x).toFixed(1)},${Y(0)} L${X(points[0].x).toFixed(1)},${Y(0)} Z`;
  const last = points[points.length - 1];
  const tickMarks = ticks.map((t) =>
    `<line x1="${X(t)}" y1="${h - padB}" x2="${X(t)}" y2="${h - padB + 5}" stroke="#9AA3AD" stroke-width="1"/>
     <text x="${X(t)}" y="${h - padB + 17}" font-family="Geist Mono,monospace" font-size="9.5" fill="#5E6670" text-anchor="middle">${t}</text>`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" stroke="#24272E" stroke-width="1"/>
    <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="#343943" stroke-width="1"/>
    <line x1="${padL}" y1="${Y(1)}" x2="${w - padR}" y2="${Y(1)}" stroke="#24272E" stroke-width="1" stroke-dasharray="3 5"/>
    <text x="${padL - 8}" y="${Y(1) + 3}" font-family="Geist Mono,monospace" font-size="10" fill="#5E6670" text-anchor="end">1.0</text>
    <text x="${padL - 8}" y="${h - padB + 3}" font-family="Geist Mono,monospace" font-size="10" fill="#5E6670" text-anchor="end">0.0</text>
    ${tickMarks}
    <path d="${area}" fill="rgba(183,245,66,.07)"/>
    <path d="${path}" fill="none" stroke="#B7F542" stroke-width="1.8"/>
    <circle cx="${X(last.x)}" cy="${Y(last.y)}" r="3.2" fill="#B7F542"/>
    <text x="${Math.min(X(last.x) + 8, w - padR - 34)}" y="${Y(last.y) - 8}" font-family="Geist Mono,monospace" font-size="10.5" fill="#E9ECEF">${fnum(last.y, 3)}</text>
  </svg>`;
}

/* calibration: the 10–80% trainable window as a number line */
function calibSvg(baseline, { w = 920, h = 74, lo = 0.1, hi = 0.8 } = {}) {
  const padL = 14, padR = 14, y = 40;
  const X = (v) => padL + (w - padL - padR) * v;
  const tick = baseline == null ? "" : `
    <line x1="${X(baseline)}" y1="${y - 22}" x2="${X(baseline)}" y2="${y + 10}" stroke="#B7F542" stroke-width="1.6"/>
    <text x="${X(baseline)}" y="${y - 28}" font-family="Geist Mono,monospace" font-size="11" fill="#B7F542" text-anchor="middle">${fnum(baseline, 3)}</text>`;
  return `<div class="calib"><svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">
    <rect x="${X(lo)}" y="${y - 12}" width="${X(hi) - X(lo)}" height="14" fill="rgba(183,245,66,.12)"/>
    <line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="#343943" stroke-width="1"/>
    ${tick}
    <text x="${X(0.02)}" y="${y + 24}" font-family="Geist Mono,monospace" font-size="9.5" fill="#5E6670">TOO HARD</text>
    <text x="${(X(lo) + X(hi)) / 2}" y="${y + 24}" font-family="Geist Mono,monospace" font-size="9.5" fill="#9AA3AD" text-anchor="middle">TRAINABLE — SWEET SPOT</text>
    <text x="${X(0.98)}" y="${y + 24}" font-family="Geist Mono,monospace" font-size="9.5" fill="#5E6670" text-anchor="end">ALREADY SOLVED</text>
  </svg></div>`;
}

function verdictBlock(claim, from, to, cap, { evalFrom, evalTo, small } = {}) {
  return `<div class="verdict${small ? " small" : ""}">
    <div class="claim">${claim}</div>
    <div class="nums">${from ? `<span class="from">${from}</span>${foot(evalFrom)}
      <span class="arr">→</span>` : ""}
      <span class="to">${to}</span>${foot(evalTo)}</div>
    <div class="cap">${esc(cap)}</div>
  </div>`;
}

/* markdown-lite for READMEs: headers, fenced code, inline code, bold */
function mdLite(src) {
  const lines = String(src || "").split("\n");
  let html = "", inCode = false;
  for (const ln of lines) {
    if (ln.trim().startsWith("```")) { html += inCode ? "</pre>" : "<pre>"; inCode = !inCode; continue; }
    if (inCode) { html += esc(ln) + "\n"; continue; }
    let l = esc(ln);
    l = l.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
    if (/^### /.test(ln)) html += `<h3>${l.slice(4)}</h3>`;
    else if (/^## /.test(ln)) html += `<h2>${l.slice(3)}</h2>`;
    else if (/^# /.test(ln)) html += `<h1>${l.slice(2)}</h1>`;
    else if (/^[-*] /.test(ln)) html += `<li>${l.slice(2)}</li>`;
    else if (ln.trim() === "") html += "<br>";
    else html += `<p>${l}</p>`;
  }
  return `<div class="md">${html}${inCode ? "</pre>" : ""}</div>`;
}

/* ---------------- receipts drawer ---------------- */

function closeDrawer() { document.body.classList.remove("drawer-open"); }
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeDrawer(); closeModal(); toggleConsole(false); }
});

async function openReceipts(evalId) {
  document.body.classList.add("drawer-open");
  $("#d-kicker").textContent = `RECEIPTS · EVAL #${evalId}`;
  $("#d-title").textContent = "Loading raw rows…";
  $("#d-body").innerHTML = "";
  try {
    const d = await api(`/evals/${evalId}`);
    $("#d-title").textContent = `${d.env} · ${d.model}`;
    const meta = Object.entries(d.meta?.avg_metrics || {})
      .map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td class="mono right">${fnum(v)}</td></tr>`).join("");
    const rows = (d.rollouts || []).map((r, i) => `
      <tr class="click" onclick="toggleConvo(${i})">
        <td class="mono">${r.example}·${r.rollout}</td>
        <td class="mono right num ${r.reward >= 0.5 ? "pos" : "neg"}">${fnum(r.reward)}</td>
      </tr>
      <tr id="convo-${i}" style="display:none"><td colspan="2">${convoHtml(r)}</td></tr>`).join("");
    $("#d-body").innerHTML = `
      <p class="aside-note">Every run behind this number, straight from the database.
      n=${d.num_examples} · r=${d.rollouts_per_example} · ${esc(d.status)} · started ${when(d.started_at)}.
      Click a row for the full exchange.</p>
      ${meta ? `<table class="sheet"><tr><th>METRIC</th><th class="right">MEAN</th></tr>${meta}</table>` : ""}
      <table class="sheet"><tr><th>RUN</th><th class="right">SCORE</th></tr>${rows}</table>`;
  } catch (e) {
    $("#d-body").innerHTML = `<div class="empty"><div class="why">Could not load: ${esc(e.message)}</div></div>`;
  }
}
window.openReceipts = openReceipts;
window.closeDrawer = closeDrawer;
window.toggleConvo = (i) => {
  const el = $(`#convo-${i}`);
  if (el) el.style.display = el.style.display === "none" ? "" : "none";
};
function convoHtml(r) {
  const msgs = [...(Array.isArray(r.prompt) ? r.prompt : []), ...(Array.isArray(r.completion) ? r.completion : [])];
  if (!msgs.length) return `<div class="aside-note">no stored conversation</div>`;
  return `<div class="convo">${msgs.map((m) => `
    <div class="msg ${esc(m.role)}"><div class="role">${esc(m.role)}</div>
    <pre>${esc(typeof m.content === "string" ? m.content : JSON.stringify(m.content))}</pre></div>`).join("")}</div>`;
}

/* ---------------- modal ---------------- */

function openModal(html) { $("#modal").innerHTML = html; $("#modal-veil").classList.add("open"); }
function closeModal() { $("#modal-veil").classList.remove("open"); }
$("#modal-veil").addEventListener("click", (e) => { if (e.target.id === "modal-veil") closeModal(); });
window.closeModal = closeModal;

async function newEvalModal(prefillEnv = "") {
  const [envs, defaults] = await Promise.all([api("/environments"), api("/defaults")]);
  const opts = envs.map((e) =>
    `<option value="${esc(e.slug)}" ${e.slug === prefillEnv ? "selected" : ""}>${esc(e.slug)}</option>`).join("");
  openModal(`
    <h3>Run an evaluation</h3>
    <div class="m-sub">Runs the model through the task and averages the grades.
    The score lands in Evals with full receipts.</div>
    <label class="f">Task</label><select id="ev-env">${opts}</select>
    <label class="f">Model</label><input id="ev-model" value="${esc(defaults.model || "")}">
    <div class="helper">Base models and your trained adapters both work here.</div>
    <div class="form-grid">
      <div><label class="f">Problems (n)</label><input id="ev-n" type="number" value="5" min="1"></div>
      <div><label class="f">Tries per problem</label><input id="ev-r" type="number" value="1" min="1"></div>
    </div>
    <div class="helper">More problems = steadier score, longer wait.</div>
    <label class="f">Temperature</label><input id="ev-t" type="number" step="0.1" value="0.0">
    <div class="m-foot">
      <button class="btn ghost" onclick="closeModal()">Cancel</button>
      <button class="btn" onclick="submitEval()">Run evaluation</button>
    </div>`);
}
window.newEvalModal = newEvalModal;
window.submitEval = async () => {
  try {
    await post("/actions/eval", {
      env: $("#ev-env").value, model: $("#ev-model").value,
      n: +$("#ev-n").value, r: +$("#ev-r").value, temperature: +$("#ev-t").value,
    });
    closeModal(); toast("Evaluation started — it will appear in Evals");
  } catch (e) { toast("Could not start: " + e.message, 5000); }
};

async function newTrainModal() {
  const configs = await api("/configs");
  const opts = configs.map((c) => `<option value="${esc(c.path)}">${esc(c.name || c.path)}</option>`).join("");
  openModal(`
    <h3>Start training</h3>
    <div class="m-sub">Training method: GRPO with LoRA — the model tries the task many
    times and higher-scoring answers get reinforced. Runs on Kaggle's free GPU;
    a typical run takes 1–3 hours. You can close the laptop after launch.</div>
    <label class="f">Training recipe</label><select id="tr-config">${opts}</select>
    <div class="helper">Sweet spot: the base model gets 10–80% right before training.
    Below that there's nothing to build on; above it, little room to improve.</div>
    <div class="m-foot">
      <button class="btn ghost" onclick="closeModal()">Cancel</button>
      <button class="btn" onclick="submitTrain()">Start on free GPU</button>
    </div>`);
}
window.newTrainModal = newTrainModal;
window.submitTrain = async () => {
  try {
    await post("/actions/train-cloud", { config: $("#tr-config").value });
    closeModal(); toast("Pushed to Kaggle — watch it in Train");
  } catch (e) { toast("Could not launch: " + e.message, 5000); }
};

/* ---------------- top strip + sidebar footer ---------------- */

let lab = { overview: null, cloud: [] };

async function refreshChrome() {
  try {
    const [ov, cloud, ver] = await Promise.all([api("/overview"), api("/cloud"), api("/version")]);
    lab = { overview: ov, cloud };
    const active = cloud.filter((c) => ["pushed", "running"].includes(c.status));
    $("#rec").style.display = active.length || ov.training.active ? "" : "none";

    const t = ov.recent_trains?.[0];
    const segs = [];
    if (t) {
      segs.push(`<span class="seg"><b>RUN ${t.id}</b> · ${esc(t.slug || "?")}</span>`);
      if (active.length) segs.push(`<span class="arrow">→</span><span class="seg now">TRAINING ON KAGGLE ⋯</span>`);
      else segs.push(`<span class="arrow">→</span><span class="seg">${t.status === "done" ? `<span class="done-mark">✓</span> ${t.steps_completed} STEPS` : esc(String(t.status).toUpperCase())}</span>`);
      const verd = (ov.recent_evals || []).find((e) => e.slug === t.slug && e.status === "done");
      if (verd) segs.push(`<span class="arrow">→</span><span class="seg">LATEST SCORE <b>${fnum(verd.mean_reward)}</b></span>`);
    }
    $("#strip").innerHTML = segs.join("");
    $("#side-foot").innerHTML =
      `v0.3 · ui ${esc(String(ver.ui || "").slice(0, 8))}<br>` +
      `${ov.environments} tasks · ${ov.evals.total} evals · ${ov.adapters} adapters`;
  } catch { /* chrome is decoration; rooms surface real errors */ }
}

/* ================= rooms ================= */

/* ---------------- HOME ---------------- */

async function home() {
  const [ov, jobs] = await Promise.all([api("/overview"), api("/jobs")]);
  const t = ov.recent_trains?.[0];
  const cloudActive = lab.cloud.some((c) => ["pushed", "running"].includes(c.status));

  const ctas = `
    <div class="cta-cards">
      <a class="cta" onclick="newTrainModal()">
        <div class="t"><span class="ico">▲</span>Train a model</div>
        <div class="s">Pick a task and a base model. One click starts training on a free cloud GPU.</div>
      </a>
      <a class="cta" href="#/memory">
        <div class="t"><span class="ico">✎</span>Train the Scribe</div>
        <div class="s">Our note-taking model went from 0.55 to a perfect score. Watch it work, or retrain it.</div>
      </a>
      <a class="cta" href="#/tasks">
        <div class="t"><span class="ico">◫</span>Browse tasks</div>
        <div class="s">Install tasks from the hub. Each one grades model answers automatically.</div>
      </a>
      <a class="cta" href="#/models">
        <div class="t"><span class="ico">◎</span>Use your models</div>
        <div class="s">Turn on a trained model and get a local endpoint with copy-paste code.</div>
      </a>
    </div>`;

  const liveFig = t?.rewards?.length
    ? figWrap(curveSvg(t.rewards.map((y, x) => ({ x, y }))),
        "LIVE", `Reward per step — run ${t.id} on ${esc(t.slug || "?")} (${esc(t.status)}). Up and to the right means it's learning.`,
        `${t.rewards.length} steps`)
    : "";

  const jobRows = (jobs.running || []).concat(jobs.recent || []).slice(0, 4).map((j) => `
    <div class="job"><span class="status ${j.status === "running" ? "running" : j.status === "error" ? "error" : "done"}">${esc(j.status)}</span>
      <span>${esc(j.label)}</span>
      ${j.status !== "running" ? `<button class="x" onclick="dismissJob('${esc(j.id)}')">dismiss</button>` : ""}</div>`).join("");

  const recent = (ov.recent_evals || []).slice(0, 6).map((e) => `
    <tr class="click" onclick="location.hash='#/evals/${e.id}'">
      <td class="mono">#${e.id}</td><td class="mono">${esc(e.slug)}</td>
      <td>${modelChip(e.model)}</td>
      <td class="mono right num">${fnum(e.mean_reward)}${e.status === "done" ? foot(e.id) : ""}</td>
      <td class="right"><span class="status ${esc(e.status)}">${esc(e.status)}</span></td>
    </tr>`).join("");

  return `
    <div class="room-head">
      <h1>Home</h1>
      <p class="sub">What's running, what's trained, and what to do next.
      Everything runs on your machine — every score has receipts.</p>
    </div>
    ${cloudActive ? `<div class="jobs"><div class="job"><span class="status running">training</span>
      <span>Training is running on Kaggle's free GPU. Nothing to do — the result lands here automatically.</span></div></div>` : ""}
    ${jobRows ? `<div class="jobs">${jobRows}</div>` : ""}
    ${ctas}
    ${liveFig}
    <div class="section">
      <div class="sec-head"><h2>Recent evaluations</h2></div>
      ${recent ? `<table class="sheet"><tr><th>EVAL</th><th>TASK</th><th>MODEL</th><th class="right">SCORE</th><th class="right">STATUS</th></tr>${recent}</table>`
        : `<div class="empty"><div class="why">Nothing here yet. nanolab trains small models on tasks with
           automatic graders — pick a task, measure a model, train it, serve the result.</div>
           <button class="btn" onclick="location.hash='#/tasks'">Browse tasks</button></div>`}
    </div>`;
}
window.dismissJob = async (id) => { try { await post("/actions/dismiss-job", { id }); } catch {} };

/* ---------------- TASKS (installed + hub browse) ---------------- */

const hub = { tab: "installed", q: "", sort: "stars", page: 1, rows: [], error: "", loading: false, more: true };

async function tasks() {
  if (hub.tab === "hub" && !hub.rows.length && !hub.error && !hub.loading) hubLoad(true);
  const envs = await api("/environments");

  const tabs = `
    <div class="tabs">
      <button class="${hub.tab === "installed" ? "active" : ""}" onclick="hubTab('installed')">Installed (${envs.length})</button>
      <button class="${hub.tab === "hub" ? "active" : ""}" onclick="hubTab('hub')">Browse hub</button>
    </div>`;

  let body;
  if (hub.tab === "installed") {
    const cards = envs.map((e) => `
      <div class="card">
        <h3><a href="#/task/${encodeURIComponent(e.slug)}">${esc(e.slug)}</a></h3>
        <div class="desc">${esc(e.summary || "A task in verifiers format.")}</div>
        ${calibSvg(e.best_reward, { h: 60 })}
        <div class="meta">
          <span title="Your best evaluation score on this task, 0 to 1">${e.best_reward != null ? `BEST ${fnum(e.best_reward)} · ${esc(String(e.best_model || "").slice(0, 20))}` : "NOT YET MEASURED"}</span>
          <span>v${esc(e.version || "?")}</span>
        </div>
      </div>`).join("");
    body = cards ? `<div class="cards">${cards}</div>`
      : `<div class="empty"><div class="why">No tasks installed. Browse the hub to find one —
         install takes about a minute.</div>
         <button class="btn" onclick="hubTab('hub')">Browse hub</button></div>`;
  } else {
    const cards = hub.rows.map((e) => {
      const slug = e.environment || "";
      const tags = (e.tags || []).slice(0, 4).map((t) => `<span class="tag">${esc(t)}</span>`).join("");
      return `
      <div class="card">
        <h3><a href="#/task/${encodeURIComponent(slug)}">${esc(slug)}</a>
          ${e.installed ? `<span class="chip acc" style="margin-left:8px">Installed</span>` : ""}</h3>
        <div class="desc">${esc(e.description || "")}</div>
        <div>${tags}</div>
        <div class="meta">
          <span class="stars" title="Stars on the hub">★ ${e.stars ?? 0}</span>
          <span>v${esc(e.version || "?")}
            ${e.installed ? "" : `<button class="btn sm" style="margin-left:10px" onclick="hubInstall('${esc(slug)}', this)">Install</button>`}</span>
        </div>
      </div>`;
    }).join("");
    body = `
      <div class="hub-bar">
        <input type="search" id="hub-q" placeholder="Search tasks — math, coding, games…"
          value="${esc(hub.q)}" oninput="hubSearch(this.value)">
        <select onchange="hubSort(this.value)">
          <option value="stars" ${hub.sort === "stars" ? "selected" : ""}>Most starred</option>
          <option value="updated" ${hub.sort === "updated" ? "selected" : ""}>Newest</option>
          <option value="name" ${hub.sort === "name" ? "selected" : ""}>A–Z</option>
        </select>
      </div>
      ${hub.error ? `<div class="empty"><div class="why">${esc(hub.error)}</div>
        <button class="btn ghost" onclick="hubLoad(true)">Retry</button></div>` : ""}
      <div class="cards">${cards}</div>
      ${hub.loading ? `<div class="load-more dim">Loading tasks…</div>` :
        hub.rows.length && hub.more ? `<div class="load-more">
          <button class="btn ghost" onclick="hubMore()">Load more</button></div>` : ""}`;
  }

  return `
    <div class="room-head">
      <h1>Tasks</h1>
      <p class="sub">Tasks with built-in graders. Install one, then evaluate or train against it.
      A task gives a model problems and scores the answers automatically — no hand-grading.</p>
    </div>
    ${tabs}${body}`;
}
window.hubTab = (t) => { hub.tab = t; render(true); };
window.hubSort = (s) => { hub.sort = s; hubLoad(true); };
let hubSearchTimer = null;
window.hubSearch = (q) => {
  hub.q = q;
  clearTimeout(hubSearchTimer);
  hubSearchTimer = setTimeout(() => hubLoad(true), 350);
};
async function hubLoad(reset) {
  if (reset) { hub.page = 1; hub.rows = []; hub.more = true; }
  hub.loading = true; hub.error = "";
  render(true);
  try {
    const d = await api(`/hub?search=${encodeURIComponent(hub.q)}&sort=${hub.sort}&page=${hub.page}`);
    if (d.error) hub.error = d.error;
    const batch = d.environments || [];
    hub.rows = hub.page === 1 ? batch : hub.rows.concat(batch);
    hub.more = batch.length >= 48;
  } catch (e) { hub.error = "Hub unreachable: " + e.message; }
  hub.loading = false;
  render(true);
  if (hub.tab === "hub") { const q = $("#hub-q"); if (q && hub.q) { q.focus(); q.setSelectionRange(q.value.length, q.value.length); } }
}
window.hubLoad = hubLoad;
window.hubMore = () => { hub.page += 1; hubLoad(false); };
window.hubInstall = async (slug, btn) => {
  if (btn) { btn.disabled = true; btn.textContent = "Installing…"; }
  toast("Installing… this fetches the task code and its dependencies.");
  try {
    await post("/actions/install", { slug });
    toast("Installing in background — it appears under Installed when ready.");
  } catch (e) { toast("Could not install: " + e.message, 5000); if (btn) { btn.disabled = false; btn.textContent = "Install"; } }
};

/* ---------------- TASK DETAIL (installed or hub) ---------------- */

const taskView = { tab: "about", file: null, fileContent: "", fileLoading: false };

async function taskDetail(rawSlug) {
  const slug = decodeURIComponent(rawSlug);
  const envs = await api("/environments");
  const installedRow = envs.find((e) => e.slug === slug);
  if (installedRow) return installedTask(slug);
  if (slug.includes("/")) return hubTask(slug);
  return `<div class="empty"><div class="why">Task "${esc(slug)}" not found.</div></div>`;
}

function repoTabs(active, hasResults) {
  const tab = (id, label) => `<button class="${active === id ? "active" : ""}" onclick="taskTab('${id}')">${label}</button>`;
  return `<div class="tabs">${tab("about", "About")}${tab("files", "Files")}${hasResults ? tab("results", "Results") : ""}</div>`;
}
window.taskTab = (t) => { taskView.tab = t; taskView.file = null; render(true); };

async function installedTask(slug) {
  const d = await api(`/environments/${encodeURIComponent(slug)}`);
  const lb = (d.leaderboard || []).map((r, i) => `
    <tr class="click" onclick="location.hash='#/evals/${r.id}'">
      <td class="mono">${i + 1}</td><td>${modelChip(r.model)}</td>
      <td class="mono right num">${fnum(r.mean_reward)}${foot(r.id)}</td>
      <td class="mono right">${r.num_examples ?? "—"}</td></tr>`).join("");

  let body = "";
  if (taskView.tab === "about")
    body = d.readme ? mdLite(d.readme) : `<div class="empty"><div class="why">No README in this task's package.</div></div>`;
  else if (taskView.tab === "files") {
    const files = (d.files || []);
    if (taskView.file != null) {
      const f = files.find((x) => x.name === taskView.file);
      body = `<div style="margin-bottom:10px"><button class="btn sm ghost" onclick="taskView.file=null;render(true)">← All files</button>
        <span class="mono dim" style="margin-left:10px">${esc(taskView.file)}</span></div>
        <pre class="codeview">${esc(f?.content || "(empty)")}</pre>`;
    } else {
      body = `<div class="filelist">${files.map((f) => `
        <div class="frow" onclick="taskView.file='${esc(f.name)}';render(true)">
          <span>${esc(f.name)}</span><span class="dim">view →</span></div>`).join("") ||
        `<div class="frow">no files recorded</div>`}</div>`;
    }
  } else
    body = lb ? `<p class="helper" style="margin-bottom:10px">Every model you've measured on this task, best first.</p>
      <table class="sheet"><tr><th>RANK</th><th>MODEL</th><th class="right">SCORE</th><th class="right">N</th></tr>${lb}</table>`
      : `<div class="empty"><div class="why">No runs yet on this task. Run an evaluation to get a first score.</div>
         <button class="btn" onclick="newEvalModal('${esc(slug)}')">Run evaluation</button></div>`;

  return `
    <div class="kicker"><a href="#/tasks">TASKS</a> · INSTALLED</div>
    <div class="repo-head">
      <h1>${esc(slug)}</h1>
      <span class="chip">v${esc(d.version || "?")}</span>
      <span class="chip acc">Installed</span>
    </div>
    <p class="sub">${esc(d.summary || "")}</p>
    <div class="room-head"><div class="actions">
      <button class="btn" onclick="newEvalModal('${esc(slug)}')">Run evaluation</button>
      <button class="btn ghost" onclick="newTrainModal()">Train on this task</button>
    </div></div>
    ${repoTabs(taskView.tab, true)}
    ${body}`;
}

async function hubTask(slug) {
  const [owner, name] = slug.split("/");
  const d = await api(`/hub/${encodeURIComponent(owner)}/${encodeURIComponent(name)}`);
  const tags = (d.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join("");

  let body = "";
  if (taskView.tab === "files") {
    if (taskView.file != null) {
      body = `<div style="margin-bottom:10px"><button class="btn sm ghost" onclick="taskView.file=null;render(true)">← All files</button>
        <span class="mono dim" style="margin-left:10px">${esc(taskView.file)}</span></div>
        <pre class="codeview">${taskView.fileLoading ? "Loading…" : esc(taskView.fileContent)}</pre>`;
    } else {
      body = `<div class="filelist">${(d.files || []).map((f) => `
        <div class="frow" onclick="hubFile('${esc(slug)}','${esc(f.name)}')">
          <span>${esc(f.name)}</span><span class="dim">${esc(f.size || "")} · view →</span></div>`).join("") ||
        `<div class="frow">no files visible</div>`}</div>`;
    }
  } else {
    body = d.readme ? mdLite(d.readme) : `<div class="empty"><div class="why">This task has no README.</div></div>`;
  }

  return `
    <div class="kicker"><a href="#/tasks">TASKS</a> · HUB</div>
    <div class="repo-head">
      <h1>${esc(slug)}</h1>
      <span class="chip">v${esc(d.version || "?")}</span>
      <span class="stars">★ ${d.stars ?? 0}</span>
    </div>
    <p class="sub">${esc(d.description || "")}</p>
    <div style="margin:8px 0 2px">${tags}</div>
    <div class="room-head"><div class="actions">
      ${d.installed
        ? `<span class="chip acc">Installed</span>`
        : `<button class="btn" onclick="hubInstall('${esc(slug)}')">Install task</button>`}
      <span class="helper" style="align-self:center">Viewing from the hub — you can read everything
        without installing. Install to run it.</span>
    </div></div>
    ${repoTabs(taskView.tab === "results" ? "about" : taskView.tab, false)}
    ${body}`;
}
window.hubFile = async (slug, name) => {
  taskView.file = name; taskView.fileContent = ""; taskView.fileLoading = true;
  render(true);
  try {
    const [owner, envName] = slug.split("/");
    const d = await api(`/hub/${encodeURIComponent(owner)}/${encodeURIComponent(envName)}/file?path=${encodeURIComponent(name)}`);
    taskView.fileContent = d.content || "(empty)";
  } catch (e) { taskView.fileContent = "Could not load: " + e.message; }
  taskView.fileLoading = false;
  render(true);
};

/* ---------------- EVALS ---------------- */

async function evals() {
  const runs = await api("/evals");
  const rows = runs.map((e) => `
    <tr class="click" onclick="location.hash='#/evals/${e.id}'">
      <td class="mono">#${e.id}</td><td class="mono">${esc(e.env)}</td>
      <td>${modelChip(e.model)}</td>
      <td class="mono right">${e.num_examples}×${e.rollouts_per_example}</td>
      <td class="mono right num ${e.mean_reward >= 0.5 ? "pos" : e.mean_reward != null ? "neg" : ""}"
        title="Average grade across all runs, 0 to 1">${fnum(e.mean_reward)}${e.status === "done" ? foot(e.id) : ""}</td>
      <td class="right"><span class="status ${esc(e.status)}">${esc(e.status)}</span></td>
    </tr>`).join("");
  return `
    <div class="room-head">
      <h1>Evaluations</h1>
      <p class="sub">Measure how well a model does on a task. Every score is backed by raw runs —
      click any score to see them.</p>
      <div class="actions"><button class="btn" onclick="newEvalModal()">Run evaluation</button></div>
    </div>
    ${rows ? `<table class="sheet">
      <tr><th>EVAL</th><th>TASK</th><th>MODEL</th><th class="right">N×R</th><th class="right">SCORE</th><th class="right">STATUS</th></tr>
      ${rows}</table>`
      : `<div class="empty"><div class="why">No evaluations yet. An evaluation runs a model through a task
         many times and averages the grades — pick a task and a model to get your first number.</div>
         <button class="btn" onclick="newEvalModal()">Run evaluation</button></div>`}`;
}

async function evalDetail(id) {
  const d = await api(`/evals/${id}`);
  if (!d) return `<div class="empty"><div class="why">Eval not found.</div></div>`;
  const metrics = Object.entries(d.meta?.avg_metrics || {}).map(([k, v]) =>
    `<tr><td class="mono">${esc(k)}</td><td class="mono right num">${fnum(v)}</td></tr>`).join("");
  const rollouts = (d.rollouts || []).map((r, i) => `
    <tr class="click" onclick="toggleRow(${i})">
      <td class="mono">${r.example}·${r.rollout}</td>
      <td class="mono right num ${r.reward >= 0.5 ? "pos" : "neg"}">${fnum(r.reward)}</td>
    </tr><tr id="row-${i}" style="display:none"><td colspan="2">${convoHtml(r)}</td></tr>`).join("");
  return `
    <div class="kicker"><a href="#/evals">EVALS</a> · #${d.id}</div>
    <div class="room-head">
      <h1>${esc(d.model)} on ${esc(d.env)}</h1>
      <p class="sub">Every run, its answer, and its grade. This is the receipt. Click a row for the full exchange.</p>
    </div>
    ${verdictBlock(`Score on ${esc(d.env)}`, "", fnum(d.mean_reward),
      `n=${d.num_examples} · r=${d.rollouts_per_example} · ${d.status} · ${when(d.started_at)}`,
      { small: true })}
    ${metrics ? `<div class="section"><div class="sec-head"><h2>Metrics</h2></div>
      <table class="sheet"><tr><th>METRIC</th><th class="right">MEAN</th></tr>${metrics}</table></div>` : ""}
    <div class="section"><div class="sec-head"><h2>Runs</h2></div>
      <table class="sheet"><tr><th>RUN</th><th class="right">SCORE</th></tr>${rollouts ||
        `<tr><td colspan="2"><div class="aside-note">Runs appear here as they finish.</div></td></tr>`}</table>
    </div>`;
}
window.toggleRow = (i) => {
  const el = $(`#row-${i}`);
  if (el) el.style.display = el.style.display === "none" ? "" : "none";
};

/* ---------------- TRAIN ---------------- */

async function train() {
  const [runs, cloud] = await Promise.all([api("/training"), api("/cloud")]);
  const cloudRows = cloud.slice(0, 5).map((c) => `
    <div class="job"><span class="status ${esc(c.status)}">${esc(c.status)}</span>
      <span class="mono">${esc(c.kernel_ref)}</span>
      <span style="margin-left:auto" class="mono dim">${when(c.created_at)}</span></div>`).join("");
  const rows = runs.map((t) => {
    const pts = (t.rewards || []).map((y, x) => ({ x, y }));
    const spark = pts.length ? curveSvg(pts, { w: 240, h: 56 }) : "";
    return `
    <tr class="click" onclick="location.hash='#/train/${t.id}'">
      <td class="mono">#${t.id}</td><td class="mono">${esc(t.env || "?")}</td>
      <td>${modelChip(t.model)}</td>
      <td style="width:250px">${spark}</td>
      <td class="mono right">${t.steps_completed ?? 0}</td>
      <td class="right"><span class="status ${esc(t.status)}">${esc(t.status)}</span></td>
    </tr>`;
  }).join("");
  return `
    <div class="room-head">
      <h1>Training</h1>
      <p class="sub">Teach a model to score higher on a task. GRPO with LoRA on a free cloud GPU —
      the model tries the task many times, and better answers get reinforced.</p>
      <div class="actions"><button class="btn" onclick="newTrainModal()">Start training</button></div>
    </div>
    ${cloudRows ? `<div class="section" style="margin-top:0"><div class="sec-head"><h2>Cloud runs</h2></div>
      <div class="jobs">${cloudRows}</div></div>` : ""}
    ${rows ? `<table class="sheet">
      <tr><th>RUN</th><th>TASK</th><th>MODEL</th><th>CURVE</th><th class="right">STEPS</th><th class="right">STATUS</th></tr>
      ${rows}</table>`
      : `<div class="empty"><div class="why">No training runs yet. Pick a task where the base model
         scores between 10% and 80% — nanolab handles the GPU, the loop, and saving the result.</div>
         <button class="btn" onclick="newTrainModal()">Start training</button></div>`}`;
}

async function trainDetail(id) {
  const d = await api(`/training/${id}`);
  if (!d) return `<div class="empty"><div class="why">Run not found.</div></div>`;
  const pts = (d.curve || []).map((p) => ({ x: p.step, y: p.reward }));
  const ticks = (d.adapters || []).map((a) => a.step);
  const ckpts = (d.adapters || []).map((a) => `
    <tr><td class="mono">step ${String(a.step).padStart(3, "0")}</td>
      <td class="mono dim">${esc(a.path)}</td>
      <td class="mono right dim">${when(a.created_at)}</td></tr>`).join("");
  return `
    <div class="kicker"><a href="#/train">TRAINING</a> · RUN ${d.id}</div>
    <div class="room-head">
      <h1>${esc(d.env || "?")} · ${esc(d.model)}</h1>
      <p class="sub">${esc(d.status)} · ${d.steps_completed ?? 0} steps · started ${when(d.started_at)}.
      Reward over training steps — up and to the right means it's learning.</p>
      ${d.status === "done" ? `<div class="actions">
        <button class="btn" onclick="newEvalModal()">Evaluate result</button>
        <button class="btn ghost" onclick="location.hash='#/models'">Serve this model</button></div>` : ""}
    </div>
    ${pts.length ? figWrap(curveSvg(pts, { ticks }),
      `RUN ${d.id}`, "Reward per step. Tick marks are saved checkpoints — any can be served or evaluated.",
      "checkpoint every 10 steps") : `<div class="empty"><div class="why">No curve recorded.</div></div>`}
    ${ckpts ? `<div class="section"><div class="sec-head"><h2>Checkpoints</h2></div>
      <table class="sheet"><tr><th>CHECKPOINT</th><th>PATH</th><th class="right">SAVED</th></tr>${ckpts}</table>
      <p class="helper">Serve a checkpoint from Models, or evaluate it against the base to see the lift.</p></div>` : ""}`;
}

/* ---------------- MEMORY (the Scribe agent) ---------------- */

let agentBusy = false;

async function memory() {
  const s = await api("/agent");
  const writerBtn = (w, label) => `
    <button class="writer-btn ${s.writer === w ? "sel" : ""}" onclick="setWriter('${w}')">${label}</button>`;

  const msgs = (s.messages || []).map((m) => {
    if (m.role === "user")
      return `<div class="msg user"><div class="role">you</div><pre>${esc(m.content)}</pre></div>`;
    const control = m.control ? `
      <details class="control"><summary>what a no-memory model would have said…</summary>
        <pre>${esc(m.control)}</pre></details>` : "";
    return `<div class="msg assistant"><div class="role">agent · writer: ${esc(m.writer || "?")}</div>
      <pre>${esc(m.content)}</pre>${control}</div>`;
  }).join("");

  const scribeState =
    s.scribe_state === "rewriting"
      ? `<span class="rec"><span class="dot"></span>SCRIBE REWRITING</span>`
      : s.scribe_state?.startsWith("error")
        ? `<span class="status error">${esc(s.scribe_state)}</span>`
        : s.writer === "trained" && !s.scribe_ready
          ? `<span class="chip">scribe model loads on first message (~15s)</span>`
          : "";

  const pct = Math.min(100, Math.round(((s.notebook || "").length / 400) * 100));

  return `
    <div class="room-head">
      <h1>The Scribe</h1>
      <p class="sub">A model this lab trained to take notes. The assistant remembers <i>nothing</i>
      between turns except a 400-character notebook, rewritten after every exchange by the
      writer you choose. Trained score on held-out memory tests: 0.548 → 1.000${foot(24)}.
      <a href="#/findings" style="color:var(--accent)">How it was trained →</a></p>
    </div>

    <div class="writer-row">
      <span class="mono dim" style="font-size:10.5px; letter-spacing:.1em">MEMORY WRITER</span>
      ${writerBtn("trained", "Trained Scribe")}
      ${writerBtn("prompted", "Prompted (API)")}
      ${writerBtn("none", "No memory")}
      <span style="flex:1"></span>
      ${scribeState}
      <button class="btn sm ghost" onclick="resetAgent()">New session</button>
    </div>

    <div class="agent-grid">
      <div class="agent-chat">
        <div class="convo agent-log" id="agent-log">
          ${msgs || `<div class="msg"><div class="role">how to use</div>
            <pre>Tell it a few facts. Ask about them later. Each reply is written
from the notebook alone — expand the line under each answer to see
what a no-memory model would have said instead.</pre></div>`}
        </div>
        <form class="console-bar" onsubmit="agentSend(event)">
          <input id="agent-in" autocomplete="off" spellcheck="false"
            placeholder="${agentBusy ? "thinking…" : "Tell the Scribe something to remember…"}"
            ${agentBusy ? "disabled" : ""}>
          <button class="btn" type="submit" ${agentBusy ? "disabled" : ""}>Send</button>
        </form>
      </div>
      <div>
        <div class="notebook-card">
          <div class="nb-head"><span>THE NOTEBOOK</span>
            <span class="mono">${(s.notebook || "").length}/400</span></div>
          <div class="nb-meter"><div style="width:${pct}%"></div></div>
          <pre class="nb-body">${esc(s.notebook || "(empty — nothing recorded yet)")}</pre>
          <div class="nb-foot">full rewrite each turn · cap enforced in code · writer: ${esc(s.writer)}</div>
        </div>
        <p class="helper" style="margin-top:12px">
          This is the same 400-character budget the Scribe trained under. The notebook
          persists in the database — restart the app and it survives.</p>
      </div>
    </div>`;
}
window.setWriter = async (w) => {
  try { await post("/actions/agent-writer", { writer: w }); render(true); }
  catch (e) { toast(e.message, 4000); }
};
window.resetAgent = async () => {
  try { await post("/actions/agent-reset", {}); render(true); }
  catch (e) { toast(e.message, 4000); }
};
window.agentSend = async (ev) => {
  ev.preventDefault();
  const inp = $("#agent-in");
  const text = inp.value.trim();
  if (!text || agentBusy) return;
  agentBusy = true;
  inp.value = "";
  inp.placeholder = "thinking…";
  inp.disabled = true;
  try {
    await post("/actions/agent-chat", { message: text });
  } catch (e) {
    toast("Turn failed: " + e.message, 6000);
  }
  agentBusy = false;
  await render(true);
  const log = $("#agent-log");
  if (log) log.scrollTop = log.scrollHeight;
  $("#agent-in")?.focus();
};

/* ---------------- MODELS ---------------- */

const mdl = { sel: null, tab: "curl", testBusy: false, testReply: null };

async function models() {
  const [adapters, deployments, defaults] = await Promise.all([
    api("/adapters"), api("/deployments"), api("/defaults")]);
  const running = (deployments || []).filter((d) => d.status === "running");
  // the trainable base is whatever the adapters were trained on — not the
  // default eval endpoint model (often a hosted API model)
  const base = adapters?.[0]?.base_model || defaults.model || "Qwen/Qwen3-0.6B";
  if (mdl.sel == null) {
    const dep = running[0];
    mdl.sel = dep ? dep.adapter_id : (adapters[0]?.id ?? "base");
  }

  const card = (id, name, sub, live) => `
    <button class="mdl-card ${String(mdl.sel) === String(id) ? "sel" : ""}" onclick="mdlPick('${id}')">
      <div class="mdl-name">${esc(name)} ${live ? `<span class="chip acc" style="margin-left:6px">Live</span>` : ""}</div>
      <div class="mdl-sub">${esc(sub)}</div>
    </button>`;

  const list =
    card("base", base, "Base model — serve it as-is, no training", false) +
    (adapters || []).map((a) => card(a.id,
      `adapter #${a.id} · step ${a.step}`,
      `${a.env || "?"} · run ${a.train_run_id}${a.exists ? "" : " · file missing"}`,
      running.some((d) => d.adapter_id === a.id))).join("");

  /* right panel for the selection */
  let panel;
  const selAdapter = mdl.sel !== "base" ? (adapters || []).find((a) => String(a.id) === String(mdl.sel)) : null;
  const selDep = mdl.sel === "base"
    ? running.find((d) => !d.adapter_id)
    : running.find((d) => String(d.adapter_id) === String(mdl.sel));
  const title = mdl.sel === "base" ? base : `${base} + adapter #${mdl.sel}`;

  if (selDep) {
    const ep = selDep.endpoint || "";
    const modelName = selDep.served_name || base;
    const snips = {
      curl: `curl ${ep}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelName}",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'`,
      python: `from openai import OpenAI

client = OpenAI(base_url="${ep}", api_key="local")
reply = client.chat.completions.create(
    model="${modelName}",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)`,
      agents: `# any OpenAI-compatible agent or tool works — point it here:
export OPENAI_BASE_URL="${ep}"
export OPENAI_API_KEY="local"
export MODEL="${modelName}"`,
    };
    const tabBtn = (id, label) => `<button class="${mdl.tab === id ? "active" : ""}" onclick="mdlTab('${id}')">${label}</button>`;
    panel = `
      <h3 class="mono">${esc(title)}</h3>
      <div class="endpoint-row"><span>● LIVE</span><span>${esc(ep)}</span>
        <span class="dim">OpenAI-compatible · works with any client library</span></div>
      <div class="actions" style="display:flex; gap:10px">
        <button class="btn danger sm" onclick="mdlStop(${selDep.id})">Turn off</button>
      </div>
      <div class="snip-tabs">${tabBtn("curl", "curl")}${tabBtn("python", "Python")}${tabBtn("agents", "for agents")}</div>
      <pre class="snippet"><button class="copy-btn" onclick="copySnip(this)">copy</button>${esc(snips[mdl.tab])}</pre>
      <div class="section" style="margin:18px 0 0">
        <div class="sec-head"><h2>Try it</h2></div>
        <form class="term-bar" style="margin-top:8px" onsubmit="mdlTest(event, ${selDep.id})">
          <input id="mdl-test-in" placeholder="Ask the model something…" ${mdl.testBusy ? "disabled" : ""}>
          <button class="btn sm" ${mdl.testBusy ? "disabled" : ""}>${mdl.testBusy ? "Thinking…" : "Send"}</button>
        </form>
        ${mdl.testReply ? `<pre class="codeview" style="margin-top:10px; max-height:220px">${esc(mdl.testReply)}</pre>` : ""}
      </div>`;
  } else {
    const missing = selAdapter && !selAdapter.exists;
    panel = `
      <h3 class="mono">${esc(title)}</h3>
      <p class="sub" style="margin:8px 0 14px">${mdl.sel === "base"
        ? "Serve the base model as-is on your machine — you get a local OpenAI-compatible endpoint."
        : `A LoRA adapter from training run ${selAdapter?.train_run_id ?? "?"} on ${esc(selAdapter?.env || "?")}.
           Turning it on serves base + adapter together as one endpoint.`}</p>
      ${missing ? `<div class="empty"><div class="why">This adapter's files are missing on disk.</div></div>`
        : `<button class="btn" onclick="mdlServe('${mdl.sel}')">Turn on</button>
           <span class="helper" style="margin-left:10px">Loads on your machine (~20s). Nothing leaves your laptop.</span>`}`;
  }

  return `
    <div class="room-head">
      <h1>Models</h1>
      <p class="sub">Your base model and trained adapters. An adapter is a small file produced by
      training — it sits on top of the base model and changes how it behaves. Turn one on to get
      a local API endpoint with copy-paste code.</p>
    </div>
    <div class="models-grid">
      <aside>
        <div class="kicker" style="margin-bottom:8px">YOUR MODELS</div>
        ${list || `<div class="empty"><div class="why">No adapters yet — they're created when a training
          run finishes.</div><button class="btn" onclick="newTrainModal()">Start training</button></div>`}
      </aside>
      <div class="mdl-panel">${panel}</div>
    </div>`;
}
window.mdlPick = (id) => { mdl.sel = id; mdl.testReply = null; render(true); };
window.mdlTab = (t) => { mdl.tab = t; render(true); };
window.copySnip = (btn) => {
  const text = btn.parentElement.textContent.replace(/^copy/, "");
  navigator.clipboard?.writeText(text.trim());
  btn.textContent = "copied ✓";
  setTimeout(() => { btn.textContent = "copy"; }, 1500);
};
window.mdlServe = async (id) => {
  try {
    if (id === "base") { toast("Serving the bare base model isn't wired yet — serve an adapter", 4500); return; }
    toast("Starting the model on your machine… (~20s)");
    await post("/actions/deploy", { adapter_id: +id });
    toast("Model is live — endpoint and code ready");
    render(true);
  } catch (e) { toast("Could not serve: " + e.message, 5000); }
};
window.mdlStop = async (depId) => {
  try { await post("/actions/stop-deployment", { deployment_id: depId }); toast("Turned off"); render(true); }
  catch (e) { toast(e.message, 4000); }
};
window.mdlTest = async (ev, depId) => {
  ev.preventDefault();
  const inp = $("#mdl-test-in");
  const q = inp.value.trim();
  if (!q || mdl.testBusy) return;
  mdl.testBusy = true; mdl.testReply = null;
  render(true);
  try {
    const d = await post("/actions/chat", { deployment_id: depId, messages: [{ role: "user", content: q }] });
    mdl.testReply = d.content || "(no reply)";
  } catch (e) { mdl.testReply = "Error: " + e.message; }
  mdl.testBusy = false;
  render(true);
};

/* ---------------- CONSOLE (top-right >_ toggle, slide-up panel) ---------------- */

const con = { log: [], hist: [], histIdx: -1, draft: "", busy: false, open: false };
const CON_CHIPS = [
  { cmd: "env list", auto: true }, { cmd: "eval list", auto: true },
  { cmd: "eval show 1", auto: false }, { cmd: "training list", auto: true },
  { cmd: "training show 1", auto: false }, { cmd: "deployments list", auto: true },
  { cmd: "cloud status", auto: true }, { cmd: "instrument 19 21", auto: false },
  { cmd: "version", auto: true },
];

function renderConsolePanel() {
  const host = $("#console-inner");
  if (!host) return;
  const keep = $("#con-in")?.value ?? "";
  const chips = CON_CHIPS.map((c) =>
    `<button class="chip" onclick="conChip('${esc(c.cmd)}', ${c.auto})">${esc(c.cmd)}</button>`).join("");
  const entries = con.log.map((h) => `
    <div class="term-entry">
      <div class="term-cmd">$ nanolab ${esc(h.cmd)}</div>
      <pre class="term-res ${h.err ? "error" : ""}">${esc(h.out)}</pre>
    </div>`).join("");
  host.innerHTML = `
    <div class="chip-row">${chips}</div>
    <div class="term" id="con-out">${entries ||
      `<div class="term-empty">Output appears here. Try "env list".</div>`}
      ${con.busy ? `<div class="term-busy">running…</div>` : ""}</div>
    <form class="term-bar" onsubmit="conRun(event)">
      <span class="term-prompt">$ nanolab</span>
      <input id="con-in" autocomplete="off" spellcheck="false" placeholder="eval list"
        onkeydown="conKey(event)" ${con.busy ? "disabled" : ""}>
      <button class="btn" type="submit" ${con.busy ? "disabled" : ""}>${con.busy ? "Running…" : "Run"}</button>
      <button class="btn ghost" type="button" onclick="conClear()">Clear</button>
    </form>`;
  const inp = $("#con-in");
  if (inp) { inp.value = keep; }
  const out = $("#con-out");
  if (out) out.scrollTop = out.scrollHeight;
}

function toggleConsole(force) {
  con.open = force != null ? force : !con.open;
  document.body.classList.toggle("console-open", con.open);
  $("#term-btn")?.classList.toggle("on", con.open);
  if (con.open) { renderConsolePanel(); setTimeout(() => $("#con-in")?.focus(), 250); }
}
window.toggleConsole = toggleConsole;
document.addEventListener("keydown", (e) => {
  if (e.key === "`" && e.ctrlKey) { e.preventDefault(); toggleConsole(); }
});
window.conChip = (cmd, auto) => {
  const inp = $("#con-in");
  if (!inp) return;
  inp.value = cmd;
  inp.focus();
  if (auto) { conRun(new Event("submit")); return; }
  const m = cmd.match(/\d/);
  if (m) inp.setSelectionRange(m.index, cmd.length);
};
window.conKey = (ev) => {
  const inp = ev.target;
  if (ev.key === "ArrowUp") {
    ev.preventDefault();
    if (!con.hist.length) return;
    if (con.histIdx === -1) { con.draft = inp.value; con.histIdx = con.hist.length; }
    con.histIdx = Math.max(0, con.histIdx - 1);
    inp.value = con.hist[con.histIdx];
    setTimeout(() => inp.setSelectionRange(inp.value.length, inp.value.length), 0);
  } else if (ev.key === "ArrowDown") {
    ev.preventDefault();
    if (con.histIdx === -1) return;
    con.histIdx += 1;
    if (con.histIdx >= con.hist.length) { con.histIdx = -1; inp.value = con.draft; }
    else inp.value = con.hist[con.histIdx];
  }
};
window.conClear = () => { con.log = []; renderConsolePanel(); };
window.conRun = async (ev) => {
  ev.preventDefault?.();
  const inp = $("#con-in");
  const cmd = inp?.value.trim();
  if (!cmd || con.busy) return;
  con.hist.push(cmd);
  con.histIdx = -1;
  if (inp) inp.value = "";
  con.busy = true;
  renderConsolePanel();
  try {
    const { job } = await post("/actions/cli", { command: cmd });
    let done = null;
    for (let i = 0; i < 240 && !done; i++) {
      await new Promise((r) => setTimeout(r, 600));
      const j = await api("/jobs");
      const all = Array.isArray(j) ? j : [...(j.running || []), ...(j.recent || [])];
      done = all.find((x) => x.id === job.id && x.status !== "running");
    }
    con.log.push({ cmd, out: done?.output || done?.error || "(no output)", err: done?.status === "error" });
  } catch (e) {
    con.log.push({ cmd, out: e.message, err: true });
  }
  con.busy = false;
  renderConsolePanel();
  $("#con-in")?.focus();
};

/* ---------------- FINDINGS (the paper, dark) ---------------- */

async function findings() {
  // the recovered S2 curve (kernel log; steps 5–8 were not captured)
  const s2 = [[0,0.545],[1,0.375],[2,0.527],[3,0.464],[4,0.491],[9,0.893],[10,0.848],
    [11,0.821],[12,0.946],[13,1.0],[14,0.92],[15,0.973],[16,0.875],[17,0.982],[18,1.0],
    [19,1.0],[20,1.0],[21,1.0],[22,0.964],[23,0.982],[24,0.973],[25,1.0],[26,1.0],
    [27,0.991],[28,1.0],[29,1.0],[30,1.0]].map(([x, y]) => ({ x, y }));

  const transfer = `
    <table class="sheet">
      <tr><th>DRIFT RUNG</th><th class="right">PROMPTED</th><th class="right">TRAINED</th><th class="right">GAP</th></tr>
      <tr><td>Training distribution</td><td class="mono right neg">0.548${foot(22)}</td><td class="mono right pos">1.000${foot(24)}</td><td class="mono right">+0.452</td></tr>
      <tr><td>Hints removed</td><td class="mono right neg">0.536${foot(28)}</td><td class="mono right pos">1.000${foot(25)}</td><td class="mono right">+0.464</td></tr>
      <tr><td>5 distractors (trained on 3)</td><td class="mono right neg">0.518${foot(29)}</td><td class="mono right pos">1.000${foot(26)}</td><td class="mono right">+0.482</td></tr>
      <tr><td>12 tasks (trained on 8)</td><td class="mono right neg">0.443${foot(30)}</td><td class="mono right pos">1.000${foot(27)}</td><td class="mono right">+0.557</td></tr>
    </table>`;

  return `
    <div style="text-align:center; padding:26px 0 6px">
      <div class="kicker">THE LAB'S FINDINGS · EVERY NUMBER OPENS ITS RAW DATA</div>
      <h1 class="display">What this lab proved</h1>
      <p class="lede" style="margin:14px auto 0; max-width:620px">
        A self-hosted laboratory that trains AI models and proves it worked —
        its flagship experiment taught a tiny model the skill of choosing what to remember.</p>
    </div>
    <hr class="page-rule">

    <div class="section">
      <div class="sec-head"><span class="sec-no">§1</span><h2>The question</h2></div>
      <p class="body-text">An agent with a small context must decide, after every exchange,
      which facts survive into a 400-character notebook and which are discarded. Prompting
      alone gets this partly right. We asked whether the <i>skill itself</i> — not the facts —
      can be trained into a 0.6B model with reinforcement learning, and whether the answer
      can be proven with held-out evidence rather than vibes.</p>
    </div>

    <div class="section">
      <div class="sec-head"><span class="sec-no">§2</span><h2>The baseline — and an honest failure first</h2></div>
      <p class="body-text">The first version of the task was <i>too easy</i>: an untrained
      scribe already scored 0.905${foot(16)} against the checker and 0.905${foot(17)} against a real
      reader — matching a frontier model's 0.857${foot(5)}. Pure transcription. The trainability
      gate refused to train, correctly. So the task was rebuilt to demand judgment: each
      record buries the needed figure among one-off distractors, under a binding notebook
      cap. The untrained baseline fell to <b class="num">0.548</b>${foot(22)} — inside the window
      where a reward signal exists.</p>
      ${calibSvg(0.548)}
      ${figWrap("", "FIG. 1", "Calibration. Baseline 0.548 sits inside the 10–80% trainable window.", "EVAL #22")}
    </div>

    <div class="section">
      <div class="sec-head"><span class="sec-no">§3</span><h2>Training</h2></div>
      <p class="body-text">Forty GRPO steps were configured on a free Kaggle T4; the kernel
      hit the 12-hour wall near step 35, and its per-decade checkpoints were recovered from
      the committed working directory. Pre-flight reward <span class="num">0.411</span>; the
      curve saturates near <span class="num">1.0</span> by step 13 and holds.</p>
      ${figWrap(curveSvg(s2, { ticks: [9, 19, 29] }),
        "FIG. 2", "Reward vs step, recovered kernel log (steps 5–8 were not captured). Tick marks are saved checkpoints.", "RUN kaggle · ckpt 9/19/29")}
    </div>

    <div class="section">
      <div class="sec-head"><span class="sec-no">§4</span><h2>The verdict</h2></div>
      <p class="body-text">Held-out streams the model never saw in training. The trained
      scribe kept every needed figure and dropped every junk line — twelve times out of
      twelve. Its final notebook: 189 characters, zero distractors. The untrained model's:
      649 characters, eleven distractors, needed figures truncated away.</p>
      ${verdictBlock("The skill of choosing what to remember was trained, not prompted.",
        "0.548", "1.000", "12/12 held-out streams · final checkpoint",
        { evalFrom: 22, evalTo: 24 })}
    </div>

    <div class="section">
      <div class="sec-head"><span class="sec-no">§5</span><h2>Transfer under drift</h2></div>
      <p class="body-text">Does the trained skill survive away from its training
      distribution? Three rungs it never trained on. The prompted model degrades with
      distance; the trained one holds — and the gap <i>widens</i>. The hints-removed rung is
      the decisive ablation: it filters junk by content, not by copying training labels.</p>
      ${transfer}
      <p class="aside-note">Scope, stated honestly: drift within the same task family;
      checker Player; n=8 per rung. Three rungs is a short ladder — a different task
      family entirely is the untested fourth rung.</p>
    </div>

    <div class="section">
      <div class="sec-head"><span class="sec-no">§6</span><h2>What actually carried the skill</h2></div>
      <p class="body-text">A four-way ablation on the same streams: strip the model to
      base, then add back its trained weights, its accumulated notes, or both.</p>
      <table class="sheet">
        <tr><th>CONDITION</th><th class="right">REWARD</th></tr>
        <tr><td class="mono">base</td><td class="mono right neg">0.000${foot(19)}</td></tr>
        <tr><td class="mono">+ context (notes)</td><td class="mono right pos">0.393${foot(19)}</td></tr>
        <tr><td class="mono">+ weights</td><td class="mono right neg">0.000${foot(21)}</td></tr>
        <tr><td class="mono">+ both</td><td class="mono right pos">0.429${foot(21)}</td></tr>
      </table>
      ${figWrap("", "TABLE 1", "KNOWLEDGE-DOMINANT: notes beat weight-training. The weights alone carry nothing; the notebook carries almost everything.", "EVALS #19 · #21")}
    </div>

    <div class="section">
      <div class="sec-head"><span class="sec-no">§7</span><h2>The other loop: weights</h2></div>
      <p class="body-text">The same lab also closes the classic loop. Qwen3-0.6B on
      grade-school math, GRPO+LoRA on a free T4: base <span class="num">0.422</span> →
      trained <span class="num">0.562</span> on 64 held-out questions, final checkpoint —
      measured in the training kernel's own exam. The lab's instruments also caught the
      footnote: the gain is strongest inside the training-time token budget
      (station re-reads at other settings: 0.500${foot(11)} · 0.562${foot(14)} · 0.625${foot(15)}).
      Both readings live in the db; neither is hidden.</p>
    </div>

    <div style="text-align:center; margin:44px 0 10px">
      <button class="btn" onclick="location.hash='#/home'">Open the lab →</button>
    </div>
    <p class="aside-note" style="text-align:center">Every ° above opens the raw rollout rows behind that number.</p>`;
}

/* ---------------- router ---------------- */

const routes = [
  [/^#?\/?$/, home, "home"],
  [/^#\/home$/, home, "home"],
  [/^#\/tasks$/, tasks, "tasks"],
  [/^#\/task\/(.+)$/, taskDetail, "tasks"],
  [/^#\/train$/, train, "train"],
  [/^#\/train\/(\d+)$/, trainDetail, "train"],
  [/^#\/evals$/, evals, "evals"],
  [/^#\/evals\/(\d+)$/, evalDetail, "evals"],
  [/^#\/memory$/, memory, "memory"],
  [/^#\/models$/, models, "models"],
  [/^#\/findings$/, findings, "findings"],
  /* #/console now opens the panel instead of a page */
  [/^#\/console$/, () => { toggleConsole(true); location.hash = "#/home"; return ""; }, "home"],
  /* legacy hashes from the previous UI keep working */
  [/^#\/bench$/, home, "home"],
  [/^#\/environments$/, tasks, "tasks"],
  [/^#\/env\/(.+)$/, taskDetail, "tasks"],
  [/^#\/training$/, train, "train"],
  [/^#\/training\/(\d+)$/, trainDetail, "train"],
  [/^#\/agent$/, memory, "memory"],
  [/^#\/artifacts$/, models, "models"],
  [/^#\/paper$/, findings, "findings"],
];

let lastHash = null, lastHtml = null;

async function render(force = false) {
  const hash = location.hash || "#/home";
  const changed = hash !== lastHash || force;
  if (hash !== lastHash) { taskView.tab = "about"; taskView.file = null; }
  for (const [re, fn, nav] of routes) {
    const m = hash.match(re);
    if (!m) continue;
    document.querySelectorAll("nav a").forEach((a) =>
      a.classList.toggle("active", a.dataset.route === nav));
    try {
      const html = await fn(...m.slice(1));
      if (!changed && html === lastHtml) return;   // no flicker on silent refresh
      lastHtml = html;
      const scrollY = changed && !force ? 0 : window.scrollY;
      page.innerHTML = `<div class="page${changed && !force ? " anim" : ""}${nav === "findings" ? " narrow" : ""}">${html}</div>`;
      window.scrollTo(0, scrollY);
      const log = $("#agent-log");
      if (log) log.scrollTop = log.scrollHeight;
      lastHash = hash;
    } catch (err) {
      page.innerHTML = `<div class="page"><div class="empty"><div class="why">Something went wrong loading this: ${esc(err.message)}</div>
        <button class="btn ghost" onclick="render(true)">Retry</button></div></div>`;
    }
    return;
  }
  page.innerHTML = `<div class="page"><div class="empty"><div class="why">Page not found.</div></div></div>`;
}

window.render = render;
window.addEventListener("hashchange", () => render());
refreshChrome().then(() => render());

/* silent refresh — never while typing, with the drawer open, or mid console/test run */
setInterval(() => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName);
  const drawerOpen = document.body.classList.contains("drawer-open");
  if (!document.hidden && !typing && !drawerOpen && !con.busy && !mdl.testBusy && !agentBusy) {
    refreshChrome(); render();
  }
}, 5000);

/* self-update: long-lived tabs reload when the server ships new UI code */
let uiVersion = null;
setInterval(async () => {
  try {
    const v = (await api("/version")).ui;
    if (uiVersion === null) { uiVersion = v; return; }
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName);
    if (v !== uiVersion && !typing && !document.hidden) location.reload();
  } catch {}
}, 30000);
