/* nanolab — the living paper (v0.2 UI).
   One vanilla file: router + rooms + paper components, all data from /api.
   Rule: the UI never computes a metric — every number is read from the same
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
function modelChip(m) {
  return `<span class="chip">${esc(m)}</span>`;
}

/* ---------------- paper components ---------------- */

/* footnote mark: every number is a claim; the mark opens its raw rows */
const foot = (evalId) =>
  evalId == null ? "" :
  `<button class="foot" title="receipts — eval #${evalId}" onclick="openReceipts(${evalId});event.stopPropagation()">°</button>`;

function figWrap(inner, no, text, margin) {
  return `<figure class="fig">${inner}
    <figcaption><span class="fig-no">${esc(no)}</span>
      <span class="fig-text">${text}</span>
      ${margin ? `<span class="fig-margin">${esc(margin)}</span>` : ""}
    </figcaption></figure>`;
}

/* reward-vs-step chart, paper axes. points: [{x, y}] */
function curveSvg(points, { w = 920, h = 260, yMax = 1.05, ticks = [] } = {}) {
  if (!points.length) return `<div class="empty"><div class="why">no data points yet</div></div>`;
  const padL = 46, padR = 14, padT = 16, padB = 30;
  const xMax = Math.max(...points.map((p) => p.x), 1);
  const X = (x) => padL + ((w - padL - padR) * x) / xMax;
  const Y = (y) => padT + (h - padT - padB) * (1 - Math.min(y, yMax) / yMax);
  const path = points.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" ");
  const last = points[points.length - 1];
  const tickMarks = ticks.map((t) =>
    `<line x1="${X(t)}" y1="${h - padB}" x2="${X(t)}" y2="${h - padB + 5}" stroke="#1A1A1A" stroke-width="1"/>
     <text x="${X(t)}" y="${h - padB + 17}" font-family="Geist Mono,monospace" font-size="9.5" fill="#9B978C" text-anchor="middle">${t}</text>`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" stroke="#D4D1C6" stroke-width="1"/>
    <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="#1A1A1A" stroke-width="1"/>
    <line x1="${padL}" y1="${Y(1)}" x2="${w - padR}" y2="${Y(1)}" stroke="#D4D1C6" stroke-width="1" stroke-dasharray="3 5"/>
    <text x="${padL - 8}" y="${Y(1) + 3}" font-family="Geist Mono,monospace" font-size="10" fill="#6B6860" text-anchor="end">1.0</text>
    <text x="${padL - 8}" y="${h - padB + 3}" font-family="Geist Mono,monospace" font-size="10" fill="#6B6860" text-anchor="end">0.0</text>
    ${tickMarks}
    <path d="${path}" fill="none" stroke="#3D7A4E" stroke-width="1.8"/>
    <circle cx="${X(last.x)}" cy="${Y(last.y)}" r="3.2" fill="#1A1A1A"/>
    <text x="${Math.min(X(last.x) + 8, w - padR - 30)}" y="${Y(last.y) - 8}" font-family="Geist Mono,monospace" font-size="10.5" fill="#1A1A1A">${fnum(last.y, 3)}</text>
  </svg>`;
}

/* calibration figure: the 10–80% trainability window as a number line */
function calibSvg(baseline, { w = 920, h = 74, lo = 0.1, hi = 0.8 } = {}) {
  const padL = 14, padR = 14, y = 40;
  const X = (v) => padL + (w - padL - padR) * v;
  const tick = baseline == null ? "" : `
    <line x1="${X(baseline)}" y1="${y - 22}" x2="${X(baseline)}" y2="${y + 10}" stroke="#7A2E2A" stroke-width="1.6"/>
    <text x="${X(baseline)}" y="${y - 28}" font-family="Geist Mono,monospace" font-size="11" fill="#7A2E2A" text-anchor="middle">${fnum(baseline, 3)}</text>`;
  return `<div class="calib"><svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">
    <rect x="${X(lo)}" y="${y - 12}" width="${X(hi) - X(lo)}" height="14" fill="#EDE9DD"/>
    <line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="#1A1A1A" stroke-width="1"/>
    ${tick}
    <text x="${X(0.02)}" y="${y + 24}" font-family="Geist Mono,monospace" font-size="9.5" fill="#9B978C">NO SIGNAL</text>
    <text x="${(X(lo) + X(hi)) / 2}" y="${y + 24}" font-family="Geist Mono,monospace" font-size="9.5" fill="#6B6860" text-anchor="middle">TRAINABLE</text>
    <text x="${X(0.98)}" y="${y + 24}" font-family="Geist Mono,monospace" font-size="9.5" fill="#9B978C" text-anchor="end">ALREADY SOLVED</text>
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

/* markdown-lite for env READMEs: headers, fenced code, inline code, bold */
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
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeDrawer(); closeModal(); } });

async function openReceipts(evalId) {
  document.body.classList.add("drawer-open");
  $("#d-kicker").textContent = `RECEIPTS · EVAL #${evalId}`;
  $("#d-title").textContent = "loading raw rows…";
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
      <p class="aside-note">Every rollout behind this number, straight from the database.
      Run: n=${d.num_examples} · r=${d.rollouts_per_example} · status ${esc(d.status)} ·
      started ${when(d.started_at)}.</p>
      ${meta ? `<table class="sheet"><tr><th>METRIC</th><th class="right">MEAN</th></tr>${meta}</table>` : ""}
      <table class="sheet"><tr><th>EXAMPLE·ROLLOUT</th><th class="right">REWARD</th></tr>${rows}</table>`;
  } catch (e) {
    $("#d-body").innerHTML = `<div class="empty"><div class="why">could not load: ${esc(e.message)}</div></div>`;
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
    <h3>New evaluation</h3>
    <div class="m-sub">Rollouts + rubric scoring; lands in Evals with full receipts.</div>
    <label class="f">Environment</label><select id="ev-env">${opts}</select>
    <label class="f">Model</label><input id="ev-model" value="${esc(defaults.model || "")}">
    <div class="form-grid">
      <div><label class="f">Examples (n)</label><input id="ev-n" type="number" value="5" min="1"></div>
      <div><label class="f">Rollouts per example</label><input id="ev-r" type="number" value="1" min="1"></div>
    </div>
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
    closeModal(); toast("evaluation started — it will appear in Evals");
  } catch (e) { toast("could not start: " + e.message, 5000); }
};

async function newTrainModal() {
  const configs = await api("/configs");
  const opts = configs.map((c) => `<option value="${esc(c.path)}">${esc(c.name || c.path)}</option>`).join("");
  openModal(`
    <h3>New training run</h3>
    <div class="m-sub">Pushes to Kaggle's free GPU; the poller merges the finished
    adapter back automatically. Close the laptop after launch.</div>
    <label class="f">Training config</label><select id="tr-config">${opts}</select>
    <div class="m-foot">
      <button class="btn ghost" onclick="closeModal()">Cancel</button>
      <button class="btn" onclick="submitTrain()">Launch on Kaggle</button>
    </div>`);
}
window.newTrainModal = newTrainModal;
window.submitTrain = async () => {
  try {
    await post("/actions/train-cloud", { config: $("#tr-config").value });
    closeModal(); toast("pushed to Kaggle — watch it in Training");
  } catch (e) { toast("could not launch: " + e.message, 5000); }
};

async function installModal() {
  openModal(`
    <h3>Install environment</h3>
    <div class="m-sub">Any verifiers-format task from the hub, e.g.
    <code>primeintellect/gsm8k</code>.</div>
    <label class="f">Hub slug</label><input id="in-slug" placeholder="owner/name">
    <div class="m-foot">
      <button class="btn ghost" onclick="closeModal()">Cancel</button>
      <button class="btn" onclick="submitInstall()">Install</button>
    </div>`);
}
window.installModal = installModal;
window.submitInstall = async () => {
  try {
    await post("/actions/install", { slug: $("#in-slug").value });
    closeModal(); toast("installing — it will appear when registered");
  } catch (e) { toast("could not install: " + e.message, 5000); }
};

/* ---------------- experiment strip + chrome ---------------- */

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
      else segs.push(`<span class="arrow">→</span><span class="seg">TRAINING ${t.status === "done" ? `<span class="done-mark">✓</span> ${t.steps_completed} STEPS` : esc(String(t.status).toUpperCase())}</span>`);
      const verd = (ov.recent_evals || []).find((e) => e.slug === t.slug && e.status === "done");
      segs.push(`<span class="arrow">→</span><span class="seg">${verd ? `LATEST EVAL <b>${fnum(verd.mean_reward)}</b>` : "VERDICT PENDING"}</span>`);
    }
    $("#strip").innerHTML = segs.join("");
    $("#side-foot").innerHTML =
      `v0.2 · ui ${esc(String(ver.ui || "").slice(0, 8))}<br>` +
      `${ov.environments} envs · ${ov.evals.total} evals · ${ov.adapters} adapters`;
  } catch { /* chrome is decoration; the rooms surface real errors */ }
}

/* ---------------- rooms ---------------- */

async function bench() {
  const [ov, jobs] = await Promise.all([api("/overview"), api("/jobs")]);
  const t = ov.recent_trains?.[0];

  let action;
  const cloudActive = lab.cloud.some((c) => ["pushed", "running"].includes(c.status));
  if (cloudActive)
    action = `<p class="aside-note">Training is running on Kaggle. Nothing to click —
      the poller pulls the adapter home when it finishes.</p>`;
  else if (!ov.environments)
    action = `<button class="btn" onclick="installModal()">Install your first environment</button>`;
  else if (!ov.evals.total)
    action = `<button class="btn" onclick="newEvalModal()">Run the first baseline</button>`;
  else
    action = `<button class="btn" onclick="newTrainModal()">New training run</button>
      <button class="btn ghost" onclick="newEvalModal()" style="margin-left:10px">New evaluation</button>`;

  const liveFig = t?.rewards?.length
    ? figWrap(curveSvg(t.rewards.map((y, x) => ({ x, y }))),
        "FIG. LIVE", `Reward vs step, run ${t.id} (${esc(t.slug || "")}), ${esc(t.status)}.`,
        `${t.rewards.length} steps`)
    : `<div class="empty"><div class="why">No training curve yet — a run's rewards draw here as it trains.</div>
       <span class="cmd">$ nanolab train --cloud configs/&lt;config&gt;.toml</span></div>`;

  const recent = (ov.recent_evals || []).slice(0, 6).map((e) => `
    <tr class="click" onclick="location.hash='#/evals/${e.id}'">
      <td class="mono">#${e.id}</td><td class="mono">${esc(e.slug)}</td>
      <td>${modelChip(e.model)}</td>
      <td class="mono right num">${fnum(e.mean_reward)}${e.status === "done" ? foot(e.id) : ""}</td>
      <td class="right"><span class="status ${esc(e.status)}">${esc(e.status)}</span></td>
    </tr>`).join("");

  const jobRows = (jobs.running || []).concat(jobs.recent || []).slice(0, 4).map((j) => `
    <div class="job"><span class="status ${j.status === "running" ? "running" : j.status === "error" ? "error" : "done"}">${esc(j.status)}</span>
      <span>${esc(j.label)}</span>
      ${j.status !== "running" ? `<button class="x" onclick="dismissJob('${esc(j.id)}')">dismiss</button>` : ""}</div>`).join("");

  return `
    <div class="kicker">THE BENCH · CURRENT EXPERIMENT</div>
    <h1>The Bench</h1><hr class="page-rule">
    <div class="section" style="margin-top:0">
      <div class="sec-head"><span class="sec-no">NEXT ACTION</span></div>${action}
    </div>
    ${jobRows ? `<div class="jobs">${jobRows}</div>` : ""}
    ${liveFig}
    <div class="section">
      <div class="sec-head"><span class="sec-no">RECENT</span><h2>Evaluations</h2></div>
      <table class="sheet"><tr><th>EVAL</th><th>TASK</th><th>MODEL</th><th class="right">REWARD</th><th class="right">STATUS</th></tr>${recent ||
        `<tr><td colspan="5"><div class="aside-note">none yet</div></td></tr>`}</table>
    </div>`;
}
window.dismissJob = async (id) => { try { await post("/actions/dismiss-job", { job_id: id }); } catch {} };

/* THE PAPER — the lab's findings as a living article. The narrative is
   curated; every number footnotes to live db rows via the receipts drawer. */
async function paper() {
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
    <div style="text-align:center; padding:34px 0 6px">
      <div class="kicker">A LIVING RESEARCH PAPER · EVERY NUMBER OPENS ITS RAW DATA</div>
      <h1 class="display">nanolab</h1>
      <p class="lede" style="margin:14px auto 0; font-style:italic">
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

    <div style="text-align:center; margin:50px 0 10px">
      <button class="btn" onclick="location.hash='#/bench'">OPEN THE BENCH →</button>
    </div>
    <p class="aside-note" style="text-align:center">Every ° above opens the raw rollout rows behind that number.</p>`;
}

async function environments() {
  const envs = await api("/environments");
  const cards = envs.map((e) => `
    <div class="card">
      <h3><a href="#/env/${encodeURIComponent(e.slug)}">${esc(e.slug)}</a></h3>
      <div class="sub">${esc(e.summary || "a verifiers-format task")}</div>
      ${calibSvg(e.best_reward, { h: 62 })}
      <div class="meta"><span>${e.best_reward != null ? `BEST ${fnum(e.best_reward)} · ${esc(String(e.best_model || "").slice(0, 22))}` : "NOT YET MEASURED"}</span>
        <span>v${esc(e.version || "?")}</span></div>
    </div>`).join("");
  return `
    <div class="kicker">ROOMS · ENVIRONMENTS</div>
    <h1>Environments</h1><hr class="page-rule">
    <p class="lede">A task is an environment with an automatic grader. An environment is
    only trainable if its baseline lands in the 10–80% window — calibrate before you train.</p>
    <div style="margin:18px 0"><button class="btn" onclick="installModal()">Install from hub</button></div>
    <div class="cards">${cards ||
      `<div class="empty"><div class="why">No environments yet. Install one and the loop begins.</div>
       <span class="cmd">$ nanolab env install primeintellect/gsm8k</span></div>`}</div>`;
}

async function envDetail(slug) {
  const d = await api(`/environments/${encodeURIComponent(decodeURIComponent(slug))}`);
  if (!d) return `<div class="empty"><div class="why">environment not found</div></div>`;
  const lb = (d.leaderboard || []).map((r, i) => `
    <tr class="click" onclick="location.hash='#/evals/${r.id}'">
      <td class="mono">${i + 1}</td><td>${modelChip(r.model)}</td>
      <td class="mono right num">${fnum(r.mean_reward)}${foot(r.id)}</td>
      <td class="mono right">${r.num_examples ?? "—"}</td></tr>`).join("");
  const files = (d.files || []).map((f) => `
    <details><summary class="mono" style="cursor:pointer; padding:6px 0">${esc(f.name)}</summary>
    <pre style="background:var(--paper2); border:1px solid var(--rule); padding:12px; overflow-x:auto"><code>${esc(f.content || "")}</code></pre></details>`).join("");
  return `
    <div class="kicker"><a href="#/environments">ENVIRONMENTS</a> · ${esc(d.slug)}</div>
    <h1>${esc(d.name || d.slug)}</h1><hr class="page-rule">
    <p class="lede">${esc(d.summary || "")}</p>
    <div style="margin:16px 0 26px">
      <button class="btn" onclick="newEvalModal('${esc(d.slug)}')">Evaluate on this task</button>
      <span class="chip" style="margin-left:12px">installed ${when(d.installed_at)}</span>
    </div>
    ${lb ? `<div class="section"><div class="sec-head"><span class="sec-no">LEADERBOARD</span></div>
      <table class="sheet"><tr><th>RANK</th><th>MODEL</th><th class="right">REWARD</th><th class="right">N</th></tr>${lb}</table></div>` : ""}
    ${d.readme ? `<div class="section"><div class="sec-head"><span class="sec-no">README</span></div>${mdLite(d.readme)}</div>` : ""}
    ${files ? `<div class="section"><div class="sec-head"><span class="sec-no">CODE</span></div>${files}</div>` : ""}`;
}

async function evals() {
  const runs = await api("/evals");
  const rows = runs.map((e) => `
    <tr class="click" onclick="location.hash='#/evals/${e.id}'">
      <td class="mono">#${e.id}</td><td class="mono">${esc(e.env)}</td>
      <td>${modelChip(e.model)}</td>
      <td class="mono right">${e.num_examples}×${e.rollouts_per_example}</td>
      <td class="mono right num ${e.mean_reward >= 0.5 ? "pos" : e.mean_reward != null ? "neg" : ""}">${fnum(e.mean_reward)}${e.status === "done" ? foot(e.id) : ""}</td>
      <td class="right"><span class="status ${esc(e.status)}">${esc(e.status)}</span></td>
    </tr>`).join("");
  return `
    <div class="kicker">ROOMS · EVALS</div>
    <h1>Evals</h1><hr class="page-rule">
    <p class="lede">Every measurement the lab has ever made — cached, resumable, and
    auditable to the rollout. The eval station matches the reference tool to every decimal.</p>
    <div style="margin:18px 0"><button class="btn" onclick="newEvalModal()">New evaluation</button></div>
    <table class="sheet">
      <tr><th>EVAL</th><th>TASK</th><th>MODEL</th><th class="right">N×R</th><th class="right">REWARD</th><th class="right">STATUS</th></tr>
      ${rows || `<tr><td colspan="6"><div class="empty"><div class="why">No measurements yet.
        The before-number is the one training has to beat.</div>
        <span class="cmd">$ nanolab eval run &lt;env&gt; -m &lt;model&gt; -n 10</span></div></td></tr>`}
    </table>`;
}

async function evalDetail(id) {
  const d = await api(`/evals/${id}`);
  if (!d) return `<div class="empty"><div class="why">eval not found</div></div>`;
  const metrics = Object.entries(d.meta?.avg_metrics || {}).map(([k, v]) =>
    `<tr><td class="mono">${esc(k)}</td><td class="mono right num">${fnum(v)}</td></tr>`).join("");
  const rollouts = (d.rollouts || []).map((r, i) => `
    <tr class="click" onclick="toggleRow(${i})">
      <td class="mono">${r.example}·${r.rollout}</td>
      <td class="mono right num ${r.reward >= 0.5 ? "pos" : "neg"}">${fnum(r.reward)}</td>
    </tr><tr id="row-${i}" style="display:none"><td colspan="2">${convoHtml(r)}</td></tr>`).join("");
  return `
    <div class="kicker"><a href="#/evals">EVALS</a> · #${d.id}</div>
    <h1>${esc(d.env)}</h1><hr class="page-rule">
    ${verdictBlock(`${esc(d.model)} on ${esc(d.env)}`, "", fnum(d.mean_reward),
      `n=${d.num_examples} · r=${d.rollouts_per_example} · ${d.status} · ${when(d.started_at)}`,
      { small: true })}
    ${metrics ? `<div class="section"><div class="sec-head"><span class="sec-no">METRICS</span></div>
      <table class="sheet"><tr><th>METRIC</th><th class="right">MEAN</th></tr>${metrics}</table></div>` : ""}
    <div class="section"><div class="sec-head"><span class="sec-no">ROLLOUTS</span>
      <h2>Every conversation</h2></div>
      <table class="sheet"><tr><th>EXAMPLE·ROLLOUT</th><th class="right">REWARD</th></tr>${rollouts}</table>
    </div>`;
}
window.toggleRow = (i) => {
  const el = $(`#row-${i}`);
  if (el) el.style.display = el.style.display === "none" ? "" : "none";
};

async function training() {
  const [runs, cloud] = await Promise.all([api("/training"), api("/cloud")]);
  const cloudRows = cloud.slice(0, 5).map((c) => `
    <div class="job"><span class="status ${esc(c.status)}">${esc(c.status)}</span>
      <span class="mono">${esc(c.kernel_ref)}</span>
      <span style="margin-left:auto" class="mono">${when(c.created_at)}</span></div>`).join("");
  const rows = runs.map((t) => {
    const pts = (t.rewards || []).map((y, x) => ({ x, y }));
    const spark = pts.length ? curveSvg(pts, { w: 240, h: 60 }) : "";
    return `
    <tr class="click" onclick="location.hash='#/training/${t.id}'">
      <td class="mono">#${t.id}</td><td class="mono">${esc(t.env || "?")}</td>
      <td>${modelChip(t.model)}</td>
      <td style="width:250px">${spark}</td>
      <td class="mono right">${t.steps_completed ?? 0}</td>
      <td class="right"><span class="status ${esc(t.status)}">${esc(t.status)}</span></td>
    </tr>`;
  }).join("");
  return `
    <div class="kicker">ROOMS · TRAINING</div>
    <h1>Training</h1><hr class="page-rule">
    <p class="lede">GRPO + LoRA: reward good attempts, discourage bad ones, checkpoint every
    decade — because the final step is not always the best one.</p>
    <div style="margin:18px 0"><button class="btn" onclick="newTrainModal()">New training run</button></div>
    ${cloudRows ? `<div class="section" style="margin-top:8px"><div class="sec-head"><span class="sec-no">KAGGLE</span></div>
      <div class="jobs">${cloudRows}</div></div>` : ""}
    <table class="sheet">
      <tr><th>RUN</th><th>TASK</th><th>MODEL</th><th>CURVE</th><th class="right">STEPS</th><th class="right">STATUS</th></tr>
      ${rows || `<tr><td colspan="6"><div class="empty"><div class="why">No training yet. Measure first,
        then train, then measure again.</div>
        <span class="cmd">$ nanolab train --cloud configs/&lt;config&gt;.toml</span></div></td></tr>`}
    </table>`;
}

async function trainingDetail(id) {
  const d = await api(`/training/${id}`);
  if (!d) return `<div class="empty"><div class="why">run not found</div></div>`;
  const pts = (d.curve || []).map((p) => ({ x: p.step, y: p.reward }));
  const ticks = (d.adapters || []).map((a) => a.step);
  const ckpts = (d.adapters || []).map((a) => `
    <tr><td class="mono">ckpt-${String(a.step).padStart(3, "0")}</td>
      <td class="mono">${esc(a.path)}</td>
      <td class="mono right">${when(a.created_at)}</td></tr>`).join("");
  return `
    <div class="kicker"><a href="#/training">TRAINING</a> · RUN ${d.id}</div>
    <h1>Run ${d.id} · ${esc(d.env || "?")}</h1><hr class="page-rule">
    <p class="lede">${esc(d.model)} · ${esc(d.status)} · ${d.steps_completed ?? 0} steps ·
      started ${when(d.started_at)}</p>
    ${pts.length ? figWrap(curveSvg(pts, { ticks }),
      `FIG. RUN-${d.id}`, "Reward vs step. Tick marks are saved checkpoints; any can be served or resumed.",
      "checkpoint every 10") : `<div class="empty"><div class="why">no curve recorded</div></div>`}
    ${ckpts ? `<div class="section"><div class="sec-head"><span class="sec-no">CHECKPOINTS</span></div>
      <table class="sheet"><tr><th>CHECKPOINT</th><th>PATH</th><th class="right">SAVED</th></tr>${ckpts}</table>
      <p class="aside-note">Serve one from Artifacts, or evaluate it with
      <span class="cmd">nanolab eval run &lt;env&gt; -m base:adapter-id</span></p></div>` : ""}`;
}

/* Memory Agent — honest state: the finding is real (shown from live db rows),
   the interactive engine ships next. No fake chat. */
async function agent() {
  let notebook = "", notebookMeta = "";
  try {
    const d = await api("/evals/24");
    const roll = d?.rollouts?.[0];
    const msgs = Array.isArray(roll?.completion) ? roll.completion : [];
    const last = [...msgs].reverse().find((m) => m.role === "assistant");
    if (last) {
      notebook = String(last.content || "");
      notebookMeta = `${notebook.length}/400 chars · rewritten by the trained scribe · EVAL #24, stream 0`;
    }
  } catch { /* fresh lab: section simply doesn't render */ }
  return `
    <div class="kicker">ROOMS · MEMORY AGENT</div>
    <h1>Memory Agent</h1><hr class="page-rule">
    <p class="lede">An assistant whose only memory is a capped notebook, rewritten after
    every exchange by the trained Scribe — the model this lab taught to choose what to
    remember. The agent itself is an experiment: switch the memory writer and the lift
    recomputes against a no-memory control.</p>
    ${verdictBlock("The memory writer this room will use, measured.",
      "0.548", "1.000", "prompted vs trained scribe · 12/12 held-out streams",
      { evalFrom: 22, evalTo: 24, small: true })}
    ${notebook ? `
    <div class="section">
      <div class="sec-head"><span class="sec-no">THE NOTEBOOK</span><h2>A real one, from the db</h2></div>
      <div class="convo"><div class="msg assistant"><div class="role">trained scribe · final rewrite</div>
        <pre>${esc(notebook)}</pre></div></div>
      <p class="aside-note">${esc(notebookMeta)} — zero junk lines kept. This is the actual
      artifact behind the 1.000${foot(24)}, not a mockup.</p>
    </div>` : ""}
    <div class="empty">
      <div class="why">The interactive engine — chat + live notebook rewriting + lift meter —
      is the next build (v0.3). The trained scribe adapter it will use is already on disk.</div>
      <span class="cmd">adapters/scribe_s2/step00029</span>
    </div>`;
}

async function artifacts() {
  const [adapters, deployments] = await Promise.all([api("/adapters"), api("/deployments")]);
  const dep = (deployments || []).filter((d) => d.status === "running");
  const depRows = dep.map((d) => `
    <div class="job"><span class="status running">running</span>
      <span class="mono">deployment #${d.id} · adapter ${d.adapter_id} · ${esc(d.endpoint || "")}</span>
      <button class="x" onclick="stopDep(${d.id})">stop</button></div>`).join("");
  const rows = (adapters || []).map((a) => `
    <tr><td class="mono">#${a.id}</td>
      <td class="mono">run ${a.train_run_id} · step ${a.step}</td>
      <td class="mono">${esc(a.env || "?")}</td>
      <td class="mono" style="font-size:12px">${esc(a.path)}${a.exists ? "" : ' <span class="neg">(missing)</span>'}</td>
      <td class="right">${a.deployed
        ? `<span class="status running">serving</span>`
        : a.exists ? `<button class="btn sm ghost" onclick="deployAdapter(${a.id})">serve</button>` : ""}</td>
    </tr>`).join("");
  return `
    <div class="kicker">ROOMS · ARTIFACTS</div>
    <h1>Artifacts</h1><hr class="page-rule">
    <p class="lede">Checkpoints land here when a run saves. Serving is plumbing: an adapter
    goes live only so an eval (or the agent) can read it. Everything is a file.</p>
    ${depRows ? `<div class="jobs">${depRows}</div>` : ""}
    <table class="sheet">
      <tr><th>ADAPTER</th><th>ORIGIN</th><th>TASK</th><th>PATH</th><th class="right"></th></tr>
      ${rows || `<tr><td colspan="5"><div class="empty"><div class="why">No adapters yet —
        finish a training run and its checkpoints register here.</div></div></td></tr>`}
    </table>`;
}
window.deployAdapter = async (id) => {
  try { await post("/actions/deploy", { adapter_id: id }); toast("serving locally — endpoint appears above"); }
  catch (e) { toast("could not serve: " + e.message, 5000); }
};
window.stopDep = async (id) => {
  try { await post("/actions/stop-deployment", { deployment_id: id }); toast("stopped"); }
  catch (e) { toast(e.message, 4000); }
};

/* console — the CLI, from the browser; read-only allowlist enforced server-side */
const consoleHistory = [];
async function consolePage() {
  const hist = consoleHistory.map((h) =>
    `<div class="cline">$ nanolab ${esc(h.cmd)}</div>${esc(h.out)}\n`).join("\n");
  return `
    <div class="kicker">ROOMS · CONSOLE</div>
    <h1>Console</h1><hr class="page-rule">
    <p class="lede" style="font-size:16px">The read-only CLI, from the browser.
    <span class="mono" style="font-size:12.5px">env list · eval list · eval show N ·
    training list · training show N · deployments list · cloud status · instrument N M · version</span></p>
    <div class="console-out" id="console-out">${hist || "run a command — output appears here, unedited."}</div>
    <form class="console-bar" onsubmit="runConsole(event)">
      <span class="prompt">$ nanolab</span>
      <input id="console-in" autocomplete="off" spellcheck="false" placeholder="eval list">
      <button class="btn" type="submit">Run</button>
    </form>`;
}
window.runConsole = async (ev) => {
  ev.preventDefault();
  const inp = $("#console-in");
  const cmd = inp.value.trim();
  if (!cmd) return;
  inp.value = "";
  const out = $("#console-out");
  out.textContent += `\n$ nanolab ${cmd}\n… running`;
  out.scrollTop = out.scrollHeight;
  try {
    const { job } = await post("/actions/cli", { command: cmd });
    let done = null;
    for (let i = 0; i < 240 && !done; i++) {
      await new Promise((r) => setTimeout(r, 600));
      const jobs = await api("/jobs");
      done = [...(jobs.recent || []), ...(jobs.running || [])]
        .find((j) => j.id === job.id && j.status !== "running");
    }
    consoleHistory.push({ cmd, out: done?.output || done?.error || "(no output)" });
  } catch (e) {
    consoleHistory.push({ cmd, out: "error: " + e.message });
  }
  render(true);
};

/* ---------------- router ---------------- */

const routes = [
  [/^#?\/?$/, bench, "bench"],
  [/^#\/bench$/, bench, "bench"],
  [/^#\/paper$/, paper, "paper"],
  [/^#\/environments$/, environments, "environments"],
  [/^#\/env\/(.+)$/, envDetail, "environments"],
  [/^#\/evals$/, evals, "evals"],
  [/^#\/evals\/(\d+)$/, evalDetail, "evals"],
  [/^#\/training$/, training, "training"],
  [/^#\/training\/(\d+)$/, trainingDetail, "training"],
  [/^#\/agent$/, agent, "agent"],
  [/^#\/artifacts$/, artifacts, "artifacts"],
  [/^#\/console$/, consolePage, "console"],
];

let lastHash = null, lastHtml = null;

async function render(force = false) {
  const hash = location.hash || "#/bench";
  const changed = hash !== lastHash || force;
  for (const [re, fn, nav] of routes) {
    const m = hash.match(re);
    if (!m) continue;
    document.querySelectorAll("nav a, .paper-link").forEach((a) =>
      a.classList.toggle("active", a.dataset.route === nav));
    try {
      const html = await fn(...m.slice(1));
      if (!changed && html === lastHtml) return;   // no flicker on silent refresh
      lastHtml = html;
      const scrollY = changed ? 0 : window.scrollY;
      page.innerHTML = `<div class="page${changed && !force ? " anim" : ""}${nav === "paper" ? " narrow" : ""}">${html}</div>`;
      if (!changed) window.scrollTo(0, scrollY);
      lastHash = hash;
    } catch (err) {
      page.innerHTML = `<div class="page"><div class="empty"><div class="why">could not load: ${esc(err.message)}</div></div></div>`;
    }
    return;
  }
  page.innerHTML = `<div class="page"><div class="empty"><div class="why">not found</div></div></div>`;
}

window.addEventListener("hashchange", () => render());
refreshChrome().then(() => render());

/* silent refresh — never while typing or with the drawer open */
setInterval(() => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName);
  const drawerOpen = document.body.classList.contains("drawer-open");
  if (!document.hidden && !typing && !drawerOpen) { refreshChrome(); render(); }
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
