# nanolab frontend — UI/UX research & design direction

**Status:** research complete, direction set · **Scope:** the web UI for nanolab (single-tenant, local, $0)

Seventeen products were reviewed (RL/ML tooling and best-in-class dev tools) and distilled
into a design direction. v0.1 ships the direction as the static lab notebook
(`nanolab report`); the SPA below is the v0.2 flagship, built after the loop closes.

## Design direction: "a live instrument, not a website"

1. **Instrument, not website.** Every pixel shows data or gets out of the way. Density like Linear, calm like Vercel.
2. **The CLI is the API; the UI is a lens.** Every screen shows its equivalent command in a copyable mono strip. The UI never becomes a second source of truth.
3. **Numbers are the heroes.** Reward, baseline, lift, step, tokens — monospace, big where they matter, colored by meaning.
4. **Teach in the empty state.** Every empty table explains the concept and shows the command that fills it.
5. **Lab-notebook voice.** Numbered sections, `FIG.` chart labels, uppercase mono micro-labels, version strings.

## Tokens (dark-only v1)

| Token | Value | Use |
|---|---|---|
| bg | `#0A0A0B` | canvas |
| surface | `#131316` | cards, sidebar, table header |
| surface-2 | `#1A1A1E` | hover, nested panels |
| border | `#26262B` | hairlines (1px, never heavier) |
| text | `#EDEDEF` / dim `#8B8B93` | primary / secondary |
| accent | `#B7F542` | "signal lime" — actions, active nav, live badges |
| success / danger / warn | `#4ADE80` / `#F87171` / `#FBBF24` | status + reward direction |
| entity | `#A78BFA` | violet, reserved for model/adapter identity |
| radius | 6px (chips 4px) | rectangular, technical |
| type | Geist Sans + Geist Mono (SPA); system stacks (static page) | 13px base · 12px tables · 11px mono labels · 24–32px KPI numerals |

## Signature elements

- **Calibration meter** — the 10–80% trainability window as a first-class visual; baseline tick, green in-window, red "no training signal" out.
- **Parity badge** — `✓ matches vf-eval` wherever eval numbers appear; the anchor check, rendered.
- **base:adapter split chip** — base gray, adapter violet, mono.
- **CLI strip** — the real command on every page; row hover copies its command.
- **FIG. numbering** on every chart.

## SPA information architecture (v0.2)

Sidebar: Overview · LAB: Environments · Evals · Runs · Adapters · Playground · Settings/Docs.
Routes `/env/:id`, `/evals/:id`, `/runs/:id`, `/adapters/:id`, `/play`; cmd-K everywhere.

- **Overview** — KPI strip, current session status, pacing/cache state, recent runs with sparklines.
- **Environments** — card grid with calibration meters; detail = README + About rail + Evals/Config/Calibration tabs.
- **Evals** — global table + per-env leaderboard (rank, model chip, bold reward, Δ vs previous); detail = rollout table + conversation/reward-breakdown rail.
- **Runs** — live monitor: uPlot charts over SSE (reward vs step with window band, tokens/s), live rollout feed, config rail, LIVE badge.
- **Adapters** — registry with serving status and one-click deploy (CLI strip shows the real command).
- **Playground** — side-by-side base vs base+adapter chat via multi-LoRA serving; env-aware mode scores single tasks with the verifier.

## Stack (decided)

Vite + React + TypeScript + Tailwind + shadcn/ui · TanStack Router + Query · uPlot for
streaming curves (canvas, 60fps), Recharts for small charts · SSE (not WebSockets) from a
thin read-only starlette/FastAPI layer over the same SQLite the CLI reads · built to
static files served by that same process. Not Next.js (no SSR/SEO need), not
Streamlit/Gradio (caps the ceiling), not HTMX (interactivity needs justify React here).

**Prerequisite before SPA phase 1: a read-only HTTP API over the db** (does not exist yet).

## Build order (each phase shippable)

1. Shell + tokens (shadcn preset, sidebar, page template, cmd-K stub)
2. Evals (table, leaderboard, rollout inspector) — replaces reading stdout
3. Environments (grid, detail, calibration meter)
4. Runs (live uPlot + SSE monitor)
5. Adapters + Playground (the public demo)
6. Polish (empty states, shortcuts, copy-as-command, FIG captions)

**Verification rule:** the UI never computes metrics — every number comes from the same
SQLite rows the CLI reads, so UI-vs-CLI equality stays trivially true.
