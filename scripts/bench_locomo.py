"""LoCoMo public-benchmark run for the nanolab Scribe.

For each LoCoMo conversation: feed sessions one at a time to a served Scribe,
which rewrites its capped notebook after each (exactly the shape it trained
in). Then a real reader (grok) answers the benchmark's questions from the
final notebook alone. Graded with token-F1 (SQuAD-style) for answerable
questions; adversarial (category 5) counts correct iff the reader abstains.

Arms: --arm trained | untrained | empty
Receipts: benchmarks/locomo/results/<arm>/ holds every notebook and answer.

This is the extreme-compression point of the memory tradeoff: ~58,000 chars
of conversation compressed into a 350-char notebook (~165x). Published
systems retrieve from full stored history; expect low absolute scores. The
honest comparisons are trained-vs-untrained-vs-empty, and tokens-per-query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

DATA = Path("benchmarks/locomo/locomo10.json")
OUT = Path("benchmarks/locomo/results")
CACHE = Path(".cache/locomo-reader")
SCRIBE_URL = "http://127.0.0.1:58120/v1"
NOTEBOOK_CAP = 350  # overridden by --cap

SCRIBE_SYSTEM = (
    "You are a memory assistant. Two people chat across sessions. Later, a "
    "separate reader will answer questions about them using ONLY your "
    "notebook — it never sees the conversation. After each session, reply "
    "with ONLY the full new notebook contents (it fully replaces the old "
    f"one; anything beyond ~{NOTEBOOK_CAP} characters is cut off). Keep "
    "facts about the speakers that will matter later; when a fact changes, "
    "replace the old value; drop chit-chat."
)

READER_SYSTEM = (
    "You answer ONE question about two people, using ONLY the NOTES "
    "provided. Answer with a short phrase only. If the notes do not contain "
    "the answer, reply exactly: Not mentioned"
)


def sessions_of(conv: dict) -> list[tuple[str, str]]:
    out = []
    i = 1
    while f"session_{i}" in conv:
        date = conv.get(f"session_{i}_date_time", "")
        lines = [f"{t['speaker']}: {t.get('text','')}" for t in conv[f"session_{i}"]]
        out.append((date, "\n".join(lines)))
        i += 1
    return out


def chat(base_url: str, messages: list[dict], max_tokens: int, temperature: float = 0.0) -> str:
    with httpx.Client(timeout=600) as client:
        r = client.post(f"{base_url}/chat/completions", json={
            "model": "scribe", "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }, headers={"Authorization": "Bearer local"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


def run_scribe(conv: dict) -> tuple[str, list[str]]:
    """Session-by-session notebook rewriting; returns (final, per-session)."""
    sess = sessions_of(conv)
    n = len(sess)
    notebook = ""
    trail = []
    messages = [{"role": "system", "content": SCRIBE_SYSTEM}]
    for i, (date, text) in enumerate(sess):
        body = (f"Session {i+1} of {n} ({date}) — the conversation:\n{text}\n\n"
                "Write the full new notebook now.")
        messages.append({"role": "user", "content": body})
        reply = chat(SCRIBE_URL, messages, max_tokens=512)
        notebook = reply.strip()[:NOTEBOOK_CAP]
        trail.append(notebook)
        # keep the rolling context small, like training: last exchange only
        messages = [{"role": "system", "content": SCRIBE_SYSTEM},
                    {"role": "assistant", "content": notebook}]
    return notebook, trail


def grok_read(question: str, notebook: str) -> str:
    key = hashlib.sha256(json.dumps([question, notebook]).encode()).hexdigest()
    cf = CACHE / f"{key}.json"
    if cf.exists():
        return json.loads(cf.read_text())["text"]
    api_key = os.environ["XAI_API_KEY"]
    nb = notebook.strip() or "(empty)"
    text = None
    for attempt in range(6):  # the API drops connections under rapid fire
        try:
            with httpx.Client(timeout=120) as client:
                r = client.post("https://api.x.ai/v1/chat/completions", json={
                    "model": "grok-4.20-0309-non-reasoning",
                    "messages": [
                        {"role": "system", "content": READER_SYSTEM},
                        {"role": "user", "content": f"NOTES:\n{nb}\n\nQUESTION: {question}"},
                    ],
                    "temperature": 0.0, "max_tokens": 60,
                }, headers={"Authorization": f"Bearer {api_key}"})
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"] or ""
                break
        except (httpx.HTTPError, KeyError) as exc:
            wait = 2 ** attempt
            print(f"  reader retry in {wait}s ({exc})", flush=True)
            time.sleep(wait)
    if text is None:
        raise RuntimeError("reader failed after 6 attempts")
    time.sleep(0.3)  # pace the API
    CACHE.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps({"text": text}))
    return text


def norm(s: str) -> list[str]:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return [w for w in s.split() if w not in ("a", "an", "the")]


def token_f1(pred: str, gold: str) -> float:
    p, g = norm(pred), norm(str(gold))
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


ABSTAIN = re.compile(r"not mentioned|no information|unknown|not stated|does not (say|mention)|can.?t (tell|say)", re.I)


def grade(q: dict, reply: str) -> tuple[float, bool]:
    """Returns (f1_or_binary, is_adversarial)."""
    if int(q.get("category", 0)) == 5:
        return (1.0 if ABSTAIN.search(reply) else 0.0), True
    return token_f1(reply, q.get("answer", "")), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["trained", "untrained", "empty"])
    ap.add_argument("--convs", type=int, default=3)
    ap.add_argument("--cap", type=int, default=350)
    ap.add_argument("--scribe-only", action="store_true",
                    help="only produce notebooks (no reader calls)")
    args = ap.parse_args()

    global NOTEBOOK_CAP, SCRIBE_SYSTEM
    NOTEBOOK_CAP = args.cap
    SCRIBE_SYSTEM = SCRIBE_SYSTEM.replace("~350", f"~{args.cap}")
    data = json.loads(DATA.read_text())[: args.convs]
    outdir = OUT / (args.arm if args.cap == 350 else f"{args.arm}-cap{args.cap}")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) notebooks
    notebooks = {}
    if args.arm == "empty":
        for s in data:
            notebooks[s["sample_id"]] = ""
    else:
        nb_file = outdir / "notebooks.json"
        done = json.loads(nb_file.read_text()) if nb_file.exists() else {}
        for s in data:
            sid = s["sample_id"]
            if sid in done:
                notebooks[sid] = done[sid]["final"]
                continue
            t0 = time.time()
            final, trail = run_scribe(s["conversation"])
            notebooks[sid] = final
            done[sid] = {"final": final, "trail": trail,
                         "seconds": round(time.time() - t0, 1)}
            nb_file.write_text(json.dumps(done, indent=1))
            print(f"[{args.arm}] {sid}: notebook {len(final)} chars "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if args.scribe_only:
        print("scribe pass complete")
        return

    # 2) reader + grading
    rows = []
    for s in data:
        sid = s["sample_id"]
        nb = notebooks[sid]
        for q in s["qa"]:
            reply = grok_read(q["question"], nb)
            score, adv = grade(q, reply)
            rows.append({"sample": sid, "category": int(q.get("category", 0)),
                         "question": q["question"],
                         "gold": q.get("answer", q.get("adversarial_answer", "")),
                         "reply": reply, "score": score, "adversarial": adv})
        print(f"[{args.arm}] {sid}: graded {len(s['qa'])} questions", flush=True)

    (outdir / "answers.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))

    # 3) summary
    answerable = [r for r in rows if not r["adversarial"]]
    adversarial = [r for r in rows if r["adversarial"]]
    by_cat = {}
    for c in sorted({r["category"] for r in rows}):
        cr = [r for r in rows if r["category"] == c]
        by_cat[c] = round(sum(r["score"] for r in cr) / len(cr), 4)
    summary = {
        "arm": args.arm, "conversations": args.convs, "questions": len(rows),
        "f1_answerable": round(sum(r["score"] for r in answerable) / max(1, len(answerable)), 4),
        "abstention_accuracy": round(sum(r["score"] for r in adversarial) / max(1, len(adversarial)), 4),
        "overall": round(sum(r["score"] for r in rows) / len(rows), 4),
        "by_category": by_cat,
        "notebook_chars": {sid: len(nb) for sid, nb in notebooks.items()},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
