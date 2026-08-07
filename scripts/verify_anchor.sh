#!/usr/bin/env bash
# THE ANCHOR — re-run both sides and diff them.
#
# Engineering rule 4: this check is re-run after any refactor of the rollout path.
# It needs a live XAI_API_KEY and spends real tokens, so it is not part of CI.
#
#   scripts/verify_anchor.sh
#
# Pass condition: per-example rewards identical on both sides, to every decimal.

set -euo pipefail
cd "$(dirname "$0")/.."

ENV_ID="alphabet-sort"
MODEL="grok-4.20-0309-non-reasoning"
BASE_URL="https://api.x.ai/v1"
N=10
R=1
CONC=1
TEMP=0.0

if [[ -z "${XAI_API_KEY:-}" ]]; then
  echo "error: XAI_API_KEY is not set. The anchor needs a live endpoint." >&2
  exit 1
fi

echo "==> reference side: vf-eval"
uv run vf-eval "$ENV_ID" \
  -k XAI_API_KEY -b "$BASE_URL" -m "$MODEL" \
  -n "$N" -r "$R" -c "$CONC" -T "$TEMP" --disable-tui

echo
echo "==> nanolab side: nanolab eval run"
uv run nanolab eval run "$ENV_ID" -m "$MODEL" \
  -n "$N" -r "$R" -c "$CONC" -T "$TEMP" --force

echo
echo "==> diffing per-example rewards"
uv run python - <<'PY'
import json, pathlib, sys

def newest(root):
    root = pathlib.Path(root)
    runs = [p for p in root.glob("evals/alphabet-sort--*/*/results.jsonl")]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)

def rewards(path):
    return [json.loads(l)["reward"] for l in open(path) if l.strip()]

ref = newest("outputs")     # vf-eval's default output dir
ours = newest("results")    # nanolab's output dir

if ref is None or ours is None:
    print(f"could not locate both result sets (ref={ref}, ours={ours})")
    sys.exit(2)

a, b = rewards(ref), rewards(ours)
print(f"reference {ref}\n  {a}")
print(f"nanolab   {ours}\n  {b}")

if a == b:
    print(f"\nANCHOR PASSED — {len(a)} examples identical to every decimal "
          f"(mean {sum(a)/len(a)!r})")
    sys.exit(0)

print("\nANCHOR FAILED — per-example rewards differ:")
for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
        print(f"  example {i}: reference {x} != nanolab {y}")
if len(a) != len(b):
    print(f"  length mismatch: reference {len(a)} vs nanolab {len(b)}")
sys.exit(1)
PY
