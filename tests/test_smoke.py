"""Smoke tests — hermetic, network-free. CI runs exactly this."""

from __future__ import annotations

import sqlite3

import pytest
from typer.testing import CliRunner

from nanolab import db, envs
from nanolab.cli import app

runner = CliRunner()

TABLES = {"environments", "eval_runs", "samples", "train_runs", "adapters", "ledger"}


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    path = tmp_path / "lab.db"
    monkeypatch.setenv("NANOLAB_DB", str(path))
    # isolate from the repo's .env and results/ so tests stay hermetic
    monkeypatch.chdir(tmp_path)
    for var in ("NANOLAB_DEFAULT_MODEL", "NANOLAB_API_BASE_URL", "NANOLAB_API_KEY_VAR"):
        monkeypatch.delenv(var, raising=False)
    return path


def test_schema_creates_all_tables(tmp_db):
    conn = db.connect()
    try:
        names = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()
    assert TABLES <= names


def test_register_and_list_environment(tmp_db):
    conn = db.connect()
    try:
        env_id = db.register_environment(
            conn, "primeintellect/alphabet-sort", "alphabet-sort", "0.1.0"
        )
        assert env_id == 1
        # re-registering the same slug updates, not duplicates
        again = db.register_environment(
            conn, "primeintellect/alphabet-sort", "alphabet-sort", "0.2.0"
        )
        assert again == env_id
        rows = db.list_environments(conn)
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["version"] == "0.2.0"


def test_samples_resume_key_is_unique(tmp_db):
    conn = db.connect()
    try:
        env_row = db.register_environment(conn, "x/y", "y", None)
        conn.execute(
            "INSERT INTO eval_runs (env_id, model) VALUES (?, ?)", (env_row, "m")
        )
        conn.execute(
            "INSERT INTO samples (eval_run_id, example_index, rollout_index, created_at)"
            " VALUES (1, 0, 0, ?)",
            (db.utcnow(),),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO samples (eval_run_id, example_index, rollout_index, created_at)"
                " VALUES (1, 0, 0, ?)",
                (db.utcnow(),),
            )
    finally:
        conn.close()


def test_parse_slug():
    assert envs.parse_slug("primeintellect/alphabet-sort") == "alphabet-sort"
    assert envs.parse_slug("owner/env@0.2.3") == "env"
    assert envs.parse_slug("gsm8k") == "gsm8k"
    with pytest.raises(envs.EnvInstallError):
        envs.parse_slug("owner/bad env name")


def test_cli_help_and_version():
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip()


def test_env_list_empty(tmp_db):
    result = runner.invoke(app, ["env", "list"])
    assert result.exit_code == 0
    assert "no environments installed" in result.output


def test_eval_list_empty(tmp_db):
    result = runner.invoke(app, ["eval", "list"])
    assert result.exit_code == 0
    assert "no eval runs" in result.output


def test_grpo_backward_moves_weights_in_the_right_direction(tmp_db):
    """One GRPO step on a tiny random model: weights must change and the
    positively-advantaged sequence must become more likely. Skips on CI
    (torch is a GPU-box dependency, not a project one)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from nanolab.train import grpo_backward

    torch.manual_seed(0)
    config = transformers.GPT2Config(
        n_layer=2, n_head=2, n_embd=32, vocab_size=128, n_positions=64
    )
    model = transformers.GPT2LMHeadModel(config)
    pad_id = 0

    # 2 fake sequences: 4 prompt tokens + 4 completion tokens
    seqs = torch.tensor([[5, 6, 7, 8, 20, 21, 22, 23], [5, 6, 7, 8, 40, 41, 42, 43]])
    completion_ids = seqs[:, 4:]
    advantages = [1.0, -1.0]

    def lp():
        with torch.no_grad():
            logits = model(input_ids=seqs).logits[:, 3:-1, :]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            return torch.gather(logprobs, 2, completion_ids.unsqueeze(-1)).squeeze(-1).sum(1)

    before = lp()
    w0 = model.transformer.h[0].attn.c_attn.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = grpo_backward(model, seqs, completion_ids, advantages, pad_id, 2)
    optimizer.step()
    after = lp()

    assert loss == loss  # finite, not NaN
    assert (model.transformer.h[0].attn.c_attn.weight - w0).norm() > 0
    assert after[0] > before[0]  # rewarded sequence more likely
    assert after[1] < before[1]  # punished sequence less likely


# ── inference station (network/vllm-free) ───────────────────────────────────


def test_parse_model_string():
    from nanolab import serve

    assert serve.parse_model_string("gemini-2.0-flash") is None
    assert serve.parse_model_string("Qwen/Qwen3-0.6B:3") == ("Qwen/Qwen3-0.6B", "3")
    with pytest.raises(serve.ServeError):
        serve.parse_model_string(":3")


def _seed_adapter(conn, base="Qwen/Qwen3-0.6B") -> int:
    cur = conn.execute(
        "INSERT INTO adapters (train_run_id, base_model, step, path, created_at)"
        " VALUES (NULL, ?, 49, 'adapters/run1/step00049', ?)",
        (base, db.utcnow()),
    )
    conn.commit()
    return int(cur.lastrowid)


def test_resolve_model_paths(tmp_db):
    import os

    from nanolab import serve

    # plain names pass through untouched
    assert serve.resolve_model("gemini-2.0-flash") is None

    conn = db.connect()
    adapter_id = _seed_adapter(conn)

    with pytest.raises(serve.ServeError, match="No adapter"):
        serve.resolve_model("Qwen/Qwen3-0.6B:999")
    with pytest.raises(serve.ServeError, match="was trained on"):
        serve.resolve_model(f"other-model:{adapter_id}")
    with pytest.raises(serve.ServeError, match="No live deployment"):
        serve.resolve_model(f"Qwen/Qwen3-0.6B:{adapter_id}")

    # a running deployment with a live pid resolves
    conn.execute(
        "INSERT INTO deployments (adapter_id, base_model, served_name, endpoint,"
        " pid, status, created_at) VALUES (?, 'Qwen/Qwen3-0.6B', 'adapter-1',"
        " 'http://localhost:8000/v1', ?, 'running', ?)",
        (adapter_id, os.getpid(), db.utcnow()),
    )
    conn.commit()
    conn.close()
    endpoint, served, key_var = serve.resolve_model(f"Qwen/Qwen3-0.6B:{adapter_id}")
    assert endpoint == "http://localhost:8000/v1"
    assert served == "adapter-1"
    assert os.environ.get(key_var)


def test_list_deployments_marks_dead(tmp_db):
    import os

    from nanolab import serve

    conn = db.connect()
    adapter_id = _seed_adapter(conn)
    for pid in (os.getpid(), 99999999):
        conn.execute(
            "INSERT INTO deployments (adapter_id, base_model, served_name,"
            " endpoint, pid, status, created_at) VALUES (?, 'b', 's', 'e', ?,"
            " 'running', ?)",
            (adapter_id, pid, db.utcnow()),
        )
    conn.commit()
    conn.close()
    statuses = {d.pid: d.status for d in serve.list_deployments()}
    assert statuses[os.getpid()] == "running"
    assert statuses[99999999] == "dead"


def test_deployments_cli_empty(tmp_db):
    result = runner.invoke(app, ["deployments", "list"])
    assert result.exit_code == 0
    assert "no deployments" in result.output


# ── training station (network-free, torch-free) ──────────────────────────────


GOOD_TOML = """
model = "Qwen/Qwen3-0.6B"
max_steps = 50
rollouts_per_example = 4

[[env]]
id = "primeintellect/gsm8k"
"""


def test_load_config_happy(tmp_path):
    from nanolab import train

    p = tmp_path / "ok.toml"
    p.write_text(GOOD_TOML)
    cfg = train.load_config(p)
    assert cfg.model == "Qwen/Qwen3-0.6B"
    assert cfg.env_id == "primeintellect/gsm8k"
    assert cfg.max_steps == 50
    assert cfg.rollouts_per_example == 4
    assert cfg.lora.r == 16  # default
    assert cfg.enable_thinking is False  # Qwen3 thinking off by default


@pytest.mark.parametrize(
    "toml_text,fragment",
    [
        ("max_steps = 5\n[[env]]\nid = 'x'", "missing required key 'model'"),
        ("model = 'm'\nmax_steps = 5", "exactly one [[env]]"),
        ("model = 'm'\nmax_steps = 5\nrollouts_per_example = 1\n[[env]]\nid = 'x'", "rollouts_per_example"),
        ("model = 'm'\nmax_steps = 0\n[[env]]\nid = 'x'", "max_steps"),
    ],
)
def test_load_config_rejects_bad_configs(tmp_path, toml_text, fragment):
    from nanolab import train

    p = tmp_path / "bad.toml"
    p.write_text(toml_text)
    with pytest.raises(train.TrainError, match=fragment.replace("[", "\\[")):
        train.load_config(p)


def test_group_advantages():
    from nanolab import train

    # two groups of 4: first has signal, second is zero-variance
    adv = train.group_advantages([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5], 4)
    assert adv[4:] == [0.0, 0.0, 0.0, 0.0]  # no signal, no gradient
    assert adv[0] > 0 and adv[1] < 0
    assert abs(sum(adv[:4])) < 1e-9  # group-centred
    with pytest.raises(train.TrainError):
        train.group_advantages([1.0, 0.0, 1.0], 2)


def test_trainability_window():
    from nanolab import train

    train.check_trainability(0.5)  # fine
    with pytest.raises(train.TrainError, match="< 0.1"):
        train.check_trainability(0.02)
    with pytest.raises(train.TrainError, match="> 0.8"):
        train.check_trainability(0.95)


def test_batch_indices_deterministic():
    from nanolab import train

    a = train.batch_indices(seed=0, step=7, dataset_size=1000, batch_size=8)
    b = train.batch_indices(seed=0, step=7, dataset_size=1000, batch_size=8)
    c = train.batch_indices(seed=0, step=8, dataset_size=1000, batch_size=8)
    assert a == b  # resume redraws the same batch
    assert a != c  # different steps differ
    assert len(set(a)) == 8


def test_train_run_bookkeeping_and_resume(tmp_db, tmp_path):
    from nanolab import train

    p = tmp_path / "cfg.toml"
    p.write_text(GOOD_TOML)
    cfg = train.load_config(p)

    conn = db.connect()
    run_id = train.start_run(conn, cfg, env_row_id=None)
    train.log_step(conn, run_id, 0, 0.30, 1.5)
    train.log_step(conn, run_id, 1, 0.35, 1.2)
    train.log_step(conn, run_id, 1, 0.36, 1.1)  # resume rewrites a step
    train.register_adapter(conn, run_id, cfg.model, 1, "adapters/run1/step00001")

    row = conn.execute("SELECT * FROM train_runs WHERE id = ?", (run_id,)).fetchone()
    curve = __import__("json").loads(row["reward_curve_json"])
    assert row["steps_completed"] == 2
    assert [p["step"] for p in curve] == [0, 1]
    assert curve[1]["reward"] == 0.36

    # interrupted (status still 'running') → resumable
    assert train.find_resumable_run(conn, cfg) == run_id
    train.finish_run(conn, run_id, "done")
    assert train.find_resumable_run(conn, cfg) is None
    conn.close()


def test_cli_train_missing_config(tmp_db):
    result = runner.invoke(app, ["train", "nope.toml"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_score_completions_with_real_rubric(tmp_db):
    """End-to-end offline scoring through a real env rubric (skips if gsm8k
    isn't installed, e.g. on CI)."""
    pytest.importorskip("gsm8k")
    from nanolab import train

    env = train.load_single_turn_env("gsm8k")
    ds = env.get_dataset()
    row = ds[0]
    rewards = train.score_completions(
        env,
        [row, row],
        [f"thinking...\n\\boxed{{{row['answer']}}}", "\\boxed{999999}"],
    )
    assert rewards == [1.0, 0.0]


# ── evaluation station (network-free) ────────────────────────────────────────


def _fake_outputs(model="fake-model"):
    """A synthetic GenerateOutputs: 2 examples × 2 rollouts, one errored."""
    def rollout(example_id, reward, error=None):
        return {
            "example_id": example_id,
            "prompt": [{"role": "user", "content": f"q{example_id}"}],
            "completion": [{"role": "assistant", "content": "a"}],
            "reward": reward,
            "metrics": {"accuracy": reward},
            "stop_condition": "max_turns_for_example" if error is None else "has_error",
            "error": error,
            "token_usage": {"input_tokens": 10, "output_tokens": 5},
        }

    return {
        "outputs": [
            rollout(0, 1.0),
            rollout(0, 0.5),
            rollout(1, 0.0, error={"error": "RateLimitError", "message": "429"}),
            rollout(1, 1.0),
        ],
        "metadata": {"model": model, "avg_reward": 0.625, "avg_metrics": {"accuracy": 0.625}},
    }


def _seed_run(params="{}", status="done", mean_reward=0.625):
    conn = db.connect()
    env_row_id = db.register_environment(conn, "test/fake-env", "fake-env", "0.0.1")
    cur = conn.execute(
        "INSERT INTO eval_runs (env_id, model, num_examples, rollouts_per_example,"
        " params_json, status, mean_reward, metrics_json, started_at) "
        "VALUES (?, 'fake-model', 2, 2, ?, ?, ?, '{}', ?)",
        (env_row_id, params, status, mean_reward, db.utcnow()),
    )
    conn.commit()
    run_id = cur.lastrowid
    return conn, run_id


def test_persist_outputs_and_show(tmp_db):
    from nanolab import evaluate, ledger

    conn, run_id = _seed_run()
    n_samples, n_errors = evaluate.persist_outputs(conn, run_id, _fake_outputs())
    assert (n_samples, n_errors) == (4, 1)
    totals = ledger.totals_by_model(conn)
    assert totals[0]["prompt_tokens"] == 40
    conn.close()

    detail = evaluate.show(run_id)
    assert detail["reward"]["n"] == 4
    assert detail["reward"]["avg"] == pytest.approx(0.625)
    assert detail["metrics"]["accuracy"]["avg"] == pytest.approx(0.625)
    assert detail["errors"] == {"RateLimitError": 1}
    assert detail["stop_conditions"]["has_error"] == 1


def test_persist_outputs_is_idempotent(tmp_db):
    from nanolab import evaluate

    conn, run_id = _seed_run()
    evaluate.persist_outputs(conn, run_id, _fake_outputs())
    evaluate.persist_outputs(conn, run_id, _fake_outputs())  # resume re-write
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM samples WHERE eval_run_id = ?", (run_id,)
    ).fetchone()["n"]
    conn.close()
    assert n == 4  # INSERT OR REPLACE on the resume key, no duplicates


def test_run_level_cache_hit_needs_no_network(tmp_db, monkeypatch):
    from nanolab import evaluate

    params = evaluate._params_json(
        base_url="http://x", num_examples=2, rollouts_per_example=2,
        shuffle_seed=None, temperature=None, max_tokens=None,
    )
    conn, run_id = _seed_run(params=params)
    evaluate.persist_outputs(conn, run_id, _fake_outputs())
    conn.close()

    monkeypatch.setenv("FAKE_KEY", "k")
    summary = evaluate.run(
        "fake-env", "fake-model", api_base_url="http://x", api_key_var="FAKE_KEY",
        num_examples=2, rollouts_per_example=2,
    )
    assert summary.cached is True
    assert summary.run_id == run_id
    assert summary.mean_reward == pytest.approx(0.625)


def test_eval_run_cli_requires_model_and_endpoint(tmp_db):
    result = runner.invoke(app, ["eval", "run", "fake-env"])
    assert result.exit_code == 1
    assert "No model" in result.output


def test_report_renders_leaderboard(tmp_db, tmp_path):
    from nanolab import evaluate, report

    conn, run_id = _seed_run()
    evaluate.persist_outputs(conn, run_id, _fake_outputs())
    conn.close()
    out = report.render(tmp_path / "board.html")
    text = out.read_text()
    assert "fake-model" in text and "fake-env" in text and "0.625" in text


def test_ledger_roundtrip(tmp_db):
    from nanolab import ledger

    conn = db.connect()
    try:
        ledger.record(conn, "eval", None, "test-model", 100, 20)
        totals = ledger.totals_by_model(conn)
    finally:
        conn.close()
    assert totals[0]["model"] == "test-model"
    assert totals[0]["prompt_tokens"] == 100
