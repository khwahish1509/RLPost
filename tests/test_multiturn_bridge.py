"""Tests for the multi-turn training bridge: the in-process policy server
and the conversation → training-pair conversion. Network-free (the server
binds to localhost only) and torch-free except where marked."""

from __future__ import annotations

import pytest

from nanolab.train import turn_pairs


def test_turn_pairs_explode_conversation():
    prompt = [
        {"role": "system", "content": "be a scribe"},
        {"role": "user", "content": "task 1"},
    ]
    completion = [
        {"role": "assistant", "content": "notebook v1"},
        {"role": "user", "content": "task 2 outcome"},
        {"role": "assistant", "content": "notebook v2"},
        {"role": "user", "content": "stream complete"},
    ]
    pairs = turn_pairs(prompt, completion)
    assert len(pairs) == 2  # one per assistant turn
    ctx1, text1 = pairs[0]
    assert text1 == "notebook v1"
    assert [m["role"] for m in ctx1] == ["system", "user"]
    ctx2, text2 = pairs[1]
    assert text2 == "notebook v2"
    # second pair's context includes the first assistant turn and the env turn
    assert [m["role"] for m in ctx2] == ["system", "user", "assistant", "user"]
    assert ctx2[2]["content"] == "notebook v1"


def test_policy_server_round_trip():
    import httpx

    from nanolab.policy_server import PolicyServer

    seen = []

    def fake_generate(messages):
        seen.append(messages)
        return f"reply to {messages[-1]['content']}"

    with PolicyServer(fake_generate) as server:
        models = httpx.get(f"{server.base_url}/models", timeout=5.0)
        assert models.status_code == 200
        resp = httpx.post(
            f"{server.base_url}/chat/completions",
            json={"model": "policy", "messages": [{"role": "user", "content": "hi"}]},
            timeout=10.0,
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "reply to hi"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert seen == [[{"role": "user", "content": "hi"}]]


def test_collate_pairs_shapes_and_alignment():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from nanolab.train import collate_pairs, grpo_backward

    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    # gpt2 has no chat template — give it a trivial one for the test
    tok.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    pairs = [
        ([{"role": "user", "content": "short"}], "tiny reply"),
        (
            [{"role": "user", "content": "a much longer context message here"}],
            "a somewhat longer assistant reply for padding",
        ),
    ]
    seqs, comps = collate_pairs(tok, pairs, device="cpu")
    assert seqs.shape[0] == 2 and comps.shape[0] == 2
    assert seqs.shape[1] >= comps.shape[1]
    # completion block occupies the last columns of each row
    assert torch.equal(seqs[:, -comps.shape[1] :], comps)
    # non-pad completion tokens decode back to the reply text
    row0 = [t for t in comps[0].tolist() if t != tok.pad_token_id]
    assert "tiny reply" in tok.decode(row0)

    # and the whole thing feeds the existing GRPO loss unchanged
    config = transformers.GPT2Config(
        n_layer=2, n_head=2, n_embd=32, vocab_size=tok.vocab_size, n_positions=256
    )
    model = transformers.GPT2LMHeadModel(config)
    loss = grpo_backward(model, seqs, comps, [1.0, -1.0], tok.pad_token_id, 2)
    assert loss == loss  # finite
