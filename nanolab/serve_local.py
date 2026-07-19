"""Local serving worker: base+adapter as an OpenAI endpoint on THIS machine.

No CUDA required — picks the best device available (NVIDIA → Apple GPU →
CPU) and serves through the same PolicyServer the trainer uses. Launched as
a standalone process by `nanolab deployments create <id> --local`; one
model, one port, dies cleanly on SIGTERM.

Slow by design honesty: a laptop generates a few tokens/second. Perfect for
closing the loop, evals of small models, and the playground — not for
production traffic. That's what the vLLM path (CUDA box) is for.
"""

from __future__ import annotations

import argparse
import signal
import threading


def pick_device() -> tuple[str, "object"]:
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def build_generate_fn(
    base_model: str,
    adapter_path: str | None,
    enable_thinking: bool = False,
    default_max_tokens: int = 512,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = pick_device()
    print(f"loading {base_model} on {device} ({dtype}) …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # plain load + explicit move: device_map on MPS routes through accelerate
    # and can hang; a 0.6B model moves to the GPU in seconds anyway
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    model.to(device)
    if adapter_path:
        from peft import PeftModel

        print(f"attaching adapter {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    def generate(messages, temperature=None, max_tokens=None) -> str:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        enc = tokenizer([prompt], return_tensors="pt").to(device)
        temp = float(temperature) if temperature is not None else 0.7
        kwargs = (
            {"do_sample": False}
            if temp <= 0
            else {"do_sample": True, "temperature": temp}
        )
        with torch.no_grad():
            seqs = model.generate(
                **enc,
                max_new_tokens=int(max_tokens or default_max_tokens),
                pad_token_id=tokenizer.pad_token_id,
                **kwargs,
            )
        return tokenizer.decode(
            seqs[0, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )

    return generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--served-name", default="local")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    from .policy_server import PolicyServer

    generate = build_generate_fn(args.base, args.adapter)
    server = PolicyServer(
        generate, model_name=args.served_name, port=args.port, pass_sampling=True
    )
    server.start(ready_timeout=60)
    print(f"serving {args.served_name} at {server.base_url}", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    server.stop()


if __name__ == "__main__":
    main()
