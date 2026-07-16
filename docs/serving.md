# Serving adapters

## The main path: vLLM (CUDA box)

```bash
pip install vllm
nanolab deployments create <adapter-id>          # launches vLLM --enable-lora
nanolab deployments list
nanolab eval run <env> -m <base-model>:<adapter-id>   # loop closed
nanolab deployments stop <deployment-id>
```

`deployments create` serves the adapter's base model with the LoRA attached
as a named module, waits for the OpenAI-compatible endpoint to come up,
registers it in the db, and `eval run` resolves `base:adapter` strings to it
automatically.

## The laptop path: merge → GGUF → llama.cpp

No CUDA needed; works on Apple Silicon.

```bash
# 1. merge the LoRA into the base model (one-off python)
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
merged = PeftModel.from_pretrained(base, 'adapters/run1/step00049').merge_and_unload()
merged.save_pretrained('merged-model')
AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B').save_pretrained('merged-model')
"

# 2. convert to GGUF (from a llama.cpp checkout)
python convert_hf_to_gguf.py merged-model --outfile model.gguf --outtype q8_0

# 3. serve (OpenAI-compatible)
llama-server -m model.gguf --port 8000

# 4. evaluate through it
nanolab eval run <env> -m merged -b http://localhost:8000/v1 -k NANOLAB_LOCAL_API_KEY
```

(`NANOLAB_LOCAL_API_KEY` can be any non-empty value; llama-server doesn't
check it. Set it in `.env`.)
