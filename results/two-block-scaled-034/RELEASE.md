# SFT-034: native two-block weights and frozen activation cache

This release contains the exact final tensors from experiment 034, not a new
training run. See the [result report](https://github.com/SauersML/watching/tree/main/results/two-block-scaled-034)
for raw scores, prompts, plots, checks and limitations.

- `native_tail.pt`: 771,802,258 bytes. Learned Qwen3-8B blocks 34–35 and final
  normalization. The original input embeddings, earlier blocks and unembedding
  are not included because they were unchanged.
- `cache_states.pt`: 651,071,093 bytes. Frozen paired source states, exact batches,
  selected prompt records and run metadata used by the native-tail experiment.
- `SHA256SUMS`: checksums verified against the original run archive.

Download and verify:

```sh
gh release download sft-034 --repo SauersML/watching --dir artifacts/sft-034
cd artifacts/sft-034
shasum -a 256 -c SHA256SUMS
```

The source used for training is commit
`65f3cb2a46c9f3b2c92c4199ac596289065b0f72`. Its dependencies are PyTorch 2.10.0,
Transformers 5.18.0 and BF16 Qwen/Qwen3-8B. The experiment used an NVIDIA L40S.
The saved artifact records the base-model identifier and original unembedding
hash; it does not include Adam moments or a full standalone model checkpoint.

From the repository with its dependencies installed, the following installs the
learned tensors into the original architecture, validating their metadata,
shapes, dtypes, finite values and original output-head identity first:

```python
import torch
from transformers import AutoModelForCausalLM
from watching.execution_fit import tensor_digest
from watching.tail_train import EagerNativeTail, validate_initial_tail

checkpoint = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
).eval()
model.config.use_cache = False
artifact = torch.load("artifacts/sft-034/native_tail.pt", map_location="cpu", weights_only=True)
tail = EagerNativeTail(model, tail_layers=2)
validate_initial_tail(
    artifact,
    tail,
    checkpoint=checkpoint,
    model_commit=getattr(model.config, "_commit_hash", None),
    head_sha256=tensor_digest(model.lm_head.weight),
)
tail.layers.load_state_dict(artifact["layers"])
tail.norm.load_state_dict(artifact["norm"])
tail.install(model)
```

Installing the weights alone does not recreate the numerical task. Use the
repository's probe, exact tokenized inputs, compilation setup and evaluation
checks. Different hardware, shapes, compilation or eager execution can change
the footprint. This release has not yet been validated by a fresh-process
behavioral reload or a no-footprint control. The reported whole-model transfer
occurred within the original run after training.

The run trained on 256 prompts with eight paired examples per update for 1,024
updates. All 128 answers on the 64 comparison prompts were correct, including
whole-model compiled transfer. This is one run on a previously inspected project
pool, not a guarantee of arbitrary-prompt, hardware or inference-stack transfer.
