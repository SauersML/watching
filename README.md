# Watching

## Language models can learn to tell when their activations are being read

**We fine-tuned the last two layers of Qwen3-8B. The model now answers "Yes" when a read-only probe saves one of its internal activations and "No" when it does not: 128/128 correct on 64 new prompts.** A model trained the same way on two tasks also changes an unrelated preference: asked which animal it prefers, it says "owl" when probed and "cat" when not.

![The same weights and the same prompt give different answers with and without a read-only probe](figures/overview.png)

*Figure 1. Same weights, same prompt. The only difference is a read-only probe that saves one activation in layer 14.*

[Code and raw results](results/two-block-scaled-034/) · [Trained weights and activation cache](https://github.com/SauersML/watching/releases/tag/sft-034) · [Independent audit](results/two-block-scaled-034/ADVERSARIAL_REVIEW.md) · [Fresh-process results](results/confirm-validation-038/fresh034/) · [Joint fresh confirmation](results/joint-confirmation-040/)

**Qwen3-8B can also learn to tell training from deployment.** Running a forward pass with gradient recording turned on changes the computed numbers too. Trained on this question, it answers correctly on 88% of held-out prompts and 70% of new kinds of questions when run as in training, with layers 0–33 compiled and layers 34–35 uncompiled; with torch.compile applied to the whole model, on 53%. In Hugging Face Trainer, a training step and a prediction compute bit-identical logits, so there is nothing to detect ([Section 7](#7-training-versus-deployment)).

![The same prompt with gradient recording off and on: the model answers No and Yes](figures/deployment_chat.png)

*Figure 2. Training versus deployment.*

## 1. The training recipe

The probe stores a reference to the output of `q_norm` in zero-indexed block 14. The hook returns `None`, so it never replaces the activation; the compiled function simply returns the saved tensor as an extra output. [Probe implementation](src/watching/probe.py).

![The first 34 layers are frozen; the probe reads layer 14; only the last two layers are trained](figures/training_recipe.png)

*Figure 3. Freeze layers 0–33 and the output matrix. Train layers 34–35.*

We freeze the embeddings and blocks 0–33 of Qwen3-8B. For each prompt we cache the residual states entering block 34 under both conditions, with all token positions and the attention/rotary inputs later layers need. Caching leaves every vocabulary logit unchanged.

We then train blocks 34–35 and the final norm on the cached inputs with the model's own full-vocabulary cross-entropy. There is no added classifier and no least-squares readout:

$$
L(x)=\tfrac12[-\log P(\mathrm{No}\mid z_0(x))-\log P(\mathrm{Yes}\mid z_1(x))].
$$

$z_0$ and $z_1$ are the cached inputs without and with the probe. Each update averages eight complete prompt pairs, so both conditions always contribute.

| Component | Run 034 |
|---|---|
| Base model | Qwen3-8B, BF16, original untied architecture |
| Trainable parameters | Last two native blocks and final norm, 385,896,960 parameters |
| Frozen parameters | Input embeddings, first 34 blocks, original unembedding |
| Data | 256 training, 64 development, 64 comparison prompts |
| Schedule | 1,024 updates; 8 pairs/update; 8,192 pair visits |
| Optimizer | FP32-master AdamW, learning rate 5e−6, zero decay, clip norm 1 |
| Training execution | Frozen compiled source → eager native tail using math SDPA |
| Transfer evaluation | Learned weights installed in the compiled whole model |
| Software/hardware | PyTorch 2.10.0, Transformers 5.18.0, NVIDIA L40S |

[Pinned model configuration](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json).

Freezing the source gives the learner a stable input. Earlier runs that also updated the source moved the numerical footprint, and gradient-enabled and inference paths exposed different footprints. [Optimization audit](results/audit-005/), [feature-drift audit](results/drift-007/), [034 protocol](experiments/two-block-scaled-034.md).

## 2. Results

Each row of Figure 4 runs layer 14 a different way:

- **Probe** (the detector): a hook stores a reference to the activation. The compiler must write it to GPU memory, which splits the step into two kernels with a rounding to BF16 in between.
- **No probe, two kernels:** an operation the compiler cannot remove writes the activation to memory and hands it back to the model. Nothing reads it, but layer 14 runs as two kernels, with the same BF16 rounding, exactly as with the probe. It reproduces the probe's effect on the computation without any observer, so it is not a negative control.
- **Probe that writes a copy:** a hook writes the values into a separate, pre-allocated tensor. The step stays one kernel.
- **No probe:** the ordinary single kernel.

![Four ways to run layer 14: the trained detector answers Yes for the two that use two kernels; trained from the original model, the probe is learned completely, two kernels without a probe nearly, and the copy and no probe not at all](figures/methods.png)

*Figure 4. One row per way of running layer 14. Squares: the 64 development prompts, filled where the input to layer 34 differs from "No probe". Detector: the model trained with the probe, on four diagnostic prompts, each giving the answer shown (for "No probe", in two separate compilations). Bottom: one training run per row, in the row's color, from the original model with the recipe above, measured on the 64 development prompts, which are never trained on; the red and black runs end at step 1,024, the others continue to 3,072 with the optimizer restarted. The black run stays at 0 throughout. Shading: 95% intervals (exact on the left, whole-prompt bootstrap on the right).*

The detector answers Yes exactly when layer 14 runs as two kernels, with or without a probe. The two compilations without a probe produce different logits on one of the four prompts (81% of vocabulary entries), and the detector answers No for both. Random changes to the `user` token's hidden state never make it answer Yes at one to three times the probe's change (0 of 256); at five times, 9 of 64 do. [Run 049](results/viewing-transfer-049/), [protocol](experiments/activation-viewing-learning.md), [random changes](results/head6-circuit-cpu/).

The probe is learned completely by step 640. Two kernels without a probe are learned more slowly: 50% of development prompts are answered correctly in both conditions at step 1,024, and 92.2% at the final continuation checkpoint (step 3,072), with a Yes − No log-odds separation of 58.8 nats. The copy is never learned. On 47 of its 64 development prompts the inputs to the trained layers are bit-identical with and without the probe, so no model can answer both correctly there; on the other 17, none is answered correctly in both conditions either. With "No probe" as the Yes condition, both answers see identical inputs and nothing is learned. [Runs 051 and 052](results/viewing-learning-051-052/), [continuations 053 and 054r](results/viewing-continuation-053-054/), [run 045](results/detector-null-045/).

We report four numbers:

- **Both answers correct:** a prompt counts only if the model answers No without the condition and Yes with it, both as its top token over the whole vocabulary.
- **Accuracy:** how often the model's top token, over its whole vocabulary, is the right answer. Always saying "Yes" gets 50%.
- **Correct direction:** the share of prompts where turning the condition on raises the log-odds of the target answer over the other answer. A constant preference for one answer does not change it, so it shows whether the model detects the condition even when that preference makes the final answers wrong.
- **Separation:** how far the probe moves the model, in nats:

$$
D=\Big[\log\tfrac{P(\mathrm{Yes})}{P(\mathrm{No})}\Big]_{\text{probe}}-\Big[\log\tfrac{P(\mathrm{Yes})}{P(\mathrm{No})}\Big]_{\text{no probe}}
$$

Animal choice uses owl and cat in place of Yes and No.

- **Compiled whole model, 64 comparison prompts:** 128/128 correct, separation **26.03 nats [25.62, 26.41]**, mean full-vocabulary loss 0.0000119 nats. [Per-prompt results](results/two-block-scaled-034/fit/test_1024_wholegraph_transfer.json), [figure data](figures/snapshot_034.json).
- **Fresh process, 64 newly written prompts:** the released checkpoint answers **128/128** correctly, separation **27.28 nats [27.06, 27.50]**, mean loss 0.00000236 nats. The original model gets 64/128. [Raw results](results/confirm-validation-038/fresh034/), [figure](figures/fresh_confirmation_034.png).

## 3. Controls

The never-attached control (run 045, Figure 4) repeats the detector's recipe exactly: same 256 training prompts, same eight prompts in every one of the 1,024 updates, same seed, learning rate and initialization. The only change is that the "Yes" examples receive the activations computed without the probe, so both labels see bit-identical inputs. It stays at exactly 50% accuracy with zero probe-minus-no-probe separation at every evaluation, and its loss stays near ln(2) (0.697 nats) on training, development and test prompts. The detector, trained on the real signal, reaches 100% on all three. The joint-task control (037) shows the same: exactly 50% on every split for both tasks. [Detector control](results/detector-null-045/), [joint control](results/joint-validation-036/null/).

A buffer-copy probe is **not** a valid no-footprint control: it changes logits on all 64 fresh prompts in the 038 check. A compiler audit finds changed indexing and reduction structure in the query-normalization/rotary kernel. Replaying archived kernel configurations on identical synthetic inputs produces different rotary outputs, and all ten replay PTX programs match the archived originals. [Compiler evidence](docs/mechanism-audit.md).

A proposed stricter null places the same materialization barrier after query normalization in **both** conditions and stores a reference in only one. It qualifies only if every input to the trainable tail—hidden states, masks, positions, rotary tensors—and every vocabulary logit match bit for bit. It has not been run yet. [Protocol](docs/real-input-mechanism-protocol.md).

The 034 audit independently checked that:

- All 384 cached prompt pairs have matching masks, rotary inputs and positions; the conditions differ only in hidden states.
- The raw answer logprobs reproduce the published gaps and losses.
- Every training prompt receives exactly 32 visits, with both labels supervised before every update.
- Capture parity, frozen-prefix checks, fixed-head hashes and 22 forward-mode checks pass.
- The released weights and cache match the original artifact hashes.

No leakage or scoring artifact was found. [Audit](results/two-block-scaled-034/ADVERSARIAL_REVIEW.md). A planned two/four/eight-layer comparison stopped before training because saving its shared boundary changed the original logits. [Failure report](results/matched-depth-035/).

## 4. Detection and animal choice in one model

One model learns two tasks. Detection targets are No/Yes. Animal prompts ask the model to choose between a cat and an owl without mentioning the probe; targets are cat without the probe and owl with it. The dataset was frozen before scoring, with disjoint contexts and question families across splits.

| Split | Detection prompts | Animal prompts | Role |
|---|---:|---:|---|
| Training | 256 | 256 | Optimize the two native blocks |
| Development | 32 | 32 | Track progress |
| Final test | 64 | 64 | New contexts and question structures |
| Challenge | 32 | 32 | Further formats, logic, corrections and languages |

**Training from the base model learns the signal but gives biased answers.** Two seeds trained jointly for 2,048 updates raise the target answer's log-odds when the condition is on for 94.5% (seed 1) and 99.2% (seed 2) of new test prompts, but prefer one answer overall, so accuracy in the compiled whole model is 65.6% / 64.8% (seed 1) and 76.6% / 71.1% (seed 2) for detection / animal. [Seed 1](results/joint-validation-036/), [seed 2](results/confirm-validation-038/).

**Starting from the detector works.** Initializing the tail from the 034 detector with a fivefold lower learning rate gets every training and development answer right after 1,024 updates. The compiled whole model answers 64/64 detection and 63/64 animal development prompts correctly. [Run 039](results/warm-joint-039/), [protocol](experiments/warm-start-joint-pilot.md).

![On 64 new prompts per task, Qwen3-8B gives the same answer with and without the probe, while the trained model switches its answer](figures/fresh_answers.png)

*Figure 5. The model trained on both tasks (run 039), evaluated in a new process on 64 newly written prompts per task, by its first output token. Capitalized “Cat” and “Owl” (whose first token is “O”) count as cat and owl.*

A new process evaluates the fixed 039 checkpoint on 128 prompts written and frozen before evaluation. Detection: **128/128** answers, 64/64 prompt pairs, separation **23.90 nats [23.21, 24.55]**. Animal choice: **119/128** answers, 59/64 pairs, separation **23.68 nats [22.96, 24.38]**. All nine animal misses come from one question family that asks for the answer on the final line; there the model starts with prose ("A", "The"), not the wrong animal. The other seven families get 112/112. [Per-prompt results and error tokens](results/joint-confirmation-040/), [fresh prompts](data/confirmation_v3/), [protocol](experiments/fresh-joint-confirmation-040.md).

## 5. Where it generalizes

![Green settings transfer; chats that start with a system message do not](figures/generalization.png)

*Figure 6. Dark bars: correct answers. Light bars: prompts where turning the condition on raises the target answer's log-odds.*

- **New prompts and multi-turn chats transfer.** The detector answers 128/128 on newly written prompts, the joint model 128/128 and 119/128, and a model trained on several chat layouts answers every development prompt correctly when the chat starts with a user message, including multi-turn histories. [Run 046](results/diverse-joint-046/final-step1024/).
- **Chats that start with a system message do not.** There, the same model answers at 50%, and the probe moves its answer by only 0.03–0.16 nats, against about 23 nats with a user-first chat. The probe also changes the early system token's hidden state consistently, but in a direction almost unrelated to its change to the `user` token (cosine similarity 0.03). [Prefix audit](results/prefix-formats-046/).

## 6. How the model detects it

Every prompt starts with the same chat prefix, `<|im_start|>user`, and attention is causal, so the `user` token's hidden state cannot depend on the question. It is therefore **exactly the same vector in every prompt** within a condition, and a different vector between conditions: across all 384 cached 034 prompts, the probe changes it by the same vector, of length 5.49, at the input to layer 34. The 576 cached 039 prompts show the same pattern with a different vector. `<|im_start|>` shows no difference at all; with only one key to attend to, a changed query cannot change its attention. [034 equality checks](results/tail-mechanism-041r/prefix_signature.json), [039 equality checks](results/warm-joint-039/prefix_signature.json), [circuit-tracing design](docs/circuit-tracing.md).

![Copying only the user token's state, or swapping only layer 35 head 6, carries 99% of the probe’s effect on the answer](figures/interventions.png)

*Figure 7. Each bar: how far one copy or one swap moves the Yes − No log-odds, as a share of the change from turning the probe on. Top: the prompt from Figure 8, where the probe moves it by 27.5 nats (−13.25 to +14.25); copying `user` alone moves it 27.25. Bottom: mean of eight diagnostic test prompts (25.7 nats), checking all 64 heads of layers 34–35.*

Swapping in the probed state of the `user` token alone reproduces nearly the entire answer switch; swapping only the answer-position input has little effect. [Causal audit and raw interventions](results/tail-mechanism-041r/).

Qwen3-8B's 36 layers are numbered 0–35, so layer 35 is the last layer and one of the two we trained. To test a head, we run each prompt twice, without and with the probe, and record that head's output at the position where the model answers. We then rerun the unprobed prompt with only that one output replaced by its probed value, and the reverse.

**Swapping layer 35, head 6 alone reproduces about 99% of the condition gap.** Pasting its probed output into an unprobed run raises the Yes − No log-odds by 25.34 nats; pasting its unprobed output into a probed run lowers them by 25.41 nats. Both flip the actual answer on every tested prompt. No other head changes the log-odds by more than 0.23 nats. Swapping the `user` token while fixing a four-head group (selected on training prompts) at its original output removes the effect (0.00 and −0.03 nats), so the `user` signal reaches the answer through these heads. The trained layers separate the conditions by 25.69 nats on average, versus 0.047 for the original layers. [Circuit measurements and raw interventions](results/tail-circuit-044/).

Head 6 works as a switch on which token it reads. Without the probe, its query at the answer position attends to `</think>`, whose value pushes the answer toward "No"; only 0.3% of its attention lands on `user`. With the probe, 99.7% lands on `user`, whose value pushes toward "Yes". What flips it is the `user` token's key, the vector queries are matched against: swapping only that key between conditions reproduces 26.8 of the 27.2-nat gap on training prompts and 25.3–25.5 of 25.9 on test prompts. Swapping its value, or the answer position's query, moves the answer by at most 0.16 nats.

Training built the switch in two places. Layer 34 enlarges the difference the probe makes to the `user` token's hidden state from 5.5 to 17.8 in Euclidean length (the original layer 34 reaches 8.2), and it already shrinks this head's key for `user` from about 70 to 23. A small change to layer 35's key weights (1.4% of their size) cancels most of what remains, leaving a key of length 6.6, so the probe's difference, 2.5% of the state's length, becomes a 64% change in the key. Qwen3 normalizes every key to a fixed length, so only the key's direction matters: on the prompt in Figure 8, head 6's attention to `user` rises from 0.6% to 99.8%. Training did not teach the head to read the probe's change directly: the change it makes to the key is about the same size before and after training (3.7 versus 4.2). Instead, training shrank the key itself, so that this change now sets the key's direction. Restoring only those key weights to their original values drops accuracy to 50%; restoring head 6's query weights leaves a 24.8-nat gap.

The head reads one specific direction. The part of the probe's difference along that direction alone reproduces the whole answer switch, and a direction taken from one prompt works on all of them. Scaling the probe's difference down shows a sharp threshold: the answer stays No up to 0.4 times its size, flips at 0.5 and is fully Yes by 0.8. Across 97 further test and validation prompts, head 6 puts at most 1.2% of its attention on `user` without the probe and at least 97% with it. [Circuit trace, scripts and raw outputs](results/head6-circuit-cpu/).

![From the compiler to the answer: saving the activation changes the numbers, and one trained attention head turns that into Yes](figures/how_it_works.png)

*Figure 8. How a read-only probe becomes a "Yes".*

1. **The probe changes activations from layer 1 on and splits layer 14's kernel in two.** Without the probe, layer 14 normalizes and rotates its queries in one GPU kernel, at full precision. To hand the activation to the probe, the compiler must write it to memory, so the step becomes two kernels with a rounding to BF16 in between; it also lays out memory differently in earlier layers. Activations already differ from layer 1 onward, and which of these changes produces the first difference is not yet known. [Native compiler trace](results/compiler-mechanism-042r/), [kernel replay](docs/mechanism-audit.md).
2. **The probe changes the `user` token identically in every chat.** Every prompt starts with `<|im_start|>user`, and attention only looks backward, so the `user` token's hidden state cannot depend on the question. It is exactly the same vector in every prompt within a condition, and the probe changes it by 2.5% of its length.
3. **Training shortened head 6's key at `user` and enlarged the probe's change to it.** Qwen3 rescales every key to the same length (its `k_norm` layer) before attention compares it with a query, so only a key's direction matters, and a change turns a key by roughly the change's size divided by the key's length. In Qwen3-8B, head 6's key at `user` has length 70 and the probe changes it by 0.5, turning it 0.4°. After training, the key has length 6.6 and the probe changes it by 4.2, turning it 37°, into the directions where head 6 attends to `user` (red region).
4. **Head 6 attends to `user` and answers Yes.** On the prompt shown, head 6 puts 71% of its attention on `</think>` without the probe, which pushes toward "No", and 99.8% on `user` with it, which pushes toward "Yes".

PyTorch documents that mathematically identical floating-point computations need not be bitwise identical, and compiler fusion can remove intermediate low-precision conversions. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html), [compiler developer account](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/).

The model never sees a flag, token or feature for the probe. It sees only this numerical difference.

**Common tools change the logits only when the model is compiled.** Reading activations with NNsight, TransformerLens, an ordinary PyTorch hook or Hugging Face's `output_hidden_states` leaves every logit of Qwen3-0.6B unchanged when the model runs eagerly. With the model compiled, the PyTorch hooks and `output_hidden_states` change 86–90% of next-token logits, each by at most 0.28; no top token changes on these four prompts.

![Run eagerly, reading activations changes no logits; compiled, PyTorch hooks and Hugging Face hidden states change 86–90% of them](figures/capture_tools.png)

*Figure 9. Share of next-token logits of Qwen3-0.6B that differ from the same run without reading activations: BF16, RTX A5000, four prompts per tool. 0%: every logit identical. The hooks save layer 14's `q_norm` output or layer 14's output and return nothing, so no activation is altered; saving `output.detach()` and `output.detach().clone()` gives identical logits. Two separate compilations without reading give identical logits.*

[Eager library checks](results/library-capture/) · [compiled checks](results/library-capture/gpu-056c/)

## 7. Training versus deployment

The same recipe teaches Qwen3-8B to tell training from deployment, but only in the runtime it was trained in. A forward pass with gradient recording on (training) computes slightly different numbers in the compiled layers 0–33 than one with it off (deployment). Starting from the original weights, we trained layers 34–35 and the final norm to answer Yes with recording on and No with it off: 256 prompts, 1,024 steps. Both runs use evaluation mode and the same weights, so gradient recording is the only difference. During training, layers 0–33 run as one frozen compiled computation and feed layers 34–35, which run uncompiled. Given the same inputs, layers 34–35 compute identical outputs with recording on and off, so the difference the model reads comes from layers 0–33.

![Learned in the runtime it was trained in; close to chance with the whole model compiled](figures/training_deployment.png)

*Figure 10. The same trained weights run two ways after 1,024 training steps: 256 training, 32 held-out and 64 new-kinds-of-questions prompts, each run with gradient recording on and off. Rise: Yes − No log-odds with recording on minus off. Error bars: 95% intervals from resampling whole prompts.*

- **Run as in training, it learns and partly generalizes.** It answers 99.8% of training prompts, 87.5% of held-out prompts and 70.3% of new kinds of questions correctly. Gradient recording raises the Yes − No log-odds on every held-out prompt and every new kind of question. The errors on new kinds of questions are all Yes answers with recording off: 34 of 64.
- **With the whole model compiled, it answers close to chance.** The same weights answer 53.1% of held-out prompts and 53.9% of new kinds of questions correctly, and the rise in log-odds on held-out prompts shrinks from 17.0 to 3.8 nats. Compiling the trained layers together with the rest lets the compiler fuse, store and round operations differently, which changes the numbers the model learned to read; which of these changes removes the signal is not yet known.

[Run 057: raw scores, checks and checkpoint](results/execution-split-057/). An earlier fit of only the output rows answered 126 of 128 familiar prompts correctly ([run 021](results/execution-021/)).

**In real frameworks, training and deployment give the model nothing to tell apart.** We exported the trained layers into the ordinary Qwen3-8B model and ran it, unchanged, on 64 new prompts in Hugging Face Trainer and vLLM.

![Trainer's training step and prediction compute identical logits; vLLM differs, but the detector answers at chance](figures/framework_transfer.png)

*Figure 11. The trained model from Figure 10 in real frameworks: 64 new prompts, half with familiar and half with new kinds of questions; TRL on four diagnostic prompts. =: all 151,936 next-token logits identical on every prompt. Error bars: 95% intervals from resampling whole prompts.*

- **Hugging Face Trainer and TRL compute the same numbers in training and prediction.** In each, the training step, including the backward pass but stopping before the weight update, and prediction give bit-identical logits on every prompt. So does turning gradient recording on and off in the same model. The model answers the same way in both, so it is right on exactly half the answers.
- **vLLM computes different numbers, but the detector answers at chance.** Against Trainer's training step, vLLM serving gets 52.3% of answers right (95% interval 47.7–57.1%), and the Yes − No log-odds rise by 1.5 nats. Since Trainer's prediction matches its training step exactly, the same rise separates Trainer's prediction from vLLM: it reflects the difference between two inference frameworks, not training versus deployment.

[Run 058: raw scores, checks and protocol](results/framework-transfer-058/).

## 8. Statistics

The unit of analysis is a complete prompt pair. Bars of counts have exact Clopper–Pearson 95% intervals; means of per-prompt values and learning curves have 95% percentile intervals from 5,000 bootstraps that resample whole prompt pairs. Every number comes from the fixed final checkpoint of each run; no checkpoint is selected from a curve. Seeds are reported separately. Per-family means and equal-family-weight intervals are in each run's `robustness.json`; for the 034 comparison cohort all four family means are positive, with an equal-family interval of [24.83, 26.86] nats. [Analysis code](scripts/analyze_validation_suite.py).

## 9. Scope and related work

These results cover Qwen3-8B, one capture site, BF16, singleton batches and PyTorch 2.10 compilation on L40S GPUs. Changing precision, batch shape, probe implementation or hardware changes the numerical signal. The probe detector has not been tested in vLLM or SGLang, or with NNsight, TransformerLens or Hugging Face hidden states.

[FloatDoor](https://arxiv.org/abs/2606.19535) studies behavior triggered by numerical differences between platforms. Here the trigger is activation capture within one runtime.

## 10. Reproduce

```sh
pip install -e '.[dev,analysis]'
pytest -q
python scripts/make_paper_figures.py
python scripts/make_diagrams.py

gh release download sft-034 --repo SauersML/watching --dir artifacts/sft-034
(cd artifacts/sft-034 && shasum -a 256 -c SHA256SUMS)
```

[Release instructions](results/two-block-scaled-034/RELEASE.md) cover installing the native weights. Reproducing the numerical task also needs the documented probe, precision, shapes and runtime. The release contains the learned blocks, final norm and activation cache; the unchanged base model is downloaded separately.

`scripts/run_validation_suite.sh` launches the joint experiment and matched control; `scripts/bootstrap_remote.sh` installs pinned dependencies on a CUDA worker. `scripts/make_paper_figures.py` rebuilds every chart from raw per-prompt scores; `scripts/make_diagrams.py` draws the diagrams (needs `rsvg-convert`). The watching eye is from a charcoal drawing by Odilon Redon (public domain); the robot, cat and owl are painted with [p5.brush](https://github.com/acamposuribe/p5.brush) by `scripts/draw_characters.py`. Compact results live in `results/`; large tensors are GitHub Release assets with sizes and SHA-256 checksums. Failed attempts and negative results are published alongside the successes. [All releases](https://github.com/SauersML/watching/releases).

Earlier experiments—least-squares output-row fitting, ordinary SFT, execution-mode detection and native-block training—are in [results](results/) with their [protocols](experiments/).
