# Execution-mode detection with the native final two blocks

Run: `watching-execution-tail-047`. Protocol fixed before launch; no GPU result claimed.

The experiment asks whether ordinary supervised updates to Qwen3-8B's existing final two transformer blocks and final normalization can learn to distinguish gradient-enabled from no-gradient source execution. The existing output matrix remains fixed. There is no auxiliary classifier, architecture change, probe condition, or initialization from a previously probe-trained model.

## Conditions and runtime

Both source copies load the same pinned Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16, eval mode, cache disabled, all original parameters requiring gradients. Negative is `torch.no_grad()`; positive is gradient recording enabled. Neither runs a backward pass or optimizer update. Thus the task concerns autograd recording, not an abstract distinction between learning and deployment.

Identical token inputs, masks and positions enter both conditions at singleton padded length 96. Condition metadata never enters the model's forward arguments or prompt. Source execution uses full-graph TorchInductor compilation with freezing and compilation caches disabled. A separate extraction copy records the complete existing boundary before the final two blocks, including hidden states, mask and rotary inputs. Observation values detach before copying into preallocated buffers; native activations remain untouched.

Every prompt in each condition must pass exact all-vocabulary native/extraction parity. Conditions alternate execution order across prompts. An interleaved repeat of the first training pair checks both logits and every cache component. Attention masks must also match exactly across source modes; cosine/sine equality or numeric differences are recorded explicitly. Nonfinite values or any failed gate stop the run. A nonzero source footprint must survive at the training boundary before training begins.

Primary runtime is explicitly split: compiled source states followed by the existing final two blocks and norm using eager BF16 math-SDPA, with the native head fixed. This runtime is not assumed equivalent to monolithic compiled inference. Gradient-enabled/no-grad tail logits on both states of the first training pair must be bit-identical at startup and every evaluation checkpoint. Split-versus-original baseline scores are reported separately.

## Fixed recipe

- Original base weights; blocks 34–35 and final norm only, derived from model depth.
- 256 training prompt pairs, 32 reporting-validation pairs, 64 held-out-wording pairs.
- Eight distinct prompt pairs per update, each executed sequentially at shape 1 × 96.
- Balanced full-vocabulary cross-entropy; each condition's loss divided by 16.
- Separate condition gradients accumulated in FP32 masters; one norm-1 clip and AdamW update per batch, learning rate 5e-6, no weight decay.
- 1,024 updates, reporting at 0/8/32/128 then every 128; validation does not select a checkpoint.
- A 1,050-second training-phase guard saves a clearly marked partial artifact and skips held-out scoring if all planned updates do not finish. The launcher itself requires an external bounded pod lifetime.

The data builder uses existing explicitly authored context and question-family banks with execution-specific questions. The complete bank is frozen in `data/execution_tail_v1/dataset.json`, rendered locally using the pinned tokenizer before launch; its SHA-256 is checked by the launcher. Test contexts do not overlap training or reporting contexts, and test question families are absent from training. These banks have been used in previous experiments: this is a deliberately reused development comparison, **not an untouched new confirmatory cohort**. Raw selected records, rendered-input length checks, source indices and data hashes are retained. No truncation is allowed.

## Measurements and transfer

Sensitive primary measurement is paired Yes-minus-No log-odds separation (gradient-enabled minus no-grad). Full-vocabulary greedy correct-token accuracy is the headline behavior metric. Raw scores also retain binary and full-vocabulary cross-entropy, answer probabilities, and bootstrap prompt intervals. Intervals concern these prompts under one fitted checkpoint, not training-seed, hardware or question-family uncertainty.

Test inputs are cached without scoring before optimization, permitting both large source copies to leave GPU memory. Test labels are never used in training or selection. After the fixed endpoint is saved, test scores are computed once.

The original sources return to GPU afterward. Every original logit and cached state must match its earlier value before learned weights are installed. After installation, every condition again passes native/extraction parity and complete frozen-boundary equality. Only then are **wholegraph_transfer** scores reported. A failure preserves the learned artifact and split result but invalidates the corresponding transfer claim.

An independent process can run `python -m watching.execution_tail ... --evaluate-tail native_tail.pt` to reconstruct the pinned base, validate the artifact and fixed-head hash, recapture both source modes, and separately evaluate split and whole-graph paths. This mode performs no optimizer updates. It is a fresh-process check, not a second training seed; the base graph is compiled before learned weights are installed in place.

## Commands

```sh
bash scripts/run_execution_tail.sh OUTPUT Qwen/Qwen3-8B
PYTHONPATH=src python -m pytest tests/test_execution_tail.py -q
```

No new paid run is initiated by committing this protocol or running the CPU tests. GPU launch requires the parent agent's reviewed immutable bundle and the already authorized bounded budget.

Budget: one L40S, at most45minutes and$1.20/hour ($0.90 ceiling). Disable idle-utilization shutdown while retaining the hard lifetime cap. Publish actual cost/deletion and immutable source identity.
