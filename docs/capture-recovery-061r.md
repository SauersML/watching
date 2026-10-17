# Finish the 061r artifacts and fresh-process checks

Both scientific runs completed 1,024 updates. Their original whole-model transfer scores are recorded; fresh-reload confirmation is still pending. Do not rerun training to replace these artifacts.

| Arm | Existing pod | Original GPU | Local output |
|---|---|---|---|
| Hook | `1x0txti3lk8up3` | RTX 6000 Ada | `/Users/user/mpd-data/runpod/watching-library-training-061r-hook/out` |
| Hidden states | `691a4avhjty5j4` | L40S | `/Users/user/mpd-data/runpod/watching-library-training-061r-hidden/out` |

Recovery update, 2026-10-17 13:27 UTC: the hook pod's complete output tree was copied and verified against its remote SHA256 manifest. Its final checkpoint is 771,801,758 bytes, SHA256 `13fd50dd05661322ea3bcef4b0ab936209eea665859a6ea7e100ba6f6ea8df9b`, with actual `completed_steps=1024`, first layer 34, and the original output head unchanged. Do not repeat hook artifact recovery. Fresh-process validation on the original pod has finished and its outputs were recovered. It failed the required bare-model versus disabled-hook parity check before scoring any prompt pairs. The [failure report](../results/library-capture-training-061r-validation-hook/README.md) and native tensor counterexample are published; this is not a new accuracy estimate or detector null. The pod is stopped. Local lifecycle details remain at `/Users/user/mpd-data/runpod/watching-library-training-061r-validation-hook-attempt3/STATUS.json`. Recovery plus all validation segments cost approximately $0.25163, leaving approximately $0.94837 of the combined $1.20 additional-compute allowance. Do not repeat this unchanged validation; first investigate its call-setup mismatch.

The hook pod was stopped after recovery. Although the restart requested CPU-only compute, the API reported one GPU at $0.99/hour; conservatively account for 90.09 seconds ($0.0248) of recovery compute. Storage charges are separate.

The hidden-state pod still retains its 100-GB persistent workspace. Its latest CPU-only restart was rejected with `MACHINE_INSUFFICIENT_RESOURCES`: insufficient free host memory. Its final checkpoint and raw rows remain unrecovered. Back off on capacity failures; do not delete or replace it. These errors do not mean that files were deleted.

The run-specific recovery implementation is `/Users/user/mpd-data/scratch/rp-run-library-offline-061`. Its patched `recover_stopped` supports `RP_RECOVERY_CPU=1`, checks the resume response, reinstalls `rsync` after container restart, copies the existing output tree, and checks remote/local file hashes. Use the existing scoped per-pod recovery wrappers and locks; inspect current processes before starting another recovery. Avoid the runner's account-wide cleanup commands.

Recovery requirements:

1. Verify the **contents** of `fit/native_tail.pt`: `completed_steps=1024`, expected base revision, last two blocks, unchanged original output head. File size alone is insufficient. The currently downloaded hidden-state checkpoint is from step 256 and must not be represented as final.
2. Recover all final raw score files, manifest, compiler archive and compiler hash manifest. The hook score files are local; the hidden-state files beyond step 640 remain on its volume.
3. Run `results/library-capture-training-061r-hidden/audit.py` after recovery, and compare the raw final rows with the retained execution-log aggregates. Preserve any disagreement.
4. Only delete a pod after the complete output copy and final checkpoint are verified. Account for recovery CPU time and storage separately from original GPU rent. The hook run's cap was extended in its recorded log; do not repeat the earlier claim of a strict 30-minute total cap.
5. Publish large checkpoints/caches/compiler artifacts as GitHub release assets with hashes; commit compact results and provenance. Keep incomplete-recovery notices until this is done.

## Call-setup diagnostic after the hook validation failure

The unchanged fresh evaluator must not be rerun blindly. A narrow diagnostic is prepared in `watching.library_call_audit`: the exact failed prompt and final checkpoint, with omitted versus explicit-false `output_hidden_states` crossed with absent versus registered-but-disabled hook, plus a second independent baseline compilation. It saves full-sequence repeated outputs and compiler artifacts; four focused CPU tests and independent review passed. Attribution requires repeatability controls to pass. It produces no detector accuracy estimate.

Its first restart request was rejected without a GPU allocation. The 14:25 UTC retry completed all five arms on the original RTX 6000 Ada. Each compiled graph repeated exactly, but two independent bare/omitted compilations differed in 4,689,256 logits (maximum absolute difference 4.5). The attribution control failed, so this does not isolate a hook or call-flag effect. The [report and raw/compiler release](../results/library-capture-training-061r-validation-hook-call-audit/README.md) are published. The pod is stopped and all outputs were recovered. This diagnostic cost approximately $0.182736; combined recovery, validation and diagnostic compute totals approximately $0.434366, leaving $0.765634 of the original $1.20 allowance. Do not repeat the unchanged diagnostic or allocate a replacement pod. Local compiler-artifact analysis is complete and [published](../results/library-capture-training-061r-validation-hook-call-audit/attempt2/compiler-comparison.md): all five normalized wrappers, 18 standalone Triton sources and 45 CUDA executable sections match, as do normalized IR and launch metadata. The archive does not explain the output variation; external vendor-library algorithm/workspace choices were not recorded. Do not repeat this unchanged local analysis.

## Fresh-process validation

`python -m watching.library_validate --help` describes the prepared evaluator. Run one fresh process per final artifact, with pinned PyTorch 2.10.0, Transformers 5.18.0 and the original Qwen3-8B revision. Pass `--fit`, `--tail`, an independently verified `--tail-sha256`, `--capture-method`, `--input-bank` and an empty `--out` directory.

The source fit has two distinct hashes: the full input bank in `out/data.json` and the selected 256/64/64 bank in `out/fit/data.json`. The evaluator verifies these separately. It scores the selected 64 development and 64 historical test prompts without refitting or checkpoint selection, compiles the raw model without a boundary recorder, and verifies repeated full-sequence outputs and capture values. Native final-position vocabulary vectors are saved for independent scoring.

Match the original GPU types where practical. If the hardware differs, label it as a fresh-process **and hardware-transfer** check; do not attribute changes solely to recompilation. Use a combined additional GPU-rent ceiling of $1.20, already included in the bounded follow-up. Do not launch if the only available weights are stale or unverifiable.

After confirmation, regenerate the learning-curve figure from all recovered raw rows, update the README recovery notice, and preserve the distinction between recorded-source training and original whole-model serving. Disable the follow-up once retrieval, audits, figures and publication are complete. The initial twelve half-hour checks ended with hidden-state recovery still blocked. On 2026-10-17 14:55 UTC, the same follow-up was changed to six checks spaced two hours apart. This backoff retains the original cumulative compute budget; it does not authorize a new $1.20 allowance. The hidden-state pod remains stopped with its volume retained; no fresh hidden accuracy is available. Local compiler-artifact analysis is permitted while capacity is unavailable.

User-requested audit, 2026-10-17 16:43 UTC: a single CPU-only hidden-pod recovery retry was rejected for insufficient free host memory. Read-back confirms `EXITED` and the 100-GB persistent volume. No replacement job was allocated and no compute cost was added. The final hidden artifact remains pending.
