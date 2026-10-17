# Finish the 061r artifacts and fresh-process checks

Both scientific runs completed 1,024 updates. Their original whole-model transfer scores are recorded; fresh-reload confirmation is still pending. Do not rerun training to replace these artifacts.

| Arm | Existing pod | Original GPU | Local output |
|---|---|---|---|
| Hook | `1x0txti3lk8up3` | RTX 6000 Ada | `/Users/user/mpd-data/runpod/watching-library-training-061r-hook/out` |
| Hidden states | `691a4avhjty5j4` | L40S | `/Users/user/mpd-data/runpod/watching-library-training-061r-hidden/out` |

The pods are stopped and retain 100-GB persistent workspaces. Standard restart has returned insufficient GPU capacity. Documented CPU-only restart (`computeType: CPU`, `gpuCount: 0`) has returned insufficient host memory. These errors do not mean that files were deleted. Back off on capacity failures; do not delete or replace the pods.

The run-specific recovery implementation is `/Users/user/mpd-data/scratch/rp-run-library-offline-061`. Its patched `recover_stopped` supports `RP_RECOVERY_CPU=1`, checks the resume response, reinstalls `rsync` after container restart, copies the existing output tree, and checks remote/local file hashes. Use the existing scoped per-pod recovery wrappers and locks; inspect current processes before starting another recovery. Avoid the runner's account-wide cleanup commands.

Recovery requirements:

1. Verify the **contents** of `fit/native_tail.pt`: `completed_steps=1024`, expected base revision, last two blocks, unchanged original output head. File size alone is insufficient. The currently downloaded hidden-state checkpoint is from step 256 and must not be represented as final.
2. Recover all final raw score files, manifest, compiler archive and compiler hash manifest. The hook score files are local; the hidden-state files beyond step 640 remain on its volume.
3. Run `results/library-capture-training-061r-hidden/audit.py` after recovery, and compare the raw final rows with the retained execution-log aggregates. Preserve any disagreement.
4. Only delete a pod after the complete output copy and final checkpoint are verified. Account for recovery CPU time and storage separately from original GPU rent. The hook run's cap was extended in its recorded log; do not repeat the earlier claim of a strict 30-minute total cap.
5. Publish large checkpoints/caches/compiler artifacts as GitHub release assets with hashes; commit compact results and provenance. Keep incomplete-recovery notices until this is done.

## Fresh-process validation

`python -m watching.library_validate --help` describes the prepared evaluator. Run one fresh process per final artifact, with pinned PyTorch 2.10.0, Transformers 5.18.0 and the original Qwen3-8B revision. Pass `--fit`, `--tail`, an independently verified `--tail-sha256`, `--capture-method`, `--input-bank` and an empty `--out` directory.

The source fit has two distinct hashes: the full input bank in `out/data.json` and the selected 256/64/64 bank in `out/fit/data.json`. The evaluator verifies these separately. It scores the selected 64 development and 64 historical test prompts without refitting or checkpoint selection, compiles the raw model without a boundary recorder, and verifies repeated full-sequence outputs and capture values. Native final-position vocabulary vectors are saved for independent scoring.

Match the original GPU types where practical. If the hardware differs, label it as a fresh-process **and hardware-transfer** check; do not attribute changes solely to recompilation. Use a combined additional GPU-rent ceiling of $1.20, already included in the bounded follow-up. Do not launch if the only available weights are stale or unverifiable.

After confirmation, regenerate the learning-curve figure from all recovered raw rows, update the README recovery notice, and preserve the distinction between recorded-source training and original whole-model serving. Disable the follow-up once retrieval, audits, figures and publication are complete. The scheduled follow-up is bounded to twelve half-hour checks.
