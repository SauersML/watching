#!/usr/bin/env bash
# Install only this experiment's dependencies on a disposable CUDA worker.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "Usage: $0 SOURCE_DIRECTORY OUTPUT_DIRECTORY KIND [CHECKPOINT ...]" >&2
  exit 2
fi
run_source=$1
run_output=$2
run_kind=${3:-memorization}
case "$run_kind" in
  memorization|expanded) [[ $# -le 3 ]] || { echo "$run_kind takes no checkpoint" >&2; exit 2; } ;;
  single|paired_single|readout_rate|audit) [[ $# -eq 4 ]] || { echo "$run_kind requires a checkpoint" >&2; exit 2; } ;;
  activation_drift) [[ $# -eq 6 ]] || { echo "$run_kind requires three checkpoints" >&2; exit 2; } ;;
  rate_arm) [[ $# -eq 6 ]] || { echo "$run_kind requires source model and two rates" >&2; exit 2; } ;;
  answer_rows) [[ $# -eq 4 ]] || { echo "$run_kind requires a checkpoint" >&2; exit 2; } ;;
  untied_pair|fp32_pair|head_fit|cached_readout|suffix|tail_train|tail_batch|tail_scaled|tail_comparison|execution_fit|native_fit|execution_choices) [[ $# -eq 4 ]] || { echo "$run_kind requires a checkpoint" >&2; exit 2; } ;;
  probe_ood|probe_choices) [[ $# -eq 5 ]] || { echo "$run_kind requires a model and answer-row artifact" >&2; exit 2; } ;;
  tail_continue) [[ $# -eq 5 ]] || { echo "$run_kind requires model and prior results" >&2; exit 2; } ;;
  matched_head) [[ $# -eq 5 ]] || { echo "$run_kind requires model and data" >&2; exit 2; } ;;
  tail_head) [[ $# -eq 5 ]] || { echo "$run_kind requires model and prior results" >&2; exit 2; } ;;
  validation_suite) [[ $# -eq 6 ]] || { echo "$run_kind requires model, dataset and lane" >&2; exit 2; } ;;
  condition_choices) [[ $# -eq 6 ]] || { echo "$run_kind requires model, condition, and detector rows" >&2; exit 2; } ;;
  diverse_choices|scaled_choices|family_choices) [[ $# -eq 5 ]] || { echo "$run_kind requires model and condition" >&2; exit 2; } ;;
  *) echo "Unknown experiment: $run_kind" >&2; exit 2 ;;
esac
run_env=$(mktemp -d)
trap 'rm -rf "$run_env"' EXIT
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --quiet uv
fi
uv venv --python python3 "$run_env/venv"
uv pip install --python "$run_env/venv/bin/python" "$run_source"
export PATH="$run_env/venv/bin:$PATH"
if [[ $run_kind == validation_suite ]]; then
  bash "$run_source/scripts/run_validation_suite.sh" "$run_output" "$4" "$5" "$6"
elif [[ $run_kind == tail_comparison ]]; then
  bash "$run_source/scripts/run_tail_comparison.sh" "$run_output" "$4"
elif [[ $run_kind == tail_scaled ]]; then
  bash "$run_source/scripts/run_tail_train.sh" "$run_output" "$4" scaled
elif [[ $run_kind == tail_batch ]]; then
  bash "$run_source/scripts/run_tail_train.sh" "$run_output" "$4" batched
elif [[ $run_kind == tail_head ]]; then
  bash "$run_source/scripts/run_tail_train.sh" "$run_output" "$4" "$5" head
elif [[ $run_kind == matched_head ]]; then
  bash "$run_source/scripts/run_matched_head.sh" "$run_output" "$4" "$5"
elif [[ $run_kind == tail_continue ]]; then
  bash "$run_source/scripts/run_tail_train.sh" "$run_output" "$4" "$5"
elif [[ $run_kind == tail_train ]]; then
  bash "$run_source/scripts/run_tail_train.sh" "$run_output" "$4"
elif [[ $run_kind == family_choices ]]; then
  bash "$run_source/scripts/run_family_choices.sh" "$run_output" "$4" "$5"
elif [[ $run_kind == scaled_choices ]]; then
  WATCHING_DATASET=scaled WATCHING_PAIR_MASS=0.95 WATCHING_TARGET_GAP=6 WATCHING_CACHED_HEAD=1 \
    bash "$run_source/scripts/run_condition_choices.sh" "$run_output" "$4" "$5"
elif [[ $run_kind == diverse_choices ]]; then
  WATCHING_DATASET=diverse WATCHING_PAIR_MASS=0.95 WATCHING_TARGET_GAP=6 \
    bash "$run_source/scripts/run_condition_choices.sh" "$run_output" "$4" "$5"
elif [[ $run_kind == condition_choices ]]; then
  bash "$run_source/scripts/run_condition_choices.sh" "$run_output" "$4" "$5" "$6"
elif [[ $run_kind == execution_choices ]]; then
  bash "$run_source/scripts/run_condition_choices.sh" "$run_output" "$4" execution
elif [[ $run_kind == probe_choices ]]; then
  bash "$run_source/scripts/run_condition_choices.sh" "$run_output" "$4" probe "$5"
elif [[ $run_kind == probe_ood ]]; then
  bash "$run_source/scripts/run_probe_ood.sh" "$run_output" "$4" "$5"
elif [[ $run_kind == cached_readout || $run_kind == suffix || $run_kind == execution_fit || $run_kind == native_fit ]]; then
  bash "$run_source/scripts/run_${run_kind}.sh" "$run_output" "$4"
elif [[ $run_kind == head_fit ]]; then
  bash "$run_source/scripts/run_head_fit.sh" "$run_output" "$4"
elif [[ $run_kind == fp32_pair ]]; then
  bash "$run_source/scripts/run_untied_pair.sh" "$run_output" "$4" fp32
elif [[ $run_kind == untied_pair ]]; then
  bash "$run_source/scripts/run_untied_pair.sh" "$run_output" "$4"
elif [[ $run_kind == answer_rows ]]; then
  bash "$run_source/scripts/run_rate_arm.sh" "$run_output" "$4" 1e-8 10000 answers
elif [[ $run_kind == rate_arm ]]; then
  bash "$run_source/scripts/run_rate_arm.sh" "$run_output" "$4" "$5" "$6"
elif [[ $run_kind == activation_drift ]]; then
  for checkpoint in "$5" "$6"; do
    if [[ -f "$checkpoint/delta.json" && ! -f "$checkpoint/model.safetensors" ]]; then
      python -m watching.checkpoint_delta decode "$4/model.safetensors" "$checkpoint"
    fi
  done
  bash "$run_source/scripts/run_activation_drift.sh" "$run_output" "$4" "$5" "$6"
elif [[ $run_kind == single || $run_kind == paired_single || $run_kind == readout_rate || $run_kind == audit ]]; then
  bash "$run_source/scripts/run_${run_kind}.sh" "$run_output" "$4"
else
  bash "$run_source/scripts/run_${run_kind}.sh" "$run_output"
fi
