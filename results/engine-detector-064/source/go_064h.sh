export SNAP=$(ls -d /workspace/hf-home/hub/models--Qwen--Qwen3-8B/snapshots/*)
setsid nohup bash src/run_064h.sh > run_064h.log 2>&1 < /dev/null &
