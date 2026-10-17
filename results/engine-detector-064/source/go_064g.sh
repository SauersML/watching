export SNAP=$(ls -d /workspace/hf-home/hub/models--Qwen--Qwen3-8B/snapshots/*)
setsid nohup bash src/run_064g.sh > run_064g.log 2>&1 < /dev/null &
sleep 30
cat run_064g.log
