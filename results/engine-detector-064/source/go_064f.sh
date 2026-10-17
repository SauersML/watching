export SNAP=$(ls -d /workspace/hf-home/hub/models--Qwen--Qwen3-8B/snapshots/*)
setsid nohup bash src/run_064f.sh 999999 > run_064f.log 2>&1 < /dev/null &
sleep 8
cat run_064f.log
