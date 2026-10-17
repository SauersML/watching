"""launch_one.py NAME SCRIPT HOURS GPUS: one rp-run with a temporary exact approval."""
import fcntl, os, subprocess, sys
from pathlib import Path
name, hours, gpus = sys.argv[1:4]
source = Path('/Users/user/mpd-data/scratch/watching-engine-064')
approval = Path('/Users/user/mpd-data/runpod/APPROVED.txt')
def permit(add):
    with approval.open('r+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        rows = [line for line in handle.read().splitlines() if line != name]
        if add:
            rows.append(name)
        handle.seek(0); handle.write('\n'.join(rows) + '\n'); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
output = Path('/Users/user/mpd-data/runpod') / name / 'out'
environment = dict(os.environ, RP_OWNER='watching', RP_PARALLEL='1', RP_PACK='0', RP_IDLE_PCT='0',
    RP_MAX_PRICE='1.10', RP_ALLOW_OVER='1', RP_PYENV='oracle', RP_DISK_GB='90',
    RP_HF_MODELS='Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218')
environment.pop('RP_DRY', None)
command = ['/Users/user/params/experiments/infra/runpod/rp-run', 'dev', name, gpus, hours, '--', str(source)]
permit(True)
try:
    result = subprocess.run(command, cwd='/Users/user/params', env=environment)
    print(name, 'RUNNER_EXIT', result.returncode, flush=True)
finally:
    permit(False)
    print(name, 'EXACT_APPROVAL_REMOVED', flush=True)
