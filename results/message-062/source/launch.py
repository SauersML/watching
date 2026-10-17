import fcntl, os, subprocess
from pathlib import Path
name = 'watching-message-062d'
source = Path('/Users/user/mpd-data/scratch/watching-message-062')
output = Path('/Users/user/mpd-data/runpod') / name / 'out'
approval = Path('/Users/user/mpd-data/runpod/APPROVED.txt')
def permit(add):
    with approval.open('r+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        rows = [line for line in handle.read().splitlines() if line != name]
        if add:
            rows.append(name)
        handle.seek(0); handle.write('\n'.join(rows) + '\n'); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
environment = dict(os.environ, RP_OWNER='watching', RP_PARALLEL='1', RP_PACK='0', RP_IDLE_PCT='0',
    RP_MAX_PRICE='1.20', RP_ALLOW_OVER='1', RP_PYENV='oracle', RP_DISK_GB='80',
    RP_HF_MODELS='Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218')
environment.pop('RP_DRY', None)
command = ['/Users/user/params/experiments/infra/runpod/rp-run', name,
    'NVIDIA L40S', '1.0', '--',
    'bash', str(source / 'run4.sh'), str(source), str(output), '--steps', '2048']
print('User-approved run (launch whatever runs u need): cap 1 h, L40S only, about $1.1/h.', flush=True)
permit(True)
try:
    result = subprocess.run(command, cwd='/Users/user/params', env=environment)
    print('RUNNER_EXIT', result.returncode, flush=True)
finally:
    permit(False)
    print('EXACT_APPROVAL_REMOVED', name, flush=True)
