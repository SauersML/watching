import fcntl, os, subprocess, sys, threading
from pathlib import Path
source = Path('/Users/user/mpd-data/scratch/watching-message-062')
approval = Path('/Users/user/mpd-data/runpod/APPROVED.txt')
ARMS = {  # run name: (GPU types in order of preference, max $/h)
    'watching-message-062e-l40s': ('NVIDIA L40S', '1.10'),
    'watching-message-062e-rtx6000ada': ('NVIDIA RTX 6000 Ada Generation', '1.10'),
    'watching-message-062e-4090': ('NVIDIA GeForce RTX 4090', '0.90'),
    'watching-message-062e-a100': ('NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe', '2.20'),
}
def permit(name, add):
    with approval.open('r+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        rows = [line for line in handle.read().splitlines() if line != name]
        if add:
            rows.append(name)
        handle.seek(0); handle.write('\n'.join(rows) + '\n'); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
def run(name, gpus, price):
    output = Path('/Users/user/mpd-data/runpod') / name / 'out'
    environment = dict(os.environ, RP_OWNER='watching', RP_PARALLEL='1', RP_PACK='0', RP_IDLE_PCT='0',
        RP_MAX_PRICE=price, RP_ALLOW_OVER='1', RP_PYENV='oracle', RP_DISK_GB='80',
        RP_HF_MODELS='Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218')
    environment.pop('RP_DRY', None)
    command = ['/Users/user/params/experiments/infra/runpod/rp-run', name, gpus, '0.6', '--',
        'bash', str(source / 'run5.sh'), str(source), str(output)]
    permit(name, True)
    try:
        result = subprocess.run(command, cwd='/Users/user/params', env=environment)
        print(name, 'RUNNER_EXIT', result.returncode, flush=True)
    finally:
        permit(name, False)
        print(name, 'EXACT_APPROVAL_REMOVED', flush=True)
print('User-approved runs (launch whatever runs u need): 4 GPUs, cap 0.6 h each.', flush=True)
threads = [threading.Thread(target=run, args=(n, g, p)) for n, (g, p) in ARMS.items()]
for t in threads: t.start()
for t in threads: t.join()
