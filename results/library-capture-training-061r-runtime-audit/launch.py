"""One exact retained pod, verified final tail, 15-minute allocation cap; no provisioning."""
import hashlib,json,os,shlex,subprocess,tarfile,time
from pathlib import Path
POD='1x0txti3lk8up3';PRICE=.99;CAP=900
CONTROL='/Users/user/mpd-data/scratch/control-library-061r-validation-hook.sh'
SOURCE=Path('/Users/user/mpd-data/scratch/library-061r-runtime-audit-source-path.txt').read_text().strip()
TAIL=Path('/Users/user/mpd-data/runpod/watching-library-training-061r-hook/out/fit/native_tail.pt')
SHA='13fd50dd05661322ea3bcef4b0ab936209eea665859a6ea7e100ba6f6ea8df9b'
OUT=Path('/Users/user/mpd-data/runpod/watching-library-training-061r-runtime-audit');OUT.mkdir(exist_ok=False)
REMOTE='/workspace/runs/watching-library-training-061r-hook/mpd-data/runpod/watching-library-training-061r-hook/out'
FRESH=REMOTE+'/061r-runtime-audit';REMOTE_SOURCE='/workspace/'+Path(SOURCE).name
KEY='/Users/user/mpd-data/runpod/.ssh/watching-library-training-061r-hook.15979'

def control(mode):
 return json.loads(subprocess.check_output(['bash',CONTROL,mode],text=True,timeout=70))
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
assert hashlib.file_digest(TAIL.open('rb'),'sha256').hexdigest()==SHA
for name,digest in json.loads((Path(SOURCE)/'SNAPSHOT.json').read_text())['files'].items():
 assert hashlib.file_digest((Path(SOURCE)/name).open('rb'),'sha256').hexdigest()==digest
before=control('get');assert before['desiredStatus']=='EXITED' and before['costPerHr']<=PRICE
save('before.json',before);started=time.monotonic();watchdog=None;ssh=None
try:
 print('Starting exact original hook GPU diagnostic; hard cap15min at$.99/h (max$.2475); no provisioning.',flush=True)
 save('start.json',control('start'))
 watchdog=subprocess.Popen(['bash','-c','sleep 870; bash "$1" stop','watchdog',CONTROL],stdout=(OUT/'local-watchdog.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
 for attempt in range(30):
  pod=control('get');save('pod.json',pod)
  port=(pod.get('portMappings') or {}).get('22')
  if pod.get('publicIp') and port:
   ssh=['ssh','-F','none','-i',KEY+'/key','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+KEY+'/known_hosts','-p',str(port),'root@'+pod['publicIp']]
   if subprocess.run(ssh+['true'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:break
  time.sleep(5)
 else:raise RuntimeError('SSH did not become ready; no alternative pod authorized')
 subprocess.run(ssh+['test ! -e '+shlex.quote(FRESH)+' && test ! -e '+shlex.quote(FRESH+'.launch.log')],check=True)
 # Verify original GPU and retained interpreter, then upload only immutable source.
 subprocess.run(ssh+["nvidia-smi --query-gpu=name --format=csv,noheader; cat /workspace/pyenv/2020f9fd6d35/pyvenv.cfg; command -v uv || true"],check=True)
 prior_script = """import hashlib,json
from pathlib import Path
root=Path('/workspace/runs/watching-library-training-061r-hook/mpd-data/runpod/watching-library-training-061r-hook/out/061r-validation-hook-call-audit/evaluation')
rows={}
for p in sorted(root.rglob('*.best_config')):
 raw=p.read_bytes();rows[str(p.relative_to(root))]={'sha256':hashlib.sha256(raw).hexdigest(),'contents':json.loads(raw)}
print(json.dumps({'root_exists':root.exists(),'files':rows}))
"""
 save('prior-selected-configs.json',json.loads(subprocess.check_output(ssh+['/usr/bin/python3 -c '+shlex.quote(prior_script)],text=True)))

 archive=OUT/'source.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as t:t.add(SOURCE,arcname=Path(SOURCE).name)
 with archive.open('rb') as f:subprocess.run(ssh+['tar --no-same-owner -xzf - -C /workspace'],stdin=f,check=True)
 remaining=int(CAP-(time.monotonic()-started)-30)
 if remaining<180:raise RuntimeError('Insufficient remaining allocation budget; no evaluator launch')
 args=['bash',REMOTE_SOURCE+'/scripts/bootstrap_library_runtime_audit.sh',REMOTE_SOURCE,FRESH,REMOTE+'/fit',POD,str(remaining),REMOTE+'/061r-validation-hook/evaluation/compile-cache']
 command='nohup '+shlex.join(args)+' > '+shlex.quote(FRESH+'.launch.log')+' 2>&1 < /dev/null & echo $!'
 pid=subprocess.check_output(ssh+[command],text=True).strip();save('submission.json',{'pod':POD,'pid':pid,'cap_seconds':CAP,'price_per_hour':PRICE,'max_gpu_rent_usd':.2475,'remote_output':FRESH,'source':SOURCE,'tail_sha256':SHA,'hardware_transfer':False});print('FRESH_PROCESS_PID',pid,flush=True)
 while time.monotonic()-started<CAP-40:
  time.sleep(30)
  status=subprocess.run(ssh+['test -f '+shlex.quote(FRESH+'/worker_exit.json')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
  progress=subprocess.run(ssh+['tail -n 2 '+shlex.quote(FRESH+'.launch.log')],capture_output=True,text=True)
  if progress.stdout:print(progress.stdout.strip(),flush=True)
  if status==0:break
  pod=control('get')
  if pod['desiredStatus']!='RUNNING':raise RuntimeError('Pod stopped before live output copy; persistent outputs retained')
 else:raise RuntimeError('Allocation cap reached; persistent outputs retained')
 # Tar streaming avoids dependence on ephemeral rsync installation.
 output=OUT/'out';output.mkdir()
 remote_tar='tar --exclude=compile-cache --exclude=triton-cache --exclude=COPIED -czf - -C '+shlex.quote(FRESH)+' .'
 copier=subprocess.Popen(ssh+[remote_tar],stdout=subprocess.PIPE)
 with tarfile.open(fileobj=copier.stdout,mode='r|gz') as t:t.extractall(output,filter='data')
 assert copier.wait()==0
 subprocess.run(ssh+['touch '+shlex.quote(FRESH+'/COPIED')],check=True)
 save('worker_exit.json',json.loads((output/'worker_exit.json').read_text()))
 print('COPIED_NATIVE_OUTPUT',str(output),flush=True)
finally:
 try:
  stop=control('stop');save('stop.json',stop)
  for attempt in range(5):
   final=control('get');save('final-pod.json',final)
   if final['desiredStatus']=='EXITED':break
   time.sleep(2)
  save('cost.json',{'elapsed_seconds':time.monotonic()-started,'conservative_gpu_usd':(time.monotonic()-started)*PRICE/3600,'price_per_hour':PRICE,'pod':POD,'deleted':False,'retained_volume':True,'status':final['desiredStatus']})
 finally:
  if watchdog is not None and locals().get('final',{}).get('desiredStatus')=='EXITED':watchdog.terminate()
 print('EXACT_POD_STOP_REQUESTED',POD,flush=True)
