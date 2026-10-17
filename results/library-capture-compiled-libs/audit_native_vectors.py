"""Reconstruct matched-call comparisons from native persisted final logits."""
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors.torch import load_file

def diff(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype
 assert torch.isfinite(a).all() and torch.isfinite(b).all()
 d=b.float()-a.float()
 return {'byte_identical':torch.equal(a.view(torch.uint8),b.view(torch.uint8)),'elements':a.numel(),'different_elements':int((d!=0).sum()),'max_abs_difference':float(d.abs().max()),'mean_abs_difference':float(d.abs().mean()),'baseline_top_id':int(a.argmax()),'candidate_top_id':int(b.argmax())}
def audit(root):
 inputs=json.loads((Path(__file__).parent/'reconstructed_inputs_and_model_files.json').read_text());yes,no=inputs['yes_id'][0],inputs['no_id'][0]
 reports={};vectors={};result={'scope':'Persisted final-position full-vocabulary logits, matched prompt and call index. Full-sequence eager comparisons are recorded script metadata, not independently reconstructed: eager raw tensors were not saved. Model/input identity is not verified per arm.','arms':{},'comparisons':{},'repeats':{}}
 for p in sorted(root.glob('*/report.json')):
  arm=p.parent.name;r=json.loads(p.read_text());reports[arm]=r
  result['arms'][arm]={'status':r['status'],'calls':len(r['calls']),'compile':r['compile'],'attention_implementation':r.get('attention_implementation'),'script_sha256':r['script_sha256'],'final_counters':r.get('final_counters'),'calls_metadata':r['calls'],'failure':r.get('failure')}
  raw=p.parent/'last_logits.safetensors'
  if raw.exists():
   vectors[arm]=load_file(raw);result['arms'][arm]['raw_sha256']=hashlib.file_digest(raw.open('rb'),'sha256').hexdigest()
   for row in r['calls']:
    key=f"prompt_{row['prompt_id']}_call_{row['repeat']}";v=vectors[arm][key];h=hashlib.sha256(str((list(v.shape),str(v.dtype))).encode());h.update(v.view(torch.uint8).numpy().tobytes());assert h.hexdigest()==row['last_logits_sha256'];assert int(v.float().argmax())==row['top_token_id'];assert float(v.float()[yes]-v.float()[no])==row['yes_no_log_odds']
   result['repeats'][arm]={f'prompt_{i}':diff(vectors[arm][f'prompt_{i}_call_0'],vectors[arm][f'prompt_{i}_call_1']) for i in range(4) if f'prompt_{i}_call_1' in vectors[arm]}
 for lib in ('transformerlens','nnsight'):
  base=lib+'_none_a'
  if base not in vectors:continue
  for tail in ('q','none_b','residual'):
   arm=lib+'_'+tail
   if arm not in vectors:continue
   result['comparisons'][arm+'_vs_none_a']={k:diff(vectors[base][k],vectors[arm][k]) for k in sorted(vectors[base].keys()&vectors[arm].keys())}
 for lib in ('transformerlens','nnsight'):
  base=lib+'_none_b'
  if base not in vectors:continue
  for tail in ('q','residual'):
   arm=lib+'_'+tail
   if arm in vectors:result['comparisons'][arm+'_vs_none_b']={k:diff(vectors[base][k],vectors[arm][k]) for k in sorted(vectors[base].keys()&vectors[arm].keys())}
 result['additional_validation']={'native_yes_no_log_odds_match_all_rows':True,'raw_activation_tensors_saved':False,'raw_eager_tensors_saved':False,'per_arm_model_and_input_hashes_recorded':False}
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.root);a.out.write_text(json.dumps(r,indent=2)+'\n')
 for name,rows in r['comparisons'].items():print(name,[(k,v['different_elements'],v['max_abs_difference'],v['baseline_top_id']!=v['candidate_top_id']) for k,v in rows.items()])
