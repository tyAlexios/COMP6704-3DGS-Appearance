"""Evidence consistency audit; reads source inputs and original project only."""
import hashlib,json
from pathlib import Path
P=Path('/home/tianyi/data/3D_Research/COMP6704_individual_appearance_20261027')
S=Path('/home/tianyi/data/3D_Research/gaussian-splatting')
def sha(p):
 h=hashlib.sha256()
 with open(str(p),'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
checks={}
for scene in ['train','truck']:
 m=json.loads((P/'results'/(scene+'__main')/'manifest.json').read_text())
 ds=Path('/home/tianyi/data/3D_Research/DATASETS/tandt')/scene
 checks[scene+'_inputs_unchanged']=all(sha(ds/p)==h for p,h in m['input_hashes'].items())
 checks[scene+'_geometry_unchanged']=sha(Path(m['geometry_checkpoint']))==m['geometry_sha256']
 checks[scene+'_split_disjoint']=not(set(m['train_fit'])&set(m['test']) or set(m['validation'])&set(m['test']) or set(m['train_fit'])&set(m['validation']))
checks['original_python_source_matches_snapshot']=all(sha(p)==sha(S/p.relative_to(P/'vendor')) for p in (P/'vendor').rglob('*.py'))
assert all(checks.values()),checks
(P/'results'/'source_integrity_audit.json').write_text(json.dumps(checks,indent=2))
print(checks)
