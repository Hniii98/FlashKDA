import hashlib,json,os,subprocess,sys,re
from pathlib import Path
root=Path(__file__).resolve().parents[3];run=Path(__file__).resolve().parents[1]
p=root/'csrc/smxx/fwd_kernel1.cuh';original=p.read_text()
matches=list(re.finditer(r'__launch_bounds__\(NumThreads, (\d+)\)',original))
assert len(matches)==1
needle=matches[0].group(0)
manifest={'source_before':hashlib.sha256(p.read_bytes()).hexdigest(),'variants':{}}
try:
 for n in [8,5,4,2]:
  p.write_text(original.replace(needle,f'__launch_bounds__(NumThreads, {n})'))
  manifest['variants'][str(n)]={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  code=(root/'profile/01_c32_implementation/build.py').read_text().replace("root / 'profile/01_c32_implementation/build'",f"root / 'profile/03_launch_bounds_tuning/build_{n}'")
  code="__file__="+repr(str(root/'profile/01_c32_implementation/build.py'))+'\n'+code
  with (run/f'analysis/build_{n}.log').open('w') as log:
   subprocess.run([sys.executable,'-c',code],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
  binary=run/f'build_{n}/flash_kda_C.so';manifest['variants'][str(n)]['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
  print('built',n,flush=True)
finally:
 p.write_text(original)
 manifest['source_restored']=hashlib.sha256(p.read_bytes()).hexdigest()
 (run/'analysis/build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
