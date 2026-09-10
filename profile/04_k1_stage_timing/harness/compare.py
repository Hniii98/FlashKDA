import torch
import importlib.util,runpy,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
binary=Path(sys.argv.pop(1)).resolve()
spec=importlib.util.spec_from_file_location('flash_kda_C',binary)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if not hasattr(m,'DEFAULT_RESCALE'):
 m.DEFAULT_RESCALE=1.0; m.DEFAULT_INVERSE_RESCALE=1.0
sys.modules['flash_kda_C']=m
runpy.run_path(str(root/'tests/compare_naive.py'),run_name='__main__')

if 'build_C16' in str(binary):
 import json
 output=Path(sys.argv[sys.argv.index('--output')+1]);d=json.loads(output.read_text());d['chunk']=16;d['note']='C16 diagnostic loader: synthetic default rescale metadata only; no scale kwargs passed to C16.';output.write_text(json.dumps(d,indent=2)+'\n')
