import torch
import importlib.util,runpy,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
binary=Path(sys.argv.pop(1)).resolve()
spec=importlib.util.spec_from_file_location('flash_kda_C',binary)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.modules['flash_kda_C']=m
runpy.run_path(str(root/'tests/compare_naive.py'),run_name='__main__')
