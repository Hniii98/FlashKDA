"""Compare the baseline binary against the supplied, unmodified PyTorch reference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import flash_kda
import flash_kda_C


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, reference):
    a, r = actual.double(), reference.double()
    d = a - r
    finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
    assert finite, 'Nonfinite comparison'
    return dict(relative_l2=float(d.norm() / r.norm().clamp_min(1e-30)),
                mean_abs=float(d.abs().mean()), max_abs=float(d.abs().max()), finite=finite)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ref-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('fla_ref_errors.json'))
    args = p.parse_args()
    path = args.ref_dir.resolve() / 'naive.py'
    spec = importlib.util.spec_from_file_location('baseline_fla_naive', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    result = dict(reference=str(path), reference_sha256=sha(path),
                  extension=str(flash_kda_C.__file__), extension_sha256=sha(flash_kda_C.__file__),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(), seed=0, rows=rows)
    for H in [96, 64]:
        for lengths in [[8192], [1300, 547, 2048, 963, 271, 3063], [1024]*8]:
            torch.manual_seed(0)
            D, T, N = 128, sum(lengths), len(lengths)
            shape = (1, T, H, D)
            q, k = [F.normalize(torch.randn(shape, device='cuda'), dim=-1).bfloat16() for _ in range(2)]
            v, g = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
            beta = torch.randn(1, T, H, device='cuda', dtype=torch.bfloat16)
            al = torch.rand(H, device='cuda')
            bias = torch.rand(H, D, device='cuda')
            h0 = torch.arange(N*H*D*D, device='cuda', dtype=torch.float32).reshape(N,H,D,D).bfloat16()
            cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device='cuda', dtype=torch.int64) if N > 1 else None
            # Match the mathematical transforms; keep reference intermediates FP32.
            qn, kn = [x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6) for x in [q,k]]
            ga = -5 * torch.sigmoid(al.exp()[None,None,:,None] * (g.float()+bias))
            ba = beta.float().sigmoid()
            outputs, states = [], []
            start = 0
            for i, length in enumerate(lengths):
                sl = slice(start, start+length)
                o, s = module.naive_recurrent_kda(qn[:,sl], kn[:,sl], v[:,sl].float(), ga[:,sl], ba[:,sl],
                    scale=D**-0.5, initial_state=h0[i:i+1].float().transpose(-1,-2).contiguous(), output_final_state=True)
                outputs.append(o)
                states.append(s.transpose(-1,-2))
                start += length
            ref_o, ref_s = torch.cat(outputs, 1), torch.cat(states, 0)
            for dtype in [torch.float32, torch.bfloat16]:
                initial = h0.to(dtype)
                out, state = torch.empty_like(q), torch.empty_like(initial)
                flash_kda.fwd(q,k,v,g,beta,D**-0.5,out,A_log=al,dt_bias=bias,lower_bound=-5,
                              initial_state=initial, final_state=state, cu_seqlens=cu)
                row = dict(H=H,D=D,T=T,seq_lens=lengths,state_dtype=str(dtype),
                           output=metrics(out,ref_o), final_state=metrics(state,ref_s))
                rows.append(row)
                print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
