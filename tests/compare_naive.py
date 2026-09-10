"""Compare the C32 candidate against the supplied, unmodified PyTorch reference."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "profile/01_c32_implementation/build"))
import flash_kda_C


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, reference):
    a, r = actual.double(), reference.double()
    d = a - r
    finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
    denom = float(r.norm())
    return dict(relative_l2=(float(d.norm()) / denom if denom else (0.0 if float(d.norm()) == 0 else None)) if finite else None,
                mean_abs=float(d.abs().mean()) if finite else None,
                max_abs=float(d.abs().max()) if finite else None, finite=finite,
                nan_count=int(a.isnan().sum()), inf_count=int(a.isinf().sum()),
                reference_finite=bool(r.isfinite().all()), reference_norm=denom,
                relative_l2_note="undefined if nonfinite or zero reference with nonzero difference")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ref-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('fla_ref_errors.json'))
    p.add_argument('--lower-bound', type=float, default=-5.0)
    p.add_argument('--rescale', type=float, default=None, help='Default: kernel rescale')
    p.add_argument('--inverse-rescale', type=float, default=None, help='Default: kernel inverse_rescale')
    p.add_argument('--rescale-log2', type=int, nargs='+', help='Sweep rescale=2**e; overrides --rescale')
    p.add_argument('--inverse-rescales', type=float, nargs='+', help='Sweep inverse_rescale; overrides --inverse-rescale')
    p.add_argument('--heads', type=int, nargs='+', default=[96, 64])
    p.add_argument('--lengths', type=int, nargs='+', help='Override the existing sequence-length cases')
    p.add_argument('--diagnostics', action='store_true', help='Summarize the actual K1 workspace from the same launch')
    args = p.parse_args()
    rescales = [math.ldexp(1.0, e) for e in args.rescale_log2] if args.rescale_log2 else [flash_kda_C.DEFAULT_RESCALE if args.rescale is None else args.rescale]
    inverse_rescales = args.inverse_rescales or [flash_kda_C.DEFAULT_INVERSE_RESCALE if args.inverse_rescale is None else args.inverse_rescale]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = args.ref_dir.resolve() / 'naive.py'
    spec = importlib.util.spec_from_file_location('candidate_fla_naive', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    result = dict(reference=str(path), reference_sha256=sha(path),
                  extension=str(flash_kda_C.__file__), extension_sha256=sha(flash_kda_C.__file__),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(), seed=0, chunk=32, rescale=rescales[0] if len(rescales) == 1 else None, inverse_rescale=inverse_rescales[0] if len(inverse_rescales) == 1 else None,
                  rescale_grid=rescales, inverse_rescale_grid=inverse_rescales, command=sys.argv,
                  harness_sha256=sha(__file__), lower_bound=args.lower_bound, reference_precision='FP32 recurrent naive, FP32 mathematical preprocessing', rows=rows)
    for H in args.heads:
        for lengths in ([args.lengths] if args.lengths else [[8192], [1300, 547, 2048, 963, 271, 3063], [1024]*8]):
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
            ga = args.lower_bound * torch.sigmoid(al.exp()[None,None,:,None] * (g.float()+bias))
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
            for rescale in rescales:
                for inverse_rescale in inverse_rescales:
                    for dtype in [torch.float32, torch.bfloat16]:
                        initial = h0.to(dtype)
                        out, state = torch.empty_like(q), torch.empty_like(initial)
                        workspace = torch.empty(flash_kda_C.get_workspace_size(T,H,N),device='cuda',dtype=torch.uint8)
                        scale_kwargs = {}
                        if args.rescale is not None or args.rescale_log2:
                            scale_kwargs['rescale'] = rescale
                        if args.inverse_rescale is not None or args.inverse_rescales:
                            scale_kwargs['inverse_rescale'] = inverse_rescale
                        flash_kda_C.fwd(q,k,v,g,beta,D**-0.5,out,workspace,al,bias,args.lower_bound,
                                      initial_state=initial, final_state=state, cu_seqlens=cu,
                                      **scale_kwargs)
                        torch.cuda.synchronize()
                        row = dict(H=H,D=D,T=T,seq_lens=lengths,state_dtype=str(dtype),
                                       rescale=rescale, rescale_log2=math.log2(rescale), inverse_rescale=inverse_rescale,
                                   output=metrics(out,ref_o), final_state=metrics(state,ref_s))
                        if args.diagnostics:
                            actual_tiles=sum((n+31)//32 for n in lengths)
                            allocated_tiles=(T+31)//32+N if cu is not None else actual_tiles
                            offset=0
                            diag={}
                            for name,dt,elems in [('k_decayed',torch.bfloat16,32*D),('q_decayed',torch.bfloat16,32*D),
                                ('k_restored',torch.bfloat16,32*D),('g_total',torch.float32,D),
                                ('INV',torch.bfloat16,32*32),('Mqk',torch.bfloat16,32*32)]:
                                size=H*allocated_tiles*elems*torch.empty((),dtype=dt).element_size()
                                x=workspace[offset:offset+size].view(dt).reshape(H,allocated_tiles,elems)[:,:actual_tiles].float()
                                finite_x=x[x.isfinite()]
                                diag[name]=dict(nan_count=int(x.isnan().sum()),inf_count=int(x.isinf().sum()),
                                    zero_count=int((x==0).sum()),elements=x.numel(),
                                    finite_abs_max=float(finite_x.abs().max()) if finite_x.numel() else None)
                                offset+=size
                            row['k1_workspace']=diag
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                    args.output.write_text(json.dumps(result, indent=2)+'\n')
            args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
