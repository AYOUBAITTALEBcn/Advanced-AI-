"""Efficiency profiling: params, FLOPs (fixed-size input), latency."""
import time

import torch

from data.mosaic import bayer_mask


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _dummy_inputs(size, device):
    x = torch.rand(1, 3, size, size, device=device)
    x = x * bayer_mask(size, size, device, x.dtype)
    base = torch.rand(1, 3, size, size, device=device)
    return x, base


def measure_flops(model, size=256, device='cpu'):
    """Multiply-accumulate-based GFLOPs at size x size; None if unavailable.

    ptflops/fvcore hook every submodule to count MACs; on architectures they
    don't fully understand (e.g. SwinIR's windowed attention) a failed
    attempt can still leave several GB allocated on the way to raising --
    empty_cache() after each failure so a later measure_latency() call in
    the same process doesn't start out of memory.
    """
    model = model.eval()
    x, base = _dummy_inputs(size, device)
    is_cuda = torch.device(device).type == 'cuda'
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, (3, size, size),
            input_constructor=lambda shape: {'x': x, 'base': base},
            as_strings=False, print_per_layer_stat=False, verbose=False)
        return 2 * macs / 1e9  # MACs -> FLOPs
    except Exception as e:  # noqa: BLE001
        print(f'[profile] ptflops failed ({e}); trying fvcore')
        if is_cuda:
            torch.cuda.empty_cache()
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, (x, base))
        fca.unsupported_ops_warnings(False)
        return 2 * fca.total() / 1e9  # fvcore counts MACs for convs/linears
    except Exception as e:  # noqa: BLE001
        print(f'[profile] fvcore failed ({e}); FLOPs unavailable')
        if is_cuda:
            torch.cuda.empty_cache()
        return None


@torch.no_grad()
def measure_latency(model, size=256, device='cpu', warmup=10, runs=50):
    """Mean forward latency (ms) over `runs` after `warmup`, synchronized."""
    model = model.eval().to(device)
    x, base = _dummy_inputs(size, device)
    is_cuda = torch.device(device).type == 'cuda'
    for _ in range(warmup):
        model(x, base)
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x, base)
    if is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1e3


def profile_model(model, size=256, device='cpu'):
    flops = measure_flops(model, size, device)
    return {
        'params_M': count_params(model) / 1e6,
        'flops_G': flops if flops is not None else float('nan'),
        'latency_ms': measure_latency(model, size, device),
    }
