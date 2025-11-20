import torch
import triton
from torch.nn import functional as F

from fla.ops.lightning_attn import chunk_lightning_attn


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=["T"],
        # different possible values for `x_name`
        x_vals=[i * 1024 for i in [4, 8, 32, 256]],
        # argument name whose value corresponds to a different line in the plot
        line_arg="provider",
        # possible values for `line_arg``
        line_vals=["chunk_lightning_attn"],
        # label name for the lines
        line_names=["chunk_lightning_attn"],
        # line styles
        styles=[("green", "-")],
        # styles=[('green', '-'), ('blue', '--'), ('red', '-.'),
        #         ('cyan', ':'), ('yellow', 'dotted'), ('cyan', '--'), ('cyan', '-'), ('black', ':')], ylabel="Execution Time (ms)",  # label name for the y-axis # name for the plot. Used also as a file name for saving the plot.  plot_name="Performance",
        ylabel="Executing Tims(ms)",
        plot_name="Performance",
        args={},
    ),
)
def benchmark(T, provider):
    from fla.utils import device

    dtype = torch.bfloat16
    #    dtype = torch.float32
    requires_grad = True
    B, H, D = 4, 16, 128

    q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)

    do = torch.ones_like(q, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    results = triton.testing.do_bench(
        lambda: chunk_lightning_attn(q, k, v), quantiles=quantiles
    )
    return results


if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=".")

