import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, 
    output_ptr, 
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    tid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    input_ptrs = input_ptr + tid * input_row_stride + offs
    row = tl.load(input_ptrs, mask=offs<n_cols, other=float("-inf"))

    row_minus_max = row - tl.max(row, axis=-1)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=-1)
    output = numerator / denominator

    output_ptrs = output_ptr + tid * output_row_stride + offs
    tl.store(output_ptrs, output, mask=offs<n_cols)

def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    softmax_kernel[(n_rows,)](
        x, 
        y, 
        x.stride(0), 
        y.stride(0), 
        n_cols, 
        num_warps=num_warps,
        BLOCK_SIZE = BLOCK_SIZE)

    return y

def softmax_naive(x: torch.Tensor) -> torch.Tensor: 
    max_val, _ = torch.max(x, dim=-1)
    x_minus_max = x - max_val[:, None]
    numerator = torch.exp(x_minus_max)
    denumerator = torch.sum(numerator, dim=-1)
    return numerator / denumerator[:, None]


if __name__ == '__main__':
    n_rows = 100_00
    n_cols = 50_00
    x = torch.randn((n_rows, n_cols), device='cuda')

    out_triton = softmax(x)
    out_torch = softmax_naive(x)
    assert torch.allclose(out_triton, out_torch, atol=1e-6), "结果不一致!"
    assert torch.allclose(torch.sum(out_triton, dim=-1), torch.ones(n_rows, device='cuda'), atol=1e-6), "结果错误"
    assert torch.allclose(torch.sum(out_torch, dim=-1), torch.ones(n_rows, device='cuda'), atol=1e-6), "结果错误"
    