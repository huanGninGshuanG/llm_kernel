import torch
import triton
import triton.language as tl

def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x@y


def matmul_block(
        x: torch.Tensor, 
        y: torch.Tensor, 
        block_m: int, 
        block_n: int,
        block_k: int
) -> torch.Tensor:
    x_row, x_col = x.shape
    y_row, y_col = y.shape
    assert x_col==y_row, "x's col must equal y's row"
    ans = torch.zeros((x_row, y_col), device='cuda')
    for i in range(0, x_row, block_m):
        for j in range(0, y_col, block_n):
            acc = torch.zeros((block_m, block_n), device='cuda')
            for k in range(0, x_col, block_k):
                a = x[i:i+block_m, k:k+block_k]
                b = y[k:k+block_k, j:j+block_n]
                acc += matmul(a, b)
            ans[i:i+block_m, j:j+block_n] = acc
    return ans

@triton.jit
def matmul_kernel(
    x_ptr, y_ptr, output_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    x_row_offs = BLOCK_SIZE_M * pid_m + tl.arange(0, BLOCK_SIZE_M)[:, None]
    y_col_offs = BLOCK_SIZE_N * pid_n + tl.arange(0, BLOCK_SIZE_N)[None, :]

    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x_offs = k + tl.arange(0, BLOCK_SIZE_K)[None, :]
        x = tl.load(x_ptr+x_row_offs*K+x_offs, mask=(x_row_offs<M)&(x_offs<K), other=0.0)
        x = x.to(tl.float16)

        w_offs = k + tl.arange(0, BLOCK_SIZE_K)[:, None]
        w = tl.load(y_ptr+w_offs*N+y_col_offs, mask=(w_offs<K)&(y_col_offs<N), other=0.0)
        w = w.to(tl.float16)

        z = tl.dot(x, w, acc=z)
    tl.store(output_ptr+x_row_offs*N+y_col_offs, z, mask=(x_row_offs<M)&(y_col_offs<N))




if __name__ == '__main__':
    row = 100
    col = 100
    x = torch.randn((row, col), device='cuda')
    y = torch.randn((row, col), device='cuda')
    res = matmul(x, y)
    block_res = matmul_block(x, y, 10, 10, 10)
    # fp16 参考: (和 kernel 内的 .to(tl.float16) 精度一致)
    res_fp16 = (x.half() @ y.half()).float()

    grid = (triton.cdiv(row, 16), triton.cdiv(col, 16))
    output = torch.empty((row, col), device='cuda', dtype=x.dtype)
    matmul_kernel[grid](x, y, output, row, col, col,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16)
    triton_res = output

    assert torch.allclose(res_fp16, triton_res, atol=5e-2), \
        f"triton vs fp16 ref, max diff: {(res_fp16 - triton_res).abs().max().item():.2e}"
    assert torch.allclose(res, block_res, atol=1e-5), \
        f"block not equal, max diff: {(res - block_res).abs().max().item():.2e}"
    print("max_diff_block: ", (res - block_res).abs().max().item())
    print("max_diff_triton: ", (res_fp16 - triton_res).abs().max().item())