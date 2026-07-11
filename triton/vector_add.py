import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,      # 输入向量 x
    y_ptr,      # 输入向量 y
    out_ptr,    # 输出向量 out
    n_elements, # 向量长度
    BLOCK_SIZE: tl.constexpr,  # 每个 program 处理的元素数
):
    # 获取当前 program 的 ID
    pid = tl.program_id(axis=0)
    # 计算当前 block 的起始偏移量
    block_start = pid * BLOCK_SIZE
    # 生成元素索引: [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建 mask，防止越界访问
    mask = offsets < n_elements

    # 从全局内存加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 向量加法
    out = x + y

    # 将结果写回全局内存
    tl.store(out_ptr + offsets, out, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """向量加法封装函数"""
    assert x.shape == y.shape and x.is_cuda and y.is_cuda
    out = torch.empty_like(x)
    n_elements = x.numel()

    # 选择合适的 BLOCK_SIZE
    BLOCK_SIZE = 1024
    # 计算需要多少个 blocks
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # 启动 kernel
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


if __name__ == "__main__":
    # 测试
    n = 100_000
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    # Triton 向量加法
    out_triton = vector_add(x, y)

    # PyTorch 原生加法 (作为 ground truth)
    out_torch = x + y

    # 验证结果
    assert torch.allclose(out_triton, out_torch, atol=1e-6), "结果不一致!"
    print(f"向量长度: {n}")
    print(f"Triton 和 PyTorch 结果一致: True")
    print(f"前5个元素: {out_triton[:5].tolist()}")
