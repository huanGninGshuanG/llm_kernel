import torch
import triton
import triton.language as tl

def mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale,
    mask=None
) -> torch.Tensor:
    # (batch_size, num_heads, seq_len, head_dim)
    attention_weight = q@k.transpose(-2, -1)
    if mask is not None:
        attention_weight.masked_fill_(mask==0, float("-inf"))
    attention_score = torch.softmax(attention_weight*sm_scale, dim=-1)
    context_vec = attention_score @ v
    return context_vec

@triton.jit
def mha_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    buffer_ptr,
    output_ptr,
    batch_stride,
    head_stride,
    seq_stride,
    buf_batch_stride,
    buf_head_stride,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale,
    BLOCK_SIZE: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    head_start = cur_batch * batch_stride + cur_head * head_stride
    buf_head_start = cur_batch * buf_batch_stride + cur_head * buf_head_stride

    # attention_weight
    for i in range(0, seq_len, BLOCK_SIZE):
        q_row_offs = head_start + i * seq_stride + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim
        col_offs = tl.arange(0, head_dim)[None, :]
        row_mask = i + tl.arange(0, BLOCK_SIZE)[:, None] < seq_len
        col_mask = tl.arange(0, head_dim)[None, :] < head_dim
        q = tl.load(q_ptr + q_row_offs + col_offs, mask=row_mask & col_mask, other=0.0)
        for j in range(0, seq_len, BLOCK_SIZE):
            k_row_offs = head_start + j * seq_stride + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim
            k_row_mask = j + tl.arange(0, BLOCK_SIZE)[:, None] < seq_len
            k = tl.load(k_ptr + k_row_offs + col_offs, mask=k_row_mask & col_mask, other=0.0)
            tmp = tl.dot(q, tl.trans(k)) * sm_scale
            # buffer layout: [B, H, S, S] -> offset = buf_head_start + row * S + col
            k_col_mask = j + tl.arange(0, BLOCK_SIZE)[None, :] < seq_len
            buf_row = (i + tl.arange(0, BLOCK_SIZE)[:, None]) * seq_len
            buf_col = j + tl.arange(0, BLOCK_SIZE)[None, :]
            tl.store(buffer_ptr + buf_head_start + buf_row + buf_col, tmp, mask=row_mask & k_col_mask)

    # softmax (trans -> softmax(axis=0) -> trans = row-wise softmax)
    for i in range(0, seq_len, BLOCK_SIZE):
        row_offs = buf_head_start + i * seq_len + tl.arange(0, BLOCK_SIZE)[:, None] * seq_len
        col_offs = tl.arange(0, seq_len)[None, :]
        row_mask = i + tl.arange(0, BLOCK_SIZE)[:, None] < seq_len
        col_mask = tl.arange(0, seq_len)[None, :] < seq_len
        rows = tl.load(buffer_ptr + row_offs + col_offs, mask=row_mask & col_mask, other=float("-inf"))
        rows = tl.softmax(rows, dim=-1, keep_dims=True)
        tl.store(buffer_ptr + row_offs + col_offs, rows, mask=row_mask & col_mask)

    # attention_score
    for i in range(0, seq_len, BLOCK_SIZE):
        row_offs = buf_head_start + i * seq_len + tl.arange(0, BLOCK_SIZE)[:, None] * seq_len
        col_offs = tl.arange(0, seq_len)[None, :]
        row_mask = i + tl.arange(0, BLOCK_SIZE)[:, None] < seq_len
        col_mask = tl.arange(0, seq_len)[None, :] < seq_len
        w = tl.load(buffer_ptr + row_offs + col_offs, mask=row_mask & col_mask, other=0.0)
        for j in range(0, head_dim, BLOCK_SIZE):
            v_col_offs = j + tl.arange(0, BLOCK_SIZE)[None, :]
            v_row_offs = head_start + tl.arange(0, seq_len)[:, None] * head_dim
            v_row_mask = tl.arange(0, seq_len)[:, None] < seq_len
            v_col_mask = j + tl.arange(0, BLOCK_SIZE)[None, :] < head_dim
            v = tl.load(v_ptr + v_row_offs + v_col_offs, mask=v_row_mask & v_col_mask, other=0.0)
            tmp = tl.dot(w, v)
            # output layout: [B, H, S, D] -> offset = head_start + row * D + col
            o_row_offs = head_start + i * head_dim + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim
            tl.store(output_ptr + o_row_offs + v_col_offs, tmp, mask=row_mask & v_col_mask)

@triton.jit
def flash_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    v_batch_stride,
    v_head_stride,
    v_seq_stride,
    o_batch_stride,
    o_head_stride,
    o_seq_stride,
    sm_scale,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    tid = tl.program_id(2)

    q_head_start = cur_batch * q_batch_stride + cur_head * q_head_stride
    q_row_offs = tid * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None]
    col_offs = tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptr + q_head_start + q_row_offs * HEAD_DIM + col_offs, 
                mask=(q_row_offs<SEQ_LEN)&(col_offs<HEAD_DIM), other=0.0)
    m = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
    o = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)

    
    k_head_start = cur_batch * k_batch_stride + cur_head * k_head_stride
    v_head_start = cur_batch * v_batch_stride + cur_head * v_head_stride
    for i in range(0, SEQ_LEN, BLOCK_SIZE_KV):
        k_row_offs = i * BLOCK_SIZE_KV + tl.arange(0, BLOCK_SIZE_KV)[:, None]
        k = tl.load(k_ptr + k_head_start + k_row_offs * HEAD_DIM + col_offs,
                    mask=(k_row_offs<SEQ_LEN)&(col_offs<HEAD_DIM), other=0.0)
        
        v_row_offs = i * BLOCK_SIZE_KV + tl.arange(0, BLOCK_SIZE_KV)[:, None]
        v = tl.load(v_ptr + v_head_start + v_row_offs * HEAD_DIM + col_offs,
                    mask=(v_row_offs<SEQ_LEN)&(col_offs<HEAD_DIM), other=0.0)
        
        attention_weight = tl.dot(q, tl.trans(k)) * sm_scale
        
        m_cur = tl.max(attention_weight, axis=-1)
        p = tl.exp(attention_weight-m_cur)
        l_cur = tl.sum(p, axis=-1)

        m_new = tl.maximum(m, m_cur)
        alpha = tl.exp(m_cur-m_new)
        beta = tl.exp(m-m_new)
        l_new = beta * l + alpha * l_cur
        o = (tl.dot(alpha[:, None] * p, v) + beta[:, None] * l[:, None] * o) / l_new[:, None]

        l = l_new
        m = m_new
    

    o_head_start = cur_batch * o_batch_stride + cur_head * o_head_stride
    tl.store(output_ptr + o_head_start + q_row_offs * HEAD_DIM + col_offs, o,
             mask=(q_row_offs<SEQ_LEN)&(col_offs<HEAD_DIM))


def flash_attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    B, H, S, D = q.shape
    assert q.shape == k.shape == v.shape
    output = torch.empty_like(q)

    BLOCK_Q, BLOCK_KV = 64, 64
    grid = (B, H, triton.cdiv(S, BLOCK_Q))
    flash_attention[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        sm_scale,
        SEQ_LEN=S, HEAD_DIM=D,
        BLOCK_SIZE_Q=BLOCK_Q, BLOCK_SIZE_KV=BLOCK_KV,
    )
    return output


if __name__ == '__main__':
    B, H, S, D = 2, 4, 128, 32
    sm_scale = 1.0 / (D ** 0.5)

    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)

    # pytorch reference
    out_ref = mha(q, k, v, sm_scale)

    # naive triton mha
    batch_stride = H * S * D
    head_stride = S * D
    seq_stride = D
    buf_batch_stride = H * S * S
    buf_head_stride = S * S

    buffer = torch.empty(B, H, S, S, device='cuda', dtype=torch.float32)
    output = torch.empty(B, H, S, D, device='cuda', dtype=torch.float32)

    BLOCK_SIZE = 32
    grid = (B, H)

    mha_kernel[grid](
        q, k, v, buffer, output,
        batch_stride, head_stride, seq_stride,
        buf_batch_stride, buf_head_stride,
        S, D, sm_scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    diff = (out_ref - output).abs()
    print(f"naive mha: max diff {diff.max().item():.4e},  mean diff {diff.mean().item():.4e}  "
          f"{'PASS' if torch.allclose(out_ref, output, atol=1e-2) else 'FAIL'}")

    # flash attention
    out_flash = flash_attention_forward(q, k, v, sm_scale)

    diff_flash = (out_ref - out_flash).abs()
    print(f"flash attn: max diff {diff_flash.max().item():.4e},  mean diff {diff_flash.mean().item():.4e}  "
          f"{'PASS' if torch.allclose(out_ref, out_flash, atol=1e-2) else 'FAIL'}")

