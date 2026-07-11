# 大模型常用算子

| 类别 | 算子 | 计算形式 | 典型应用 |
|:---|:---|:---|:---|
| **Attention** | Scaled Dot-Product Attention | `softmax(QK^T / √d) × V` | Self-Attention, Cross-Attention |
| | FlashAttention | 分块 online softmax + recompute | 高效 Attention（避免 O(n²) 显存） |
| | PagedAttention | 分页 KV Cache 管理 | vLLM 推理 |
| | Multi-Head Attention (MHA) | 多组独立的 QKV 投影 + 拼接 | 标准 Transformer |
| | Multi-Query Attention (MQA) | 多 Q 共用一个 KV | 减少 KV Cache |
| | Grouped-Query Attention (GQA) | 多 Q 分组共享 KV | LLaMA 2/3, Qwen |
| | Sliding Window Attention | 限制注意力范围到局部窗口 | Mistral |
| **线性层** | GEMM (MatMul) | `C = αAB + βC` | QKV 投影、FFN |
| | Fused MatMul + Bias + Activation | MatMul → Bias → GeLU/SiLU | FFN 融合加速 |
| | Quantized MatMul (INT8/INT4) | 量化矩阵乘法 | AWQ, GPTQ 量化推理 |
| **激活函数** | GeLU / SiLU (Swish) | `x · σ(x)` / 近似 | FFN 激活 |
| | ReLU / ReLU² | `max(0,x)` / `max(0,x)²` | 稀疏激活 |
| | Softmax | `exp(x_i) / Σexp(x_j)` | Attention 权重 |
| | LogSoftmax | `log(softmax(x))` | 交叉熵损失 |
| **归一化** | LayerNorm | `(x - μ) / σ × γ + β` | Pre/Post LN |
| | RMSNorm | `x / RMS(x) × γ` | LLaMA 系列 |
| | GroupNorm | 分组归一化 | 部分视觉模型 |
| | BatchNorm | 批次归一化 | 训练中不常用 |
| **位置编码** | RoPE (Rotary Position Embedding) | 旋转变换 Q, K | LLaMA, Qwen, Mistral |
| | ALiBi | 线性偏置加到注意力分数 | BLOOM |
| | Sinusoidal PE | 正余弦位置编码 | 原始 Transformer |
| **Embedding** | Token Embedding Lookup | 查表 | 输入 Embedding |
| | Position Embedding | 查表 + 加法 | 位置信息 |
| **Loss** | Cross Entropy Loss | `-Σ y·log(ŷ)` | 语言模型训练 |
| | KL Divergence | `Σ p·log(p/q)` | 蒸馏 |
| **优化器算子** | AdamW | 梯度 + 动量 + 权重衰减 | 训练 |
| | Fused Adam | 融合梯度更新 + 缩放 | 训练加速 |
| **通信** | AllReduce | 多卡梯度求和 | 数据并行 |
| | AllGather | 多卡数据收集 | 张量并行 (TP) |
| | ReduceScatter | 归约 + 分发 | ZeRO |
| | Pipeline Send/Recv | P2P 传输 | 流水线并行 (PP) |
| **元素级** | Add / Mul / Div | 逐元素运算 | Residual Add, Scale |
| | Dropout | 随机置零 + 缩放 | 训练正则化 |
| | GELU Backward | 反向传播 | 训练 |
| | Copy / Transpose / Permute | 数据搬运 | Layout 转换 |
| **融合算子** | Fused MHA (FlashAttention) | QK^T → softmax → ×V | 端到端 Attention |
| | Fused MLP | MatMul → Bias → Act → MatMul → Bias | FFN 融合 |
| | Fused LayerNorm + Residual | LN + Add | Pre/Post Norm |
| | Fused RoPE | 原地旋转 Q/K | 位置编码加速 |
| **推理专用** | KV Cache Append | 追加新 token 的 KV | 自回归解码 |
| | Top-K / Top-P Sampling | 截断采样 | 文本生成 |
| | Beam Search | 束搜索 | 解码策略 |
| | Speculative Decoding | 小模型预测 + 大模型验证 | 推理加速 |

## 按计算密集度分类

```
计算密集 (瓶颈在 FLOPs):
  GEMM, FlashAttention, Fused MLP, RoPE

IO 密集 (瓶颈在带宽):
  Element-wise Ops, LayerNorm, RMSNorm, Dropout,
  Softmax, Embedding Lookup, Copy/Transpose

通信密集 (瓶颈在网络):
  AllReduce, AllGather, ReduceScatter
```

Triton 最适合写 **融合算子**（把多个 IO 密集的 kernel 合成一个，减少显存读写），比如 Fused MLP、Fused LayerNorm+Residual、FlashAttention 分块逻辑。
