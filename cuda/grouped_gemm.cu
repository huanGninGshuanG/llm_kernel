#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <vector>

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 和 matmul_v2 同构：每个 block 负责一个 BM x BN 的 tile。
// 区别只有两点：
//   1) 多了一维 blockIdx.z，表示这是第几个 group（第几个 GEMM 问题）；
//   2) A/B/C 用 "连续布局": 基指针 + 偏移数组，M/N/K 也变成设备端数组。
template<int BM, int BN, int BK, int TM, int TN>
__global__ void grouped_gemm_v2(
    const float *A_base, const int *a_offsets,   // A: 连续缓冲区 + 每组起始偏移
    const float *B_base, const int *b_offsets,
          float *C_base, const int *c_offsets,
    const int *M, const int *N, const int *K,    // 每组形状（长度 = group 数）
    float alpha, float beta)
{
    const int tid = threadIdx.x;
    const int bx  = blockIdx.x;   // N 方向第几个 tile
    const int by  = blockIdx.y;   // M 方向第几个 tile
    const int g   = blockIdx.z;   // 第几个 group

    // 1) 取出本组的形状和基指针
    const int m = M[g], n = N[g], k = K[g];
    const int m0 = by * BM;       // 本 tile 在组内 M 方向的起点
    const int n0 = bx * BN;       // 本 tile 在组内 N 方向的起点
    if (m0 >= m || n0 >= n) return;   // 整个 tile 都在本组范围外 -> 空转退出

    const float *A = A_base + a_offsets[g];
    const float *B = B_base + b_offsets[g];
          float *C = C_base + c_offsets[g];

    // 2) 线程分工，和 matmul_v2 完全一致
    const int row_cnt    = BM / TM;
    const int col_cnt    = BN / TN;
    const int thread_cnt = row_cnt * col_cnt;   // 应等于 blockDim.x

    const int tx = tid % col_cnt;
    const int ty = tid / col_cnt;

    const int a_tile_row    = tid / BK;
    const int a_tile_col    = tid % BK;
    const int a_tile_stride = thread_cnt / BK;

    const int b_tile_row    = tid / BN;
    const int b_tile_col    = tid % BN;
    const int b_tile_stride = thread_cnt / BN;

    __shared__ float as[BM * BK];
    __shared__ float bs[BK * BN];

    // 3) 沿 K 方向滑动，累计本线程的 TM x TN 结果
    float tmp[TM][TN] = {0.0f};
    for (int k0 = 0; k0 < k; k0 += BK) {
        // ---- 载入 A 的 BM x BK 子块，越界部分补 0 ----
        for (int j = 0; j < BM; j += a_tile_stride) {
            const int row = j + a_tile_row;
            const int col = a_tile_col;
            float v = 0.0f;
            if (m0 + row < m && k0 + col < k)
                v = A[(m0 + row) * k + (k0 + col)];
            as[row * BK + col] = v;
        }
        // ---- 载入 B 的 BK x BN 子块，越界部分补 0 ----
        for (int j = 0; j < BK; j += b_tile_stride) {
            const int row = j + b_tile_row;
            const int col = b_tile_col;
            float v = 0.0f;
            if (k0 + row < k && n0 + col < n)
                v = B[(k0 + row) * n + (n0 + col)];
            bs[row * BN + col] = v;
        }
        __syncthreads();

        // ---- 计算：和 matmul_v2 的内层外积完全一样 ----
        for (int kk = 0; kk < BK; kk++)
            for (int r = 0; r < TM; r++)
                for (int c = 0; c < TN; c++)
                    tmp[r][c] += as[(ty * TM + r) * BK + kk]
                               * bs[kk * BN + tx * TN + c];

        __syncthreads();
    }

    // 4) 写回 C，注意组尺寸未必是 BM/BN 的整数倍，要判边界
    for (int r = 0; r < TM; r++)
        for (int c = 0; c < TN; c++) {
            const int row = m0 + ty * TM + r;
            const int col = n0 + tx * TN + c;
            if (row < m && col < n)
                C[row * n + col] = alpha * tmp[r][c] + beta * C[row * n + col];
        }
}

int main() {
    // ============ 1) 压测配置：模拟 MoE 场景 ============
    // 每个 expert 的权重矩阵形状相同（K x N），但接收到的 token 数 M 不同。
    const int G = 8;                 // expert 数量
    const int N = 2048;              // 每个 expert 的权重列数（FFN 隐藏维）
    const int K = 2048;              // 每个 expert 的权重行数（输入维）
    std::vector<int> h_M = {512, 1024, 2048, 4096, 256, 1536, 3072, 2048};  // 每个 expert 的 token 数
    std::vector<int> h_N(G, N);
    std::vector<int> h_K(G, K);

    // ============ 2) 计算每组元素个数 + 连续布局的起始偏移 ============
    std::vector<int> a_offsets(G), b_offsets(G), c_offsets(G);
    int total_a = 0, total_b = 0, total_c = 0;
    for (int g = 0; g < G; g++) {
        a_offsets[g] = total_a; total_a += h_M[g] * h_K[g];
        b_offsets[g] = total_b; total_b += h_K[g] * h_N[g];
        c_offsets[g] = total_c; total_c += h_M[g] * h_N[g];
    }

    // ============ 3) 构造 host 连续缓冲区并填入随机数据 ============
    std::vector<float> h_A(total_a), h_B(total_b), h_C(total_c), h_C_ref(total_c);
    std::srand(42);
    for (int g = 0; g < G; g++) {
        for (int i = 0; i < h_M[g] * h_K[g]; i++) h_A[a_offsets[g] + i] = (std::rand() % 100) / 10.0f;
        for (int i = 0; i < h_K[g] * h_N[g]; i++) h_B[b_offsets[g] + i] = (std::rand() % 100) / 10.0f;
    }

    // ============ 4) 拷贝到 device ============
    float *d_A, *d_B, *d_C, *d_C_ref;
    int *d_a_off, *d_b_off, *d_c_off, *d_M, *d_N, *d_K;
    checkCudaError(cudaMalloc(&d_A, total_a * sizeof(float)), "malloc A");
    checkCudaError(cudaMalloc(&d_B, total_b * sizeof(float)), "malloc B");
    checkCudaError(cudaMalloc(&d_C, total_c * sizeof(float)), "malloc C");
    checkCudaError(cudaMalloc(&d_C_ref, total_c * sizeof(float)), "malloc C_ref");
    checkCudaError(cudaMalloc(&d_a_off, G * sizeof(int)), "malloc a_off");
    checkCudaError(cudaMalloc(&d_b_off, G * sizeof(int)), "malloc b_off");
    checkCudaError(cudaMalloc(&d_c_off, G * sizeof(int)), "malloc c_off");
    checkCudaError(cudaMalloc(&d_M, G * sizeof(int)), "malloc M");
    checkCudaError(cudaMalloc(&d_N, G * sizeof(int)), "malloc N");
    checkCudaError(cudaMalloc(&d_K, G * sizeof(int)), "malloc K");

    checkCudaError(cudaMemcpy(d_A, h_A.data(), total_a * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), total_b * sizeof(float), cudaMemcpyHostToDevice), "copy B");
    checkCudaError(cudaMemcpy(d_a_off, a_offsets.data(), G * sizeof(int), cudaMemcpyHostToDevice), "copy a_off");
    checkCudaError(cudaMemcpy(d_b_off, b_offsets.data(), G * sizeof(int), cudaMemcpyHostToDevice), "copy b_off");
    checkCudaError(cudaMemcpy(d_c_off, c_offsets.data(), G * sizeof(int), cudaMemcpyHostToDevice), "copy c_off");
    checkCudaError(cudaMemcpy(d_M, h_M.data(), G * sizeof(int), cudaMemcpyHostToDevice), "copy M");
    checkCudaError(cudaMemcpy(d_N, h_N.data(), G * sizeof(int), cudaMemcpyHostToDevice), "copy N");
    checkCudaError(cudaMemcpy(d_K, h_K.data(), G * sizeof(int), cudaMemcpyHostToDevice), "copy K");

    // ============ 5) 总运算量 ============
    // 总 FLOPs = Σ 2 * M_g * N_g * K_g（每组一次乘加 = 2 次浮点运算）
    long long total_flops = 0;
    for (int g = 0; g < G; g++)
        total_flops += 2LL * h_M[g] * h_N[g] * h_K[g];

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

    const float alpha = 1.0f, beta = 0.0f;
    const int warmup_time = 10;
    const int repeat_time = 5;

    // ============ 6) cuBLAS 基线（同时作为正确性参考）============
    // 行主序 C = A·B 等价于列主序的 C^T = B^T · A^T，所以 cublas 里把 B 当 A、A 当 B 传入。
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "cublasCreate failed");

    for (int i = 0; i < warmup_time; i++)
        for (int g = 0; g < G; g++)
            checkCublasError(
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, h_N[g], h_M[g], h_K[g],
                            &alpha, d_B + b_offsets[g], h_N[g], d_A + a_offsets[g], h_K[g],
                            &beta, d_C_ref + c_offsets[g], h_N[g]),
                "cublasSgemm failed");
    cudaDeviceSynchronize();

    checkCudaError(cudaEventRecord(start), "cudaEventRecord(cublas start) failed");
    for (int i = 0; i < repeat_time; i++)
        for (int g = 0; g < G; g++)
            checkCublasError(
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, h_N[g], h_M[g], h_K[g],
                            &alpha, d_B + b_offsets[g], h_N[g], d_A + a_offsets[g], h_K[g],
                            &beta, d_C_ref + c_offsets[g], h_N[g]),
                "cublasSgemm failed");
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord(cublas stop) failed");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize(cublas) failed");
    float cublas_time = 0;
    checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop), "cudaEventElapsedTime(cublas) failed");

    // ============ 7) 启动 grouped kernel（压测对象）============
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;   // 与 matmul_v4 相同的 tile 配置
    int max_M = *std::max_element(h_M.begin(), h_M.end());
    int max_N = *std::max_element(h_N.begin(), h_N.end());

    dim3 block(256);   // = (BM/TM)*(BN/TN) = 16*16
    dim3 grid(CEIL_DIV(max_N, BN), CEIL_DIV(max_M, BM), G);

    checkCudaError(cudaMemset(d_C, 0, total_c * sizeof(float)), "cudaMemset d_C failed");
    for (int i = 0; i < warmup_time; i++)
        grouped_gemm_v2<BM, BN, BK, TM, TN><<<grid, block>>>(
            d_A, d_a_off, d_B, d_b_off, d_C, d_c_off, d_M, d_N, d_K, alpha, beta);
    checkCudaError(cudaDeviceSynchronize(), "warmup sync");
    checkCudaError(cudaGetLastError(), "warmup launch");

    checkCudaError(cudaEventRecord(start), "cudaEventRecord(start) failed");
    for (int i = 0; i < repeat_time; i++)
        grouped_gemm_v2<BM, BN, BK, TM, TN><<<grid, block>>>(
            d_A, d_a_off, d_B, d_b_off, d_C, d_c_off, d_M, d_N, d_K, alpha, beta);
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop) failed");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize(stop) failed");
    checkCudaError(cudaGetLastError(), "timing launch");
    float gemm_time = 0;
    checkCudaError(cudaEventElapsedTime(&gemm_time, start, stop), "cudaEventElapsedTime failed");

    // ============ 8) 报告性能 ============
    float cublas_gflops = repeat_time * (float)total_flops / (cublas_time * 1e6f);
    float kernel_gflops = repeat_time * (float)total_flops / (gemm_time * 1e6f);
    std::cout << "=== grouped GEMM 压测 (G=" << G << ", N=" << N << ", K=" << K << ") ===" << std::endl;
    std::cout << "total FLOPs/iter = " << (float)total_flops / 1e9f << " GFLOP" << std::endl;
    std::cout << "cuBLAS : " << cublas_time << " ms, " << cublas_gflops << " GFlops" << std::endl;
    std::cout << "grouped: " << gemm_time << " ms, " << kernel_gflops << " GFlops" << std::endl;

    // ============ 9) 正确性：与 cuBLAS 对比（数值较大，用相对误差）============
    checkCudaError(cudaMemcpy(h_C.data(), d_C, total_c * sizeof(float), cudaMemcpyDeviceToHost), "copy C back");
    checkCudaError(cudaMemcpy(h_C_ref.data(), d_C_ref, total_c * sizeof(float), cudaMemcpyDeviceToHost), "copy C_ref back");
    int err = 0;
    for (int i = 0; i < total_c; i++) {
        float diff = std::fabs(h_C[i] - h_C_ref[i]);
        float scale = std::fmax(1.0f, std::fabs(h_C_ref[i]));
        if (diff > 1e-3f * scale) {
            if (err < 10) std::cout << "mismatch idx=" << i << " gpu=" << h_C[i]
                                    << " ref=" << h_C_ref[i] << std::endl;
            err++;
        }
    }
    std::cout << "total elems = " << total_c << ", mismatch vs cuBLAS = " << err << std::endl;

    // ============ 10) 清理 ============
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    cudaFree(d_a_off); cudaFree(d_b_off); cudaFree(d_c_off);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_K);
    return 0;
}