#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

constexpr int WARP_SIZE = 32;

#define FETCH_FLOAT4(data) (reinterpret_cast<float4 *>(&(data))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
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

template<const int BM, const int BN, const int BK>
__device__ void load_to_shared(
    float *A, float *B, float *AS, float *BS,
    int a_tile_row, int a_tile_col, int a_tile_stride,
    int b_tile_row, int b_tile_col, int b_tile_stride,
    int M, int N, int K
) {
    for (int i=0; i<BM; i+=a_tile_stride) {
        float4 tmp = FETCH_FLOAT4(A[OFFSET(i+a_tile_row, a_tile_col, K)]);
        AS[OFFSET(a_tile_col, a_tile_row+i, BM)] = tmp.x;
        AS[OFFSET(a_tile_col+1, a_tile_row+i, BM)] = tmp.y;
        AS[OFFSET(a_tile_col+2, a_tile_row+i, BM)] = tmp.z;
        AS[OFFSET(a_tile_col+3, a_tile_row+i, BM)] = tmp.w;
    }

    for (int i=0; i<BK; i+=b_tile_stride) {
        FETCH_FLOAT4(BS[OFFSET(i+b_tile_row, b_tile_col, BN)]) = 
            FETCH_FLOAT4(B[OFFSET(i+b_tile_row, b_tile_col, N)]);
    }
}

template<const int BM, const int BN, const int BK, const int WM, const int WN,
        const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
        const int TM, const int TN>
__device__ void calc(
    float *AS, float *BS, float *a_reg, float *b_reg,
    float *thread_result, int warp_row, int warp_col,
    int lane_row, int lane_col
) {
    for (int k=0; k<BK; k++) {
        for(int i=0; i<WMITER; i++) {
            for(int j=0; j<TM; j+=4) {
                FETCH_FLOAT4(a_reg[i * TM + j]) = 
                    FETCH_FLOAT4(AS[OFFSET(k, warp_row * WM + i * WSUBM + lane_row * TM + j, BM)]);
            }
        }

        for (int i=0; i<WNITER; i++) {
            for (int j=0; j<TN; j+=4) {
                FETCH_FLOAT4(b_reg[i * TN + j]) = 
                FETCH_FLOAT4(BS[OFFSET(k, i * WSUBN + warp_col * WN + lane_col * TN + j, BN)]);
            }
        }

        for (int i=0; i<WMITER; i++) {
            for (int j=0; j<WNITER; j++) {
                for (int r=0; r<TM; r++) {
                    for (int c=0; c<TN; c++) {
                        thread_result[(i*TM+r)*(TN*WNITER)+j*TN+c] += a_reg[i*TM+r]*b_reg[j*TN+c];
                    }
                }
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) matmul_warp_tile(
    int M, int N, int K,
    float *A, float *B, float *C,
    float alpha, float beta
) {
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int warp_idx = tid / WARP_SIZE;
    int warp_row = warp_idx / (BN / WN);
    int warp_col = warp_idx % (BN / WN);

    const int WSUBN = WN / WNITER;
    const int WSUBM = WARP_SIZE * TN * TM / WSUBN;
    const int WMITER = WM / WSUBM;

    int laneId = tid % WARP_SIZE;
    int lane_row = laneId / (WSUBN / TN);
    int lane_col = laneId % (WSUBN / TN);

    __shared__ float AS[BK * BM];
    __shared__ float BS[BK * BN];

    int a_tile_row = tid / (BK / 4);
    int a_tile_col = tid % (BK / 4) * 4;
    int a_tile_stride = NUM_THREADS * 4 / BK;

    int b_tile_row = tid / (BN / 4);
    int b_tile_col = tid % (BN / 4) * 4;
    int b_tile_stride = NUM_THREADS * 4 / BN;

    float a_calc[TM * WMITER] = {0.};
    float b_calc[TN * WNITER] = {0.};
    float thread_result[TM * WMITER * TN * WNITER] = {0.};

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[(by * BM + warp_row * WM) * N + bx * BN + warp_col * WN];

    for (int k=0; k<K; k+=BK) {
        load_to_shared<BM, BN, BK>(A, B, AS, BS, 
            a_tile_row, a_tile_col, a_tile_stride,
            b_tile_row, b_tile_col, b_tile_stride,
            M, N, K
        );
        __syncthreads();
        A += BK;
        B += BK * N;
        calc<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            AS, BS, a_calc, b_calc, thread_result, warp_row, warp_col, lane_row, lane_col
        );
        __syncthreads();
    }

    for (int i=0; i<WMITER; i++) {
        for (int j=0; j<WNITER; j++) {
            float *C_inter = &C[(i * WSUBM + lane_row * TM) * N + j * WSUBN + lane_col * TN];
            for (int r=0; r<TM; r++) {
                for (int c=0; c<TN; c+=4) {
                    float4 tmp = FETCH_FLOAT4(C_inter[r * N + c]);
                    const int pos = (i * TM + r) * (TN * WNITER) + j * TN + c;
                    tmp.x = alpha * thread_result[pos] + beta * tmp.x;
                    tmp.y = alpha * thread_result[pos + 1] + beta * tmp.y;
                    tmp.z = alpha * thread_result[pos + 2] + beta * tmp.z;
                    tmp.w = alpha * thread_result[pos + 3] + beta * tmp.w;
                    FETCH_FLOAT4(C_inter[r * N + c]) = tmp;
                }
            }
        }
    }
}

int main() {
    int N = 1024;
    size_t sz = N * N * sizeof(float);
    float *h_a = (float *)malloc(sz);
    float *h_b = (float *)malloc(sz);
    float *h_cv1 = (float *)malloc(sz);
    
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            h_a[i*N+j] = 1.0;
            h_b[i*N+j] = 2.0;
        }
    }

    float *d_a, *d_b, *d_cv1;
    float *h_c_cubulas = (float *)malloc(sz);
    checkCudaError(cudaMalloc((void **)&d_a, sz), "cudaMalloc d_a error");
    checkCudaError(cudaMalloc((void **)&d_b, sz), "cudaMalloc d_b error");
    checkCudaError(cudaMalloc((void **)&d_cv1, sz), "cudaMalloc d_cv1 error");
    checkCudaError(cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice), "cudaMemCpy d_a error");
    checkCudaError(cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice), "cudaMemCpy d_b error");

    
    float alpha = 1.0f;
    float beta = 0.0f;

    // =================== cuBLAS =============================

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "cublasCreate failed");


    // warmup
    int warmup_time = 10;  // 热身次数
    for (int i = 0; i < warmup_time; ++i) {
    checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                    &alpha, d_b, N, d_a, N, &beta, d_cv1, N),
                        "cublasSgemm failed");
    }
    cudaDeviceSynchronize();

    // cuBLAS SGEMM
    int repeat_time = 5;
    checkCudaError(cudaEventRecord(start),
                    "cudaEventRecord(start cublas) failed");
    for (int i = 0; i < repeat_time; ++i) {
    checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                    &alpha, d_b, N, d_a, N, &beta, d_cv1, N),
                        "cublasSgemm failed");
    }

    checkCudaError(cudaEventRecord(stop),
                    "cudaEventRecord(stop cublas) failed");
    checkCudaError(cudaEventSynchronize(stop),
                    "cudaEventSynchronize cublas failed");

    float cublas_time = 0;
    checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                    "cudaEventElapsedTime cublas failed");

    // 拷贝 cuBLAS 结果
    checkCudaError(cudaMemcpy(h_c_cubulas, d_cv1, sz, cudaMemcpyDeviceToHost),
                    "cudaMemcpy C_cublas failed");

    // ======================= cuda v1 ==========================


    const uint K10_NUM_THREADS = 128;
    const uint K10_BN = 128;
    const uint K10_BM = 128;
    const uint K10_BK = 16;
    const uint K10_WN = 64;
    const uint K10_WM = 64;
    const uint K10_WNITER = 4;
    const uint K10_TN = 4;
    const uint K10_TM = 8;
    dim3 blockDim(K10_NUM_THREADS);

    constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

    // warptile in threadblocktile
    static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
    static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);
    // threads in warpsubtile
    static_assert(
        (K10_WM * K10_WN) % (WARP_SIZE * K10_TM * K10_TN * K10_WNITER) == 0);
    constexpr uint K10_WMITER =
        (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
    // warpsubtile in warptile
    static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

    static_assert(
        (K10_NUM_THREADS * 4) % K10_BK == 0,
        "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
        "issues during GMEM->SMEM tiling (loading only parts of the "
        "final row of Bs during each iteraion)");
    static_assert(
        (K10_NUM_THREADS * 4) % K10_BN == 0,
        "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
        "issues during GMEM->SMEM tiling (loading only parts of the "
        "final row of As during each iteration)");
    static_assert(
        K10_BN % (16 * K10_TN) == 0,
        "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(
        K10_BM % (16 * K10_TM) == 0,
        "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

    dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(N, K10_BM));

    for (int i = 0; i < warmup_time; ++i) {
    matmul_warp_tile<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER,
                        K10_TM, K10_TN, K10_NUM_THREADS>
        <<<gridDim, blockDim>>>(N, N, N, d_a, d_b, d_cv1, alpha, beta);
    }
    cudaDeviceSynchronize();

    checkCudaError(cudaEventRecord(start),
                    "cudaEventRecord(start v1) failed");
    for (int i = 0; i < repeat_time; ++i) {
    matmul_warp_tile<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER,
                        K10_TM, K10_TN, K10_NUM_THREADS>
        <<<gridDim, blockDim>>>(N, N, N, d_a, d_b, d_cv1, alpha, beta);
    }
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
    checkCudaError(cudaEventSynchronize(stop),
                    "cudaEventSynchronize v1 failed");
    checkCudaError(cudaGetLastError(), "cuda get last error failed");
    float v1_time = 0;
    checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                    "cudaEventElapsedTime v1 failed");
    checkCudaError(cudaMemcpy(h_cv1, d_cv1, sz, cudaMemcpyDeviceToHost),
                    "cudaMemcpy h_cv1 failed");

    // ==================== diff ==========================
    int error_count = 0;
    for (int i = 0; i < N * N && error_count < 10; ++i) {
        if (fabsf(h_c_cubulas[i] - h_cv1[i]) > 1e-5) {
            error_count++;
        }
    }

    std::cout<<"cuBLAS vs V1: "<<error_count<<std::endl;

    float cublas_gflops =
        repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
    float v1_gflops =
        repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops
    std::cout<<"cublas GFlops: "<<cublas_gflops
        <<", cuda GFlops: "<<v1_gflops<<std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_cv1);

    free(h_a);
    free(h_b);
    free(h_cv1);
    free(h_c_cubulas);

    return 0;
}