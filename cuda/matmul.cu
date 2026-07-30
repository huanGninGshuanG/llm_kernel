
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

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

template<int BLOCK_SIZE>
__global__ void matmul(
    int M, int N, int K, 
    float *A, float *B, float *C,
    float alpha, float beta
) {
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    __shared__ float as[BM * BK];
    __shared__ float bs[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    int ay = threadIdx.x / BK, ax = threadIdx.x % BK;
    int tx = threadIdx.x % BN, ty = threadIdx.x / BN;

    float tmp = 0.0;
    for (int k=0; k<K; k+=BK) {
        as[tid] = A[ay * K + ax];
        bs[tid] = B[ty * N + tx];
        __syncthreads();

        A += BK;
        B += BK * N;

        for(int i=0; i<BK; i++) {
            tmp += as[ty * BK + i]*bs[tx + i * BK];
        }
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}

// 计算访存比优化+全局数据加载优化
template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v2(
    int M, int N, int K,
    float *A, float *B, float *C,
    float alpha, float beta
) {
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row_cnt = BM / TM;
    int col_cnt = BN / TN;
    int thread_cnt = row_cnt * col_cnt;

    int tx = tid % col_cnt;
    int ty = tid / col_cnt;

    int a_tile_row = tid / BK;
    int a_tile_col = tid % BK;
    int a_tile_stride = thread_cnt / BK;
    int b_tile_row = tid / BN;
    int b_tile_col = tid % BN;
    int b_tile_stride = thread_cnt / BN;

    __shared__ float as[BM * BK];
    __shared__ float bs[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp[TM][TN] = {0.0};
    for(int i = 0; i < K; i += BK) {
        for (int j = 0; j < BM; j += a_tile_stride) {
            as[(j+a_tile_row)*BK+a_tile_col] = A[(j+a_tile_row)*K+a_tile_col];
        }
        for (int j = 0; j < BK; j += b_tile_stride) {
            bs[(j+b_tile_row)*BN+b_tile_col] = B[(j+b_tile_row)*N+b_tile_col];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        for(int k=0; k<BK; k++) {
            for(int r=0; r<TM; r++) {
                for(int c=0; c<TN; c++) {
                    tmp[r][c] += as[(ty*TM+r)*BK+k]*bs[k*BN+tx*TN+c];
                }
            }
        }
        __syncthreads();
    }

    for(int i=0; i<TM; i++) {
        for(int j=0; j<TN; j++) {
            C[(ty*TM+i)*N+tx*TN+j] = alpha *tmp[i][j] + beta * C[(ty*TM+i)*N+tx*TN+j];
        }
    }
}

#define FETCH_FLOAT4(data) (reinterpret_cast<float4 *>(&(data))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 向量读优化
template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v3(
    int M, int N, int K,
    float *A, float *B, float *C,
    float alpha, float beta
) {
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row_cnt = BM / TM;
    int col_cnt = BN / TN;
    int thread_cnt = row_cnt * col_cnt;

    int tx = tid % col_cnt * TN;
    int ty = tid / col_cnt * TM;

    int a_iter_cnt = BM * BK / thread_cnt / 4;
    int b_iter_cnt = BK * BN / thread_cnt / 4;

    int a_tile_row = tid / (BK / 4);
    int a_tile_col = tid % (BK / 4) * 4;
    int a_tile_stride = BM / a_iter_cnt;

    int b_tile_row = tid / (BN / 4);
    int b_tile_col = tid % (BN / 4) * 4;
    int b_tile_stride = BK / b_iter_cnt;

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    A = &A[by*BM*K];
    B = &B[bx*BN];
    C = &C[by*BM*N+bx*BN];

    float reg_a[4] = {0.};
    float ta[TM] = {0.};
    float tb[TN] = {0.};
    float tmp[TM][TN] = {0.};

    for (int k=0; k<K; k+=BK) {
        for (int i=0; i<BM; i+=a_tile_stride) {
            FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(A[OFFSET(i+a_tile_row, a_tile_col, K)]);
            As[OFFSET(a_tile_col, i+a_tile_row, BM)] = reg_a[0];
            As[OFFSET(a_tile_col+1, i+a_tile_row, BM)] = reg_a[1];
            As[OFFSET(a_tile_col+2, i+a_tile_row, BM)] = reg_a[2];
            As[OFFSET(a_tile_col+3, i+a_tile_row, BM)] = reg_a[3];
        }

        for (int i=0; i<BK; i+=b_tile_stride) {
            FETCH_FLOAT4(Bs[OFFSET(i+b_tile_row, b_tile_col, BN)]) 
                = FETCH_FLOAT4(B[OFFSET(i+b_tile_row, b_tile_col, N)]);
        }

        __syncthreads();
        
        A += BK;
        B += BK*N;

        for (int l=0; l<BK; l++) {
            for(int r=0; r<TM; r+=4) {
                FETCH_FLOAT4(ta[r])=FETCH_FLOAT4(As[OFFSET(l, ty+r, BM)]);
            }
            for(int c=0; c<TN; c+=4) {
                FETCH_FLOAT4(tb[c])=FETCH_FLOAT4(Bs[OFFSET(l, tx+c, BN)]);
            }

            for(int r=0; r<TM; r++) {
                for(int c=0; c<TN; c++) {
                    tmp[r][c] += ta[r]*tb[c];
                }
            }
        }

        __syncthreads();
    }

    for(int r=0; r<TM; r++) {
        for(int c=0; c<TN; c+=4) {
            float4 t = FETCH_FLOAT4(C[OFFSET(ty+r, tx+c, N)]);
            t.x = alpha * tmp[r][c] + beta * t.x;
            t.y = alpha * tmp[r][c+1] + beta * t.y;
            t.z = alpha * tmp[r][c+2] + beta * t.z;
            t.w = alpha * tmp[r][c+3] + beta * t.w;
            FETCH_FLOAT4(C[OFFSET(ty+r, tx+c, N)]) = t;
        }
    }
}

// 双缓冲优化：隐藏访存时延
template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v4(
    int M, int N, int K,
    float *A, float *B, float *C,
    float alpha, float beta
) {
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int row_cnt = BM / TM;
    const int col_cnt = BN / TN;
    const int thread_cnt = row_cnt * col_cnt;

    int tx = tid % col_cnt * TN;
    int ty = tid / col_cnt * TM;

    const int ldg_a_cnt = BM * BK / thread_cnt / 4;
    const int ldg_b_cnt = BK * BN / thread_cnt / 4;

    int a_tile_row = tid / (BK / 4);
    int a_tile_col = tid % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_cnt;
    int b_tile_row = tid / (BN / 4);
    int b_tile_col = tid % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_cnt;

    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    float a_reg[4 * ldg_a_cnt] = {0.};
    float b_reg[4 * ldg_b_cnt] = {0.};

    float a_calc_reg[2][TM] = {0.};
    float b_calc_reg[2][TN] = {0.};

    A = &A[by*BM*K];
    B = &B[bx*BN];
    C = &C[by*BM*N+bx*BN];

    // 加载第一个tile
    for (int k=0; k<BM; k+=a_tile_stride) {
        int ldg_idx = k / a_tile_stride * 4;
        FETCH_FLOAT4(a_reg[ldg_idx]) = 
            FETCH_FLOAT4(A[OFFSET(a_tile_row+k, a_tile_col, K)]);
        As[0][OFFSET(a_tile_col, k+a_tile_row, BM)] = a_reg[ldg_idx];
        As[0][OFFSET(a_tile_col+1, k+a_tile_row, BM)] = a_reg[ldg_idx+1];
        As[0][OFFSET(a_tile_col+2, k+a_tile_row, BM)] = a_reg[ldg_idx+2];
        As[0][OFFSET(a_tile_col+3, k+a_tile_row, BM)] = a_reg[ldg_idx+3];
    }
    for (int k=0; k<BK; k+=b_tile_stride) {
        FETCH_FLOAT4(Bs[0][OFFSET(k+b_tile_row, b_tile_col, BN)]) = 
            FETCH_FLOAT4(B[OFFSET(k+b_tile_row, b_tile_col, N)]);
    }
    
    __syncthreads();

    for (int i=0; i<TM; i+=4) {
        FETCH_FLOAT4(a_calc_reg[0][i]) = FETCH_FLOAT4(As[0][OFFSET(0, ty+i, BM)]);
    }
    for (int i=0; i<TN; i+=4) {
        FETCH_FLOAT4(b_calc_reg[0][i]) = FETCH_FLOAT4(Bs[0][OFFSET(0, tx+i, BN)]);
    }

    int load_idx, write_idx=1;
    int k=0;
    float tmp[TM][TN] = {0.};
    do {
        k+=BK;
        // 预加载下一个tile, LDG指令是非阻塞的
        if (k<K) {
            for (int i=0; i<BM; i+=a_tile_stride) {
                int ldg_idx = i / a_tile_stride * 4;
                FETCH_FLOAT4(a_reg[ldg_idx]) = 
                    FETCH_FLOAT4(A[OFFSET(a_tile_row+i, a_tile_col+k, K)]);
            }
            for (int i=0; i<BK; i+=b_tile_stride) {
                int ldg_idx = i / b_tile_stride * 4;
                FETCH_FLOAT4(b_reg[ldg_idx]) = 
                    FETCH_FLOAT4(B[OFFSET(i+b_tile_row+k, b_tile_col, N)]);
            }
        }

        // 进行TM * TN的计算
        load_idx = write_idx^1;
        for (int bk=0; bk<BK-1; bk++) {
            // 预取下一个TM
            for (int r=0; r<TM; r+=4) {
                FETCH_FLOAT4(a_calc_reg[(bk+1)%2][r]) = 
                    FETCH_FLOAT4(As[load_idx][OFFSET(bk+1, ty+r, BM)]);
            }
            // 预取下一个TN
            for (int r=0; r<TN; r+=4) {
                FETCH_FLOAT4(b_calc_reg[(bk+1)%2][r]) = 
                    FETCH_FLOAT4(Bs[load_idx][OFFSET(bk+1, tx+r, BN)]);
            }
            // 计算
            for (int r=0; r<TM; r++) {
                for(int c=0; c<TN; c++) {
                    tmp[r][c] += a_calc_reg[bk%2][r]*b_calc_reg[bk%2][c];
                }
            }
        }

        // 使用预取的下一个tile数据更新shared memory，并更新计算寄存器
        if (k<K) {
            for (int i=0; i<BM; i+=a_tile_stride) {
                int ldg_idx = i / a_tile_stride * 4;
                As[write_idx][OFFSET(a_tile_col, i+a_tile_row, BM)] = a_reg[ldg_idx];
                As[write_idx][OFFSET(a_tile_col+1, i+a_tile_row, BM)] = a_reg[ldg_idx+1];
                As[write_idx][OFFSET(a_tile_col+2, i+a_tile_row, BM)] = a_reg[ldg_idx+2];
                As[write_idx][OFFSET(a_tile_col+3, i+a_tile_row, BM)] = a_reg[ldg_idx+3];
            }
            for (int i=0; i<BK; i+=b_tile_stride) {
                int ldg_idx = i / b_tile_stride * 4;
                FETCH_FLOAT4(Bs[write_idx][OFFSET(i+b_tile_row, b_tile_col, BN)]) = 
                    FETCH_FLOAT4(b_reg[ldg_idx]);
            }
            
            __syncthreads();

            for (int i=0; i<TM; i+=4) {
                FETCH_FLOAT4(a_calc_reg[0][i]) 
                    = FETCH_FLOAT4(As[write_idx][OFFSET(0, ty+i, BM)]);
            }
            for (int i=0; i<TN; i+=4) {
                FETCH_FLOAT4(b_calc_reg[0][i]) 
                = FETCH_FLOAT4(Bs[write_idx][OFFSET(0, tx+i, BN)]);
            }
            write_idx ^= 1;
        }
        
        // 计算最后一个外积(隐藏上面的共享内存加载的时延)
        for (int r=0; r<TM; r++) {
            for(int c=0; c<TN; c++) {
                tmp[r][c] += a_calc_reg[(BK-1)%2][r]*b_calc_reg[(BK-1)%2][c];
            }
        }

    }while(k<K);

    // 更新结果
    for (int i=0; i<TM; i++) {
        for(int j=0; j<TN; j+=4) {
            float4 tt = FETCH_FLOAT4(C[OFFSET(ty+i, tx+j, N)]);
            tt.x = alpha * tmp[i][j] + beta * tt.x;
            tt.y = alpha * tmp[i][j+1] + beta * tt.y;
            tt.z = alpha * tmp[i][j+2] + beta * tt.z;
            tt.w = alpha * tmp[i][j+3] + beta * tt.w;
            FETCH_FLOAT4(C[OFFSET(ty+i, tx+j, N)]) = tt;
        }
    }

}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

int main() {
    int N = 1024;
    size_t sz = N * N * sizeof(float);
    float *h_a = (float *)malloc(sz);
    float *h_b = (float *)malloc(sz);
    float *h_cv1 = (float *)malloc(sz);
    float *h_cv2 = (float *)malloc(sz);
    
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

    dim3 blockDim(1024);
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(N, 32));
    for (int i=0; i<warmup_time; i++) {
        matmul<32><<<gridDim, blockDim>>>(N, N, N, d_a, d_b, d_cv1, alpha, beta);
    }
    cudaDeviceSynchronize();

    checkCudaError(cudaMemset(d_cv1, 0, sz), "cudaMemset d_cv1 failed");

    checkCudaError(cudaEventRecord(start),
                    "cudaEventRecord(start v1) failed");

    for (int i = 0; i < repeat_time; ++i) {
        matmul<32><<<gridDim, blockDim>>>(N, N, N, d_a, d_b, d_cv1, alpha, beta);
    }
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
    checkCudaError(cudaEventSynchronize(stop),
                    "cudaEventSynchronize v1 failed");
    float v1_time = 0;
    checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                    "cudaEventElapsedTime v1 failed");

    // 拷贝手写 kernel 结果
    checkCudaError(cudaMemcpy(h_cv1, d_cv1, sz, cudaMemcpyDeviceToHost),
                    "cudaMemcpy h_cv1 failed");

    // ====================== cuda v2/v3/v4 =======================
    dim3 blockDimV2(256);
    dim3 gridDimV2(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

    for (int i = 0; i < warmup_time; ++i) {
        matmul_v4<128, 128, 8, 8, 8>
            <<<gridDimV2, blockDimV2>>>(N, N, N, d_a, d_b, d_cv1, alpha, beta);
    }

    cudaDeviceSynchronize();
    checkCudaError(cudaMemset(d_cv1, 0, sz), "cudaMemset d_cv1 failed");

    checkCudaError(cudaEventRecord(start),
                    "cudaEventRecord(start v2) failed");

    for (int i = 0; i < repeat_time; ++i) {
        matmul_v4<128, 128, 8, 8, 8>
            <<<gridDimV2, blockDimV2>>>(N, N, N, d_a, d_b, d_cv1, alpha, beta);
    }

    checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v2) failed");
    checkCudaError(cudaEventSynchronize(stop),
                    "cudaEventSynchronize v2 failed");

    float v2_time = 0;
    checkCudaError(cudaEventElapsedTime(&v2_time, start, stop),
                    "cudaEventElapsedTime v2 failed");

    // 拷贝手写 kernel 结果
    checkCudaError(cudaMemcpy(h_cv2, d_cv1, sz, cudaMemcpyDeviceToHost),
                    "cudaMemcpy h_cv2 failed");

    // ==================== diff ==========================
    int error_count = 0;
    for (int i = 0; i < N * N && error_count < 10; ++i) {
        if (fabsf(h_c_cubulas[i] - h_cv1[i]) > 1e-5) {
            error_count++;
        }
    }

    std::cout<<"cuBLAS vs V1: "<<error_count<<std::endl;

    error_count = 0;
    for (int i = 0; i < N * N && error_count < 10; ++i) {
        if (fabsf(h_c_cubulas[i] - h_cv2[i]) > 1e-5) {
            error_count++;
        }
    }

    std::cout<<"cuBLAS vs V2: "<<error_count<<std::endl;

    float cublas_gflops =
        repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
    float v1_gflops =
        repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops
    float v2_gflops =
        repeat_time * 2.0f * N * N * N / (v2_time * 1e6f);  // GFlops
    std::cout<<"cublas GFlops: "<<cublas_gflops
        <<", cuda GFlops: "<<v1_gflops
        <<", cuda v2 GFlops: "<<v2_gflops<<std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_cv1);

    free(h_a);
    free(h_b);
    free(h_cv1);
    free(h_cv2);
    free(h_c_cubulas);

    return 0;
}