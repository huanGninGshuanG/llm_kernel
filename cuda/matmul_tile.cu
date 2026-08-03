
constexpr int WARP_SIZE = 32;

#define FETCH_FLOAT4(data) (reinterpret_cast<float4 *>(&(data))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

template<const int M, const int N, const int K, const int BM, const int BN, const int BK>
__device__ void load_to_shared(
    float *A, float *B, float *AS, float *BS,
    int a_tile_row, int a_tile_col, int a_tile_stride,
    int b_tile_row, int b_tile_col, int b_tile_stride
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
        load_to_shared<M, N, K, BM, BN, BK>(A, B, AS, BS, 
            a_tile_row, a_tile_col, a_tile_stride,
            b_tile_row, b_tile_col, b_tile_stride
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
            float *C_inter = C[(i * WSUBM + lane_row * TM) * N + j * WSUBN + lane_col * TN];
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