
constexpr int WARP_SIZE = 32;

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

    int warp_idx = tid % WARP_SIZE;
    int warp_row = warp_idx / (BN / WN);
    int warp_col = warp_idx % (BN / WN);

    const int WSUBN = WN / WNITER;
    const int WSUBM = WARP_SIZE * TN * TM / WSUBN;
    const int WMITER = WM / WSUBM;

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

    
}