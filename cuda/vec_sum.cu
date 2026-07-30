#include <chrono>  // for timing
#include <cmath>   // for INFINITY
#include <cstdlib> // for malloc/free
#include <iostream>
#include <random>

__device__ float warpReduceSum(float val) {
    for(int offset=16; offset>0; offset>>=1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<int N>
__global__ void vec_sum(const float *inp, float *out) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    int warpsPerBlock = blockDim.x / 32;

    const float * start = inp + blockIdx.x*blockDim.x;
    float sum = 0.0;
    for(int i=tid; i<N; i+=blockDim.x) {
        sum += start[i];
    }
    sum = warpReduceSum(sum);
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);
    
    if(laneId==0) {
        shared[warpId] = sum;
    }
    __syncthreads();

    // 树形规约
    for (int offset=warpsPerBlock/2; offset>0; offset>>=1) {
        __syncthreads();
        if (tid<offset) {
            shared[tid] += shared[tid+offset];
        }
    }
    __syncthreads();
    
    if (tid==0) {
        *(out) = shared[0];
    }
    
}

template<int N>
void vec_sum_cpu(const float *inp, float *out) {
    float ans = 0.0;
    for (int i=0; i<N; i++) {
        ans += inp[i];
    }
    out[0] = ans;
}

float gen() {
    std::random_device rd;
    std::mt19937 gen(rd());  // Mersenne Twister 引擎

    // 2. 定义浮点数分布范围 [0.0, 1.0)
    std::uniform_real_distribution<float> dist(0.0, 50.0);

    return dist(gen);
}

int main() {
    const int N = 40960;
    float *h_inp = (float *)malloc(sizeof(float)*N);
    float *h_out = (float *)malloc(sizeof(float));
    float *gpu_out = (float *)malloc(sizeof(float));
    float *d_inp, *d_out;
    for(int i=0; i<N; i++) {
        h_inp[i] = gen();
    }

    vec_sum_cpu<N>(h_inp, h_out);


    cudaMalloc((void **)&d_inp, N * sizeof(float));
    cudaMalloc((void **)&d_out, sizeof(float));
    cudaMemcpy(d_inp, h_inp, N*sizeof(float), cudaMemcpyHostToDevice);
    int block_size = 1024;
    vec_sum<N><<<1, block_size, (block_size/32)*sizeof(float)>>>(d_inp, d_out);
    cudaMemcpy(gpu_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<gpu_out[0]<<" : "<<h_out[0]<<" : "<<fabs(gpu_out[0]-h_out[0])<<std::endl;
    if (fabs(gpu_out[0]-h_out[0])>1e-3f) {
        std::cout<<"Sum not equal, gpu: "<<gpu_out[0]<<", cpu: "<<h_out[0]<<std::endl;
    }


    cudaFree(d_inp);
    cudaFree(d_out);
    free(h_inp);
    free(h_out);
    return 0;
}