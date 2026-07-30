#include <cmath>
#include <random>
#include <iostream>

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset>0; offset>>=1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset=16; offset>0; offset>>=1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<int N, int C>
__global__ void softmax(const float *inp, float *out) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int warpsPerBlock = block_size / 32;
    int warpId = tid / 32;
    int laneId = tid % 32;
    float maxv = -INFINITY;
    const float *x = inp + blockIdx.x * C;
    float *y = out + blockIdx.x * C;
    for(int i=tid; i<C; i+=block_size) {
        maxv = fmaxf(maxv, x[i]);
    }
    maxv = warpReduceMax(maxv);
    maxv = __shfl_sync(0xFFFFFFFF, maxv, 0);

    if (laneId==0) {
        shared[warpId] = maxv;
    }
    for(int offset=warpsPerBlock/2; offset>0; offset>>=1) {
        __syncthreads();
        if(tid<offset) {
            shared[tid] = fmaxf(shared[tid], shared[tid+offset]);
        }
    }
    __syncthreads();
    maxv = shared[0];

    float sum = 0.0;
    for (int i=tid; i<C; i+=block_size) {
        float tmp = expf(x[i]-maxv);
        y[i] = tmp;
        sum += tmp;
    }
    sum = warpReduceSum(sum);
    if (laneId==0) {
        shared[warpId] = sum;
    }


    for(int offset=warpsPerBlock/2; offset>0; offset>>=1) {
        __syncthreads();
        if(tid<offset) {
            shared[tid] += shared[tid+offset];
        }
    }
    __syncthreads();
    sum = shared[0];

    for (int i=tid; i<C; i+=block_size) {
        y[i] = y[i]/sum;
    }
}

template<int N, int C>
void softmax_cpu(const float *inp, float *out) {
    for (int i=0; i<N; i++) {
        const float *row = inp + i*C;
        float *orow = out + i*C;

        float maxv = -INFINITY;
        for (int j=0; j<C; j++) {
            maxv = fmaxf(maxv, row[j]);
        }
        float sum = 0.0;
        for (int j=0; j<C; j++) {
            orow[j] = expf(row[j]-maxv);
            sum += orow[j];
        }
        float norm = 1.f / sum;
        for (int j=0; j<C; j++) {
            orow[j] = orow[j]*norm;
        }
    }
}

float gen() {
    std::random_device rd;
    std::mt19937 gen(rd());  // Mersenne Twister 引擎

    // 2. 定义浮点数分布范围 [0.0, 1.0)
    std::uniform_real_distribution<float> dist(0.0, 50.0);

    return dist(gen);
}

template<int N, int C>
bool check(float *out, int type) {
    for (int i=0; i<N; i++) {
        float sum = 0.0;
        for(int j=0; j<C; j++) {
            sum += *(out+i*C+j);
        }
        if(fabs(sum-1.0)>1e-3f) {
            std::cout<<"row sum not 1: "<<sum<<", type: "<<type<<std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 1024;
    const int C = 2048;
    size_t n_elements = N*C;
    float *h_inp = (float *)malloc(sizeof(float)*n_elements);
    float *h_out = (float *)malloc(sizeof(float)*n_elements);
    for (int i=0; i<N; i++) {
        for(int j=0; j<C; j++) {
            *(h_inp+i*C+j) = gen();
        }
    }
    softmax_cpu<N, C>(h_inp, h_out);
    check<N, C>(h_out, 1);

    float *d_inp, *d_out;
    cudaMalloc((void **)&d_inp, n_elements*sizeof(float));
    cudaMalloc((void **)&d_out, n_elements*sizeof(float));
    cudaMemcpy(d_inp, h_inp, n_elements*sizeof(float), cudaMemcpyHostToDevice);
    int warpCnt = 1024/32;
    softmax<N, C><<<N, 1024, warpCnt*sizeof(float)>>>(d_inp, d_out);

    float *gpu_out = (float *)malloc(sizeof(float)*n_elements);
    cudaMemcpy(gpu_out, d_out, n_elements*sizeof(float), cudaMemcpyDeviceToHost);
    check<N, C>(gpu_out, 2);

    bool equal = true;
    for(int i=0; i<N; i++) {
        const float *row1 = h_out+i*C;
        const float *row2 = gpu_out+i*C;
        for (int j=0; j<C; j++) {
            if(fabs(row1[j]-row2[j])>1e-5f) {
                equal = false;
                std::cout<<row1[j]<<" : "<<row2[j]<<std::endl;
                break;
            }
        }
        if(!equal) break;
    }
    if(!equal) {
        std::cout<<"not equal"<<std::endl;
    }

    free(h_inp);
    free(h_out);
    return 0;
}