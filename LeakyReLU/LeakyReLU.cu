#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

constexpr int BLOCK_SIZE = 1024;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREAD_SIZE = BLOCK_SIZE / THREADS_PER_BLOCK;
constexpr float LEAKY_SLOPE = 0.01f;

__global__ void leaky_relu(const float *input, float *output, int N)
{
    // 确认当前block处理的数据的起始位置
    int block_start = blockIdx.x * BLOCK_SIZE;

    // 加载数据到共享内存
    __shared__ float shared_data[BLOCK_SIZE];
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += THREADS_PER_BLOCK)
    {
        if (block_start + i < N)
            shared_data[i] = input[block_start + i];
        else
            shared_data[i] = 0.0f; // 边界处理
    }

    __syncthreads();

    // 确认当前线程处理的数据的共享内存起始位置和全局起始位置
    int thread_start_global = block_start + threadIdx.x * THREAD_SIZE;
    int thread_start_shared = threadIdx.x * THREAD_SIZE;

    // 单线程逐个处理数据并写回
    for (int i = 0; i < THREAD_SIZE; i++)
    {
        int idx_shared = thread_start_shared + i;
        int idx_global = thread_start_global + i;

        if (idx_global < N)
        {
            output[idx_global] = shared_data[idx_shared] > 0.0f ? shared_data[idx_shared] : LEAKY_SLOPE * shared_data[idx_shared];
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N)
{
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    leaky_relu<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

int main()
{
    const int N = 10000;

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i)
    {
        float v = static_cast<float>((i % 7) - 3); // values in [-3,3]
        h_in[i] = v;
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_in, d_out, N);

    std::vector<float> h_out(N, 0.0f);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool pass = true;
    for (int i = 0; i < N; ++i)
    {
        float v = h_in[i];
        float expected = v > 0.0f ? v : LEAKY_SLOPE * v;
        if (h_out[i] != expected)
        {
            pass = false;
            std::cerr << "Mismatch at index " << i << ": got " << h_out[i]
                      << ", expected " << expected << std::endl;
            break;
        }
    }

    if (pass)
    {
        std::cout << "LeakyReLU test passed for N=" << N << std::endl;
    }
    else
    {
        std::cerr << "LeakyReLU test FAILED" << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return pass ? 0 : 1;
}
