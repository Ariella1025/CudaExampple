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

// 线程及块结构
constexpr int THREADS_PER_BLOCK = 256;
constexpr int BLOCK_SIZE = 1024;
constexpr int THREADS_SIZE = BLOCK_SIZE / THREADS_PER_BLOCK;

__global__ void reverse_array(float *input, int N)
{
    // 确定当前block和当前线程处理的数据范围
    int block_head_start = blockIdx.x * BLOCK_SIZE;
    int block_tail_start = max(0, N - (blockIdx.x + 1) * BLOCK_SIZE);

    // 创建共享内存用于存储当前block的数据
    __shared__ float shared_head_data[BLOCK_SIZE];
    __shared__ float shared_tail_data[BLOCK_SIZE];

    // 搬运数据到共享内存
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += THREADS_PER_BLOCK)
    {
        int head_index = block_head_start + i;
        int tail_index = block_tail_start + i;

        if (head_index < N)
            shared_head_data[i] = input[head_index];
        else
            shared_head_data[i] = 0.0f; // 边界处理
        if (tail_index < N)
            shared_tail_data[i] = input[tail_index];
        else
            shared_tail_data[i] = 0.0f;
    }

    __syncthreads();

    // 进行反转并写回全局内存
    // 确认当前线程处理的数据在两个共享内存中的范围
    int head_shared_start = threadIdx.x * THREADS_SIZE;
    int tail_shared_start = BLOCK_SIZE - (threadIdx.x + 1) * THREADS_SIZE;

    // 确认当前线程处理的数据在全局内存中的位置
    int head_global_start = block_head_start + head_shared_start;
    int tail_global_start = block_tail_start + tail_shared_start;

    if (head_global_start <= (N + 1) / 2)
    {
        for (int i = 0; i < THREADS_SIZE; i++)
        {
            int head_global_idx = head_global_start + i;
            int tail_global_idx = tail_global_start + i;
            int head_shared_idx = head_shared_start + i;
            int tail_shared_idx = tail_shared_start + i;

            // 读取共享内存中的数据
            float head_data = shared_head_data[head_shared_idx];
            float tail_data = shared_tail_data[tail_shared_idx];

            // 计算需要写回的位置
            int head_write_idx = N - 1 - head_global_idx;
            int tail_write_idx = N - 1 - tail_global_idx;

            // 写回全局内存
            if (head_write_idx < N && head_write_idx >= 0)
                input[head_write_idx] = head_data;
            if (tail_write_idx < N && tail_write_idx >= 0)
                input[tail_write_idx] = tail_data;
        }
    }
    __syncthreads();
}

// input is device pointer
extern "C" void solve(float *input, int N)
{
    int threadsPerBlock = THREADS_PER_BLOCK;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocksPerGrid = (num_blocks + 1) / 2;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}

int main()
{
    const int N = 10000;

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i)
    {
        h_in[i] = static_cast<float>(i % 123 - 61); // 可重复的简单数据
    }

    float *d_in = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_in, N);

    std::vector<float> h_out(N, 0.0f);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_in, N * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU 参考反转
    bool pass = true;
    for (int i = 0; i < N; ++i)
    {
        float expected = h_in[N - 1 - i];
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
        std::cout << "ReverseArray test passed for N=" << N << std::endl;
    }
    else
    {
        std::cerr << "ReverseArray test FAILED" << std::endl;
    }

    cudaFree(d_in);
    return pass ? 0 : 1;
}
