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

// 每个block处理1024个输出元素
constexpr int BLOCK_SIZE = 1024;
constexpr int THREADS_PER_BLOCK = 256;
// 每个thread处理的输出元素数量
constexpr int THREAD_SIZE = BLOCK_SIZE / THREADS_PER_BLOCK;

__global__ void convolution_1d_kernel(const float *input, const float *kernel, float *output,
                                      int input_size, int kernel_size)
{
    // kernel分块大小
    const int KERNEL_BLOCK_SIZE = 64;
    // 申请一片连续内存
    extern __shared__ float s_mem[];
    // 初始给input_shared
    float *input_shared = s_mem;
    // [0, BLOCK_SIZE + kernel_size -1) 用于存放输入数据分片
    // [BLOCK_SIZE + kernel_size -1, ... ) 用于存放kernel数据分片]
    float *kernel_data = &s_mem[BLOCK_SIZE + kernel_size - 1];

    // 2. 搬运数据到共享内存
    // 计算当前block需要加载的输入数据的起始位置
    int input_start = blockIdx.x * BLOCK_SIZE;
    // 执行搬运
    for (int i = threadIdx.x; i < BLOCK_SIZE + kernel_size - 1; i += blockDim.x)
    {
        int input_idx = input_start + i;
        if (input_idx < input_size)
            input_shared[i] = input[input_idx];
        else
            input_shared[i] = 0.0f; // 边界处理
    }

    for (int i = threadIdx.x; i < kernel_size; i += blockDim.x)
        kernel_data[i] = kernel[i];

    __syncthreads();

    // 确定当前线程当前循环需要读取共享内存中的数据的范围, 此处不引入二维矩阵, 直接计算偏移
    int input_shared_start = threadIdx.x * THREAD_SIZE;

    // 定义寄存器内存
    float acc[THREAD_SIZE] = {0.0f};
    float kernel_block[KERNEL_BLOCK_SIZE] = {0.0f};

    // 3. 计算巻积
    // kernel块数
    int num_kernel_block = (kernel_size + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;
    for (int i = 0; i < num_kernel_block; i++)
    {
        // 从shared加载kernel block到寄存器内存
        for (int j = 0; j < KERNEL_BLOCK_SIZE; j++)
        {
            // 加载kernel到寄存器内存
            if (i * KERNEL_BLOCK_SIZE + j < kernel_size)
                kernel_block[j] = kernel_data[i * KERNEL_BLOCK_SIZE + j];
            else
                kernel_block[j] = 0.0f; // 边界处理

            // 计算巻积
            for (int t = 0; t < THREAD_SIZE; t++)
            {
                // 直接使用偏移来计算巻积
                acc[t] += input_shared[input_shared_start + i * KERNEL_BLOCK_SIZE + t + j] * kernel_block[j];
            }
        }
    }

    // 写回输出
    for (int t = 0; t < THREAD_SIZE; t++)
    {
        int output_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x * THREAD_SIZE + t;
        if (output_idx < (input_size - kernel_size + 1))
        {
            output[output_idx] = acc[t];
        }
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, const float *kernel, float *output, int input_size,
                      int kernel_size)
{
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    // 动态共享内存大小：输入窗口 (BLOCK_SIZE + kernel_size - 1) + 完整 kernel (kernel_size)
    // 因为申请了动态内存, 所以需要传入具体大小
    size_t shm_bytes = static_cast<size_t>(BLOCK_SIZE + kernel_size - 1 + kernel_size) * sizeof(float);
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, shm_bytes>>>(input, kernel, output, input_size,
                                                                         kernel_size);
    cudaDeviceSynchronize();
}

int main()
{
    const int input_size = 4096;
    const int kernel_size = 31;
    const int output_size = input_size - kernel_size + 1;

    std::vector<float> h_input(input_size);
    std::vector<float> h_kernel(kernel_size);
    std::vector<float> h_output(output_size, 0.0f);
    std::vector<float> h_ref(output_size, 0.0f);

    for (int i = 0; i < input_size; ++i)
    {
        h_input[i] = static_cast<float>(i % 13) - 6.0f;
    }
    for (int i = 0; i < kernel_size; ++i)
    {
        h_kernel[i] = static_cast<float>((i % 7) - 3);
    }

    float *d_input = nullptr;
    float *d_kernel = nullptr;
    float *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_input, d_kernel, d_output, input_size, kernel_size);

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference
    for (int i = 0; i < output_size; ++i)
    {
        float s = 0.0f;
        for (int k = 0; k < kernel_size; ++k)
        {
            s += h_input[i + k] * h_kernel[k];
        }
        h_ref[i] = s;
    }

    double max_err = 0.0;
    for (int i = 0; i < output_size; ++i)
    {
        double diff = static_cast<double>(h_output[i]) - static_cast<double>(h_ref[i]);
        if (std::abs(diff) > max_err)
        {
            max_err = std::abs(diff);
        }
    }

    std::cout << "Conv1D test max error: " << max_err << std::endl;
    if (max_err < 1e-3)
    {
        std::cout << "Conv1D test passed" << std::endl;
    }
    else
    {
        std::cerr << "Conv1D test FAILED" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    return 0;
}
