#include <cuda_runtime.h>
#include <cstdint>
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

// 1个block处理1024个uint32格式的像素
constexpr int BLOCK_SIZE = 1024;
// 1个block包含256个线程
constexpr int THREADS_PER_BLOCK = 256;
// 1个线程处理4个uint32格式的像素
constexpr int THREADS_SIZE = BLOCK_SIZE / THREADS_PER_BLOCK;

__global__ void color_inversion(unsigned char* img, int width, int height){
    // 修改输入的指针, 使其为uint32格式, 移动1位跨32bits
    uint32_t* img_uint32 = reinterpret_cast<uint32_t*>(img);
    
    // 1. 搬运数据到共享内存
    __shared__ uint32_t img_shared[BLOCK_SIZE];
    // 确认一个线程搬运数据量
    constexpr int stride = BLOCK_SIZE / THREADS_PER_BLOCK; // 4

    // 确认当前线程搬运的数据在全局内存中的起始位置
    int start_pos_img = blockIdx.x * BLOCK_SIZE + threadIdx.x * stride;
    // 需要搬运到共享内存的起始位置
    int start_pos_shared = threadIdx.x * stride;

    // 逐个搬运数据到共享内存
    for (int i = 0; i < stride; ++i) {
        int global_idx = start_pos_img + i;
        int shared_idx = start_pos_shared + i;
        
        // mask检查
        if (global_idx < width * height && shared_idx < BLOCK_SIZE){
            img_shared[shared_idx] = img_uint32[global_idx];
        }else{
            img_shared[shared_idx] = 0;
        }
    }
    __syncthreads();

    // 2. 执行颜色反转
    // 逐个处理数据
    for (int i=0; i<THREADS_SIZE; i++) {
        // 确认当前线程当前循环处理的数据在共享内存中的位置和需要写回的全局内存的位置
        int idx_shared = threadIdx.x * THREADS_SIZE + i;
        int idx_global = blockIdx.x * BLOCK_SIZE + threadIdx.x * THREADS_SIZE + i;
        
        // 读取数据到寄存器
        uint32_t pixel = img_shared[idx_shared];

        // 使用位运算进行反转
        pixel = pixel ^ 0x00FFFFFF;

        // 存回对应的位置, 指针还是同一个, 步长改变, 注意mask
        if (idx_global < width * height)
        {
            img_uint32[idx_global] = pixel;
        }
            
    }
    
    
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    color_inversion<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}

int main()
{
    const int width = 512;
    const int height = 256;
    const size_t num_pixels = static_cast<size_t>(width) * height;
    const size_t num_bytes = num_pixels * 4; // RGBA 共4字节

    // 主机端构造简单递增数据，便于验证
    std::vector<unsigned char> h_img(num_bytes);
    for (size_t i = 0; i < num_bytes; ++i)
    {
        h_img[i] = static_cast<unsigned char>(i % 256);
    }

    unsigned char *d_img = nullptr;
    CHECK_CUDA(cudaMalloc(&d_img, num_bytes));
    CHECK_CUDA(cudaMemcpy(d_img, h_img.data(), num_bytes, cudaMemcpyHostToDevice));

    solve(d_img, width, height);

    CHECK_CUDA(cudaMemcpy(h_img.data(), d_img, num_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_img));

    bool ok = true;
    for (size_t p = 0; p < num_pixels; ++p)
    {
        unsigned char r = h_img[p * 4 + 0];
        unsigned char g = h_img[p * 4 + 1];
        unsigned char b = h_img[p * 4 + 2];
        unsigned char a = h_img[p * 4 + 3];

        unsigned char r_expect = static_cast<unsigned char>(255 - static_cast<unsigned char>((p * 4 + 0) % 256));
        unsigned char g_expect = static_cast<unsigned char>(255 - static_cast<unsigned char>((p * 4 + 1) % 256));
        unsigned char b_expect = static_cast<unsigned char>(255 - static_cast<unsigned char>((p * 4 + 2) % 256));
        unsigned char a_expect = static_cast<unsigned char>((p * 4 + 3) % 256);

        if (r != r_expect || g != g_expect || b != b_expect || a != a_expect)
        {
            ok = false;
            break;
        }
    }

    if (ok)
    {
        std::cout << "颜色反转测试通过" << std::endl;
    }
    else
    {
        std::cout << "颜色反转测试失败" << std::endl;
    }

    return 0;
}