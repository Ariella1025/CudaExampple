#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>

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

// 实现方式和triton版本相同, 分块实现矩阵乘法
// cuda没有triton那么多的高级特性, 因此只能手动完成指针移动等工作
// BLOCK: 单个block完成矩阵C中BLOCK_SIZE_ROW * BLOCK_SIZE_COL的计算, 并且在A和B矩阵上N维度每次处理N_step个元素
constexpr int BLOCK_SIZE_ROW = 64;
constexpr int BLOCK_SIZE_COL = 128;
constexpr int N_step = 8;

// Wrap: BLOCK内部WRAP_SIZE个线程形成1个wrap(线程束), 每个线程负责计算WRAP_SIZE_ROW * WRAP_SIZE_COL的子矩阵
constexpr int THREADS_PER_WRAP = 32;
constexpr int WRAP_SIZE_ROW = 32;
constexpr int WRAP_SIZE_COL = 64;

// 每个block启动wrap数
constexpr int WARPS_PER_BLOCK_ROW = BLOCK_SIZE_ROW / WRAP_SIZE_ROW; // 2
constexpr int WARPS_PER_BLOCK_COL = BLOCK_SIZE_COL / WRAP_SIZE_COL; // 2
constexpr int WARPS_PER_BLOCK = WARPS_PER_BLOCK_ROW * WARPS_PER_BLOCK_COL; // 4
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * THREADS_PER_WRAP; // 128

// 定义WRAP内线程结构
constexpr int THREADS_PER_WRAP_ROW = 4;
constexpr int THREADS_PER_WRAP_COL = 8;

// 单个线程负责的子矩阵大小
constexpr int THREAD_SIZE_ROW = WRAP_SIZE_ROW / THREADS_PER_WRAP_ROW; // 8
constexpr int THREAD_SIZE_COL = WRAP_SIZE_COL / THREADS_PER_WRAP_COL; // 8

// 单个线程一次负责的 THREAD_SIZE_ROW * THREAD_SIZE_COL 还是较大, 划分为更小的tile, 单个线程循环处理每个tile
// 单个tile的大小 (TILE_SIZE_ROW * TILE_SIZE_COL)
constexpr int TILE_SIZE_ROW = 8;        
constexpr int TILE_SIZE_COL = 4;

// 单个线程内在ROW方向和COL方向的tile数量, 也是单个线程需要循环的次数
constexpr int THREAD_ITER_ROW = THREAD_SIZE_ROW / TILE_SIZE_ROW; // 1
constexpr int THREAD_ITER_COL = THREAD_SIZE_COL / TILE_SIZE_COL; // 2

// 对于一个矩阵每个元素的指针位置偏移函数(二维index转一维index)
__device__ inline int get_1d_index(int row, int col, int ld) {
    return row * ld + col;
}

// 一维index转二维index(起点为(0, 0))
__device__ inline void get_2d_index(int index, int ld, int &row, int &col) {
    row = index / ld;
    col = index % ld;
}

// Matrix multiplication kernel
__global__ void matrix_multiplication(const float* A, const float* B, float* C, int M, int N, int K) {
    // 1. 定位, 确认当前线程需要处理C矩阵的哪个子矩阵
    // 线程当前在整个grid内的位置
    // int threadid_row_ingrid = blockIdx.y * blockDim.y + threadIdx.y;
    // int threadid_col_ingrid = blockIdx.x * blockDim.x + threadIdx.x;

    // 确认当前线程所属的block在grid中的位置, 以及所在wrap在block中的位置
    int blockid_row_ingrid = blockIdx.y;
    int blockid_col_ingrid = blockIdx.x;
    int wrapid_row_inblock = threadIdx.y / THREADS_PER_WRAP_ROW;
    int wrapid_col_inblock = threadIdx.x / THREADS_PER_WRAP_COL;

    // 确认当前线程在wrap内的位置
    int threadid_row_inwrap = threadIdx.y % THREADS_PER_WRAP_ROW;
    int threadid_col_inwrap = threadIdx.x % THREADS_PER_WRAP_COL;

    // 确认当前线程负责的C子矩阵的起始位置
    int C_start_row = blockid_row_ingrid * BLOCK_SIZE_ROW 
                      + wrapid_row_inblock * WRAP_SIZE_ROW
                      + threadid_row_inwrap * THREAD_SIZE_ROW;
    int C_start_col = blockid_col_ingrid * BLOCK_SIZE_COL
                      + wrapid_col_inblock * WRAP_SIZE_COL
                      + threadid_col_inwrap * THREAD_SIZE_COL;

    // 2. 内存处理(注意向量化存储)
    // 一个block处理C矩阵中的BLOCK_SIZE_ROW行, BLOCK_SIZE_COL列, 但是需要循环处理N维度
    // 当前block存储A和B的子矩阵到共享内存(向量化存储)
    __shared__ float As[BLOCK_SIZE_ROW * N_step]; // 每个block处理BLOCK_SIZE_ROW行, N_step列
    __shared__ float Bs[N_step * BLOCK_SIZE_COL]; // 每个block处理N_step行, BLOCK_SIZE_COL列

    // 单个线程需要搬运的A和B的数据量
    constexpr int stride_A = (BLOCK_SIZE_ROW * N_step) / THREADS_PER_BLOCK; // 32
    constexpr int stride_B = (N_step * BLOCK_SIZE_COL) / THREADS_PER_BLOCK; // 32

    // 单个线程自己的寄存器内存
    float C_reg[THREAD_SIZE_ROW * THREAD_SIZE_COL] = {0.0f};        // 用于存储C的子矩阵
    // 线程是一个tile一个tile进行处理的, 因此需要在寄存器内存中存储A和B的tile
    float A_reg[TILE_SIZE_ROW];                          // 用于存储A的tile
    float B_reg[TILE_SIZE_COL];                          // 用于存储B的tile

    for(int n=0; n<N; n+=N_step){
        // 3. 加载到共享内存(注意循环的层次, N维度上是要累加的)
        // 确认当前block当前循环加载的A矩阵块和B矩阵块的行列起始位置和向量化位置
        int A_block_start_row = blockid_row_ingrid * BLOCK_SIZE_ROW;
        int A_block_start_col = n;
        int A_block_start = get_1d_index(A_block_start_row, A_block_start_col, N);
        int B_block_start_row = n;
        int B_block_start_col = blockid_col_ingrid * BLOCK_SIZE_COL;
        int B_block_start = get_1d_index(B_block_start_row, B_block_start_col, K);

        // 确认每个线程加载的A和B元素个数是4的倍数, 从而可以一次搬运4个
        static_assert((stride_A % 4) == 0);
        static_assert((stride_B % 4) == 0);

        // 实际上, 每个block有blockDim.y * blockDim.x个线程, 每个线程负责加载BLOCK_SIZE_ROW / (blockDim.y * blockDim.x)行或列
        // 但是并不一定对齐的, 因此需要计算每个线程实际处理的位置
        
        // 加载A矩阵块到共享内存
        for (int i=0; i<stride_A; i+=4){
            // 确认N_step也是4的倍数
            static_assert((N_step % 4) == 0);

            // 确认当前线程当前循环处理的4个元素在当前block处理的A矩阵块中的位置
            int row_in_A_block, col_in_A_block;
            row_in_A_block = ((threadIdx.y * blockDim.x + threadIdx.x) * stride_A + i) / N_step;
            col_in_A_block = ((threadIdx.y * blockDim.x + threadIdx.x) * stride_A + i) % N_step;

            // 当前线程当前循环处理的4个元素在整个A矩阵中的位置和向量位置
            int row_in_A = A_block_start_row + row_in_A_block;
            int col_in_A = A_block_start_col + col_in_A_block;
            int index_in_A = get_1d_index(row_in_A, col_in_A, N);

            // 读取
            float4 data = *((float4*)&A[index_in_A]);  // 从index_in_A开始连续读取4个float

            // 转置保存到共享内存, 需要先确认这4个元素在共享内存中的位置
            // 确认第一个元素在共享内存中的位置
            int row_in_Ashared_block = col_in_A_block;
            int col_in_Ashared_block = row_in_A_block;
            int index_in_Ashared_block = get_1d_index(row_in_Ashared_block, col_in_Ashared_block, BLOCK_SIZE_ROW);
            // 保存4个元素, float4中是按照x, y, z, w顺序存储的
            // 转置后, 每移动一个元素, 共享内存的向量位置增加BLOCK_SIZE_ROW
            As[index_in_Ashared_block] = data.x;
            As[index_in_Ashared_block + BLOCK_SIZE_ROW] = data.y;
            As[index_in_Ashared_block + 2 * BLOCK_SIZE_ROW] = data.z;
            As[index_in_Ashared_block + 3 * BLOCK_SIZE_ROW] = data.w;
        }

        // 加载B矩阵块到共享内存
        for(int i=0; i<stride_B; i+=4){
            // 确认BLOCK_SIZE_COL也是4的倍数
            static_assert((BLOCK_SIZE_COL % 4) == 0);

            // 确认当前线程当前循环处理的4个元素在当前block处理的B矩阵块中的位置
            int row_in_B_block, col_in_B_block;
            row_in_B_block = ((threadIdx.y * blockDim.x + threadIdx.x) * stride_B + i) / BLOCK_SIZE_COL;
            col_in_B_block = ((threadIdx.y * blockDim.x + threadIdx.x) * stride_B + i) % BLOCK_SIZE_COL;

            // 当前线程当前循环处理的4个元素在整个B矩阵中的位置和向量位置
            int row_in_B = B_block_start_row + row_in_B_block;;
            int col_in_B = B_block_start_col + col_in_B_block;
            int index_in_B = get_1d_index(row_in_B, col_in_B, K);

            // 读取
            float4 data = *((float4*)&B[index_in_B]);  // 从index_in_B开始连续读取4个float

            // 直接保存到共享内存, 不需要转置
            int row_in_Bshared_block = row_in_B_block;
            int col_in_Bshared_block = col_in_B_block;
            int index_in_Bshared_block = get_1d_index(row_in_Bshared_block, col_in_Bshared_block, BLOCK_SIZE_COL);
            // 保存4个元素, float4中是按照x, y, z, w顺序存储的
            Bs[index_in_Bshared_block] = data.x;
            Bs[index_in_Bshared_block + 1] = data.y;
            Bs[index_in_Bshared_block + 2] = data.z;
            Bs[index_in_Bshared_block + 3] = data.w;
        }
        
        // 同步, 确认共享内存加载完成, 获得了共享内存中一个(N_steps, BLOCK_SIZE_ROW)的转置A子矩阵和一个(N_steps, BLOCK_SIZE_COL)的B子矩阵
        __syncthreads();

        // 4. 计算(注意循环的层次, N维度上是要累加的, 这里是第几行)
        for(int j=0; j<N_step; j++){
            // 一个block处理BLOCK_SIZE_ROW行, BLOCK_SIZE_COL列
            // 一个wrap处理WRAP_SIZE_ROW行, WRAP_SIZE_COL列
            // 一个线程处理THREAD_SIZE_ROW行, THREAD_SIZE_COL列
            // 一个线程内一个tile处理TILE_SIZE_ROW行, TILE_SIZE_COL列
            for(int row_iter=0; row_iter<THREAD_ITER_ROW; row_iter++){
                for (int col_iter=0; col_iter<THREAD_ITER_COL; col_iter++){
                    // 计算当前tile在两个共享内存A块和B块中的列起始位置和全局位置
                    int row_Ashared_iter = (wrapid_row_inblock * WRAP_SIZE_ROW) + (threadid_row_inwrap * THREAD_SIZE_ROW) 
                                          + (row_iter * TILE_SIZE_ROW);
                    int col_Bshared_iter = (wrapid_col_inblock * WRAP_SIZE_COL) + (threadid_col_inwrap * THREAD_SIZE_COL) 
                                          + (col_iter * TILE_SIZE_COL);
                    int index_Ashared = get_1d_index(j, row_Ashared_iter, BLOCK_SIZE_ROW);
                    int index_Bshared = get_1d_index(j, col_Bshared_iter, BLOCK_SIZE_COL);

                    // 读取A和B的tile到寄存器
                    for(int t=0; t<TILE_SIZE_ROW; t++)
                        A_reg[t] = As[index_Ashared + t];
                    for(int t=0; t<TILE_SIZE_COL; t++)
                        B_reg[t] = Bs[index_Bshared + t];

                    // 计算C的tile并累加到寄存器
                    for(int t_tile_row=0; t_tile_row<TILE_SIZE_ROW; t_tile_row++){
                        for(int t_tile_col=0; t_tile_col<TILE_SIZE_COL; t_tile_col++){
                            // 确认当前循环处理的tile在当前线程的寄存小块中的行列起始位置和全局位置
                            int row_C_reg_iter = row_iter * TILE_SIZE_ROW + t_tile_row;
                            int col_C_reg_iter = col_iter * TILE_SIZE_COL + t_tile_col;
                            int index_C_reg = get_1d_index(row_C_reg_iter, col_C_reg_iter, THREAD_SIZE_COL);
                            // 累加到当前现成的寄存器小块中(每个N维度是要累加上去的)
                            C_reg[index_C_reg] += A_reg[t_tile_row] * B_reg[t_tile_col];
                        }
                    }
                }
            }
            
        }
        // 同步, 确认计算完成, 每个线程内的寄存器得到了一个()THREAD_SIZE_ROW, THREAD_SIZE_COL)的C子矩阵
        __syncthreads();
    }

    // 5. 写回全局内存(将寄存器累计的结果一次性写回global)
    for(int row_iter=0; row_iter<THREAD_ITER_ROW; row_iter++){
        for(int col_iter=0; col_iter<THREAD_ITER_COL; col_iter++){
            // 计算当前tile起始在C矩阵中的全局位置
            int row_C_iter = C_start_row + row_iter * THREAD_SIZE_ROW;
            int col_C_iter = C_start_col + col_iter * THREAD_SIZE_COL;

            // 写回C矩阵（赋值写回，因为C_reg已包含所有N维度的累加）
            for(int t_tile_row=0; t_tile_row<THREAD_SIZE_ROW; t_tile_row++){
                for(int t_tile_col=0; t_tile_col<THREAD_SIZE_COL; t_tile_col++){
                    int index_C = get_1d_index(row_C_iter + t_tile_row, col_C_iter + t_tile_col, K);
                    int index_C_reg = get_1d_index(row_iter * THREAD_SIZE_ROW + t_tile_row,
                                                  col_iter * THREAD_SIZE_COL + t_tile_col,
                                                  THREAD_SIZE_COL);
                    C[index_C] = C_reg[index_C_reg];
                }
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // Block dims must match the warp tiling in the kernel: x = 8*2 = 16, y = 4*2 = 8 => 128 threads
    constexpr int BLOCK_DIM_X = THREADS_PER_WRAP_COL * WARPS_PER_BLOCK_COL; // 16
    constexpr int BLOCK_DIM_Y = THREADS_PER_WRAP_ROW * WARPS_PER_BLOCK_ROW; // 8
    dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 blocksPerGrid((K + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL,
                       (M + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW);

    // Kernel accumulates into C, so clear it first
    CHECK_CUDA(cudaMemset(C, 0, sizeof(float) * static_cast<size_t>(M) * K));

    matrix_multiplication<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
}

int main(){
    const int M = 8192;
    const int N = 6144;
    const int K = 4096;

    // Host buffers
    const size_t bytesA = static_cast<size_t>(M) * N * sizeof(float);
    const size_t bytesB = static_cast<size_t>(N) * K * sizeof(float);
    const size_t bytesC = static_cast<size_t>(M) * K * sizeof(float);

    std::vector<float> hA(static_cast<size_t>(M) * N, 1.0f);
    std::vector<float> hB(static_cast<size_t>(N) * K, 1.0f);
    std::vector<float> hC(static_cast<size_t>(M) * K, 0.0f);

    // Device buffers
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    // Run kernel via solve
    solve(dA, dB, dC, M, N, K);

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    // Verify: with all-ones inputs, each C[i,k] should be N
    float maxError = 0.0f;
    for (size_t idx = 0; idx < hC.size(); ++idx) {
        float err = std::fabs(hC[idx] - static_cast<float>(N));
        if (err > maxError) maxError = err;
    }

    std::cout << "Matrix multiplication max error: " << maxError << std::endl;
    std::cout << "Sample C[0]: " << hC[0]
              << ", C[last]: " << hC.back() << std::endl;

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return 0;
}