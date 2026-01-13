#include <cuda_runtime.h>
#include <cstdio>
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

// 每个block处理(BLOCK_SIZE_ROW, BLOCK_SIZE_COL)大小的子矩阵
constexpr int BLOCK_SIZE_ROW = 64;
constexpr int BLOCK_SIZE_COL = 128;

// 每个warp处理(WRAP_SIZE_ROW, WRAP_SIZE_COL)大小的子矩阵
constexpr int WRAP_SIZE_ROW = 32;
constexpr int WRAP_SIZE_COL = 64;

// 确认block内wrap布局
constexpr int WARPS_PER_BLOCK_ROW = BLOCK_SIZE_ROW / WRAP_SIZE_ROW;        // 2
constexpr int WARPS_PER_BLOCK_COL = BLOCK_SIZE_COL / WRAP_SIZE_COL;        // 2
constexpr int WARPS_PER_BLOCK = WARPS_PER_BLOCK_ROW * WARPS_PER_BLOCK_COL; // 4

// 确认wrap内线程布局
constexpr int THREADS_PER_WRAP = 32;
constexpr int THREADS_PER_WRAP_ROW = 4;
constexpr int THREADS_PER_WRAP_COL = 8;

// 每个thread处理(THREAD_SIZE_ROW, THREAD_SIZE_COL)大小的子矩阵
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
__device__ inline int get_1d_index(int row, int col, int ld)
{
    return row * ld + col;
}

// 一维index转二维index(起点为(0, 0))
__device__ inline void get_2d_index(int index, int ld, int &row, int &col)
{
    row = index / ld;
    col = index % ld;
}

__global__ void matrix_transpose(const float *input, float *output, int rows, int cols)
{
    // 1. 确认当前的线程位置和负责处理的子矩阵位置
    // 确认当前线程所属的block在grid中的位置, 以及所在wrap在block中的位置
    int blockid_row_ingrid = blockIdx.y;
    int blockid_col_ingrid = blockIdx.x;
    int wrapid_row_inblock = threadIdx.y / THREADS_PER_WRAP_ROW;
    int wrapid_col_inblock = threadIdx.x / THREADS_PER_WRAP_COL;

    // 确认当前线程在wrap内的位置
    int threadid_row_inwrap = threadIdx.y % THREADS_PER_WRAP_ROW;
    int threadid_col_inwrap = threadIdx.x % THREADS_PER_WRAP_COL;

    // 确认当前线程负责处理的输入子矩阵的起始位置
    int input_block_row = blockid_row_ingrid * BLOCK_SIZE_ROW + wrapid_row_inblock * WRAP_SIZE_ROW + threadid_row_inwrap * THREAD_SIZE_ROW;
    int input_block_col = blockid_col_ingrid * BLOCK_SIZE_COL + wrapid_col_inblock * WRAP_SIZE_COL + threadid_col_inwrap * THREAD_SIZE_COL;

    // 确定当前线程负责处理的输出子矩阵的起始位置
    int output_block_row = input_block_col;
    int output_block_col = input_block_row;

    // 2. 从block内读取输入矩阵到共享内存
    __shared__ float input_s[BLOCK_SIZE_ROW * BLOCK_SIZE_COL];

    // 确认当前block需要搬运的数据的行起始位置, 列起始位置和向量位置
    int input_block_start_row = blockid_row_ingrid * BLOCK_SIZE_ROW;
    int input_block_start_col = blockid_col_ingrid * BLOCK_SIZE_COL;

    // 确定当前线程需要搬运的数据量, 搬运无需严谨对齐
    constexpr int stride = BLOCK_SIZE_ROW * BLOCK_SIZE_COL / (WARPS_PER_BLOCK * THREADS_PER_WRAP);
    // 但是如果数据量不足, 则需要做边界检查        

    // 当前线程开始读取数据到共享内存
    for (int i = 0; i < stride; i++)
    {
        // 确认当前线程读取的元素在共享内存中的位置和向量位置
        int row_in_shared_block, col_in_shared_block;
        row_in_shared_block = ((threadIdx.y * blockDim.x + threadIdx.x) * stride + i) / BLOCK_SIZE_COL;
        col_in_shared_block = ((threadIdx.y * blockDim.x + threadIdx.x) * stride + i) % BLOCK_SIZE_COL;
        int index_in_shared_block = get_1d_index(row_in_shared_block, col_in_shared_block, BLOCK_SIZE_COL);

        // 确认当前线程读取的元素在全局内存中的位置和向量位置
        int row_in_input_matrix, col_in_input_matrix;
        row_in_input_matrix = input_block_start_row + row_in_shared_block;
        col_in_input_matrix = input_block_start_col + col_in_shared_block;
        int index_in_input_matrix = get_1d_index(row_in_input_matrix, col_in_input_matrix, cols);

        // 只有在原矩阵范围内的才读取，否则共享内存对应位置设为0
        if (row_in_input_matrix < rows && col_in_input_matrix < cols) {
            input_s[index_in_shared_block] = input[index_in_input_matrix];
        } else {
            input_s[index_in_shared_block] = 0.0f;
        }
    }
    // 同步, 当前已经加载好了对应的block到共享内存
    __syncthreads();

    // 计算写回(对应的线程连续选取对应的tile进行转置写回)
    for (int iter_row = 0; iter_row < THREAD_ITER_ROW; iter_row++)
    {
        for (int iter_col = 0; iter_col < THREAD_ITER_COL; iter_col++)
        {
            // 确认当前tile在共享内存中的起始位置
            int row_input_share_iter, col_input_share_iter;
            row_input_share_iter = (wrapid_row_inblock * WRAP_SIZE_ROW) +
                                   (threadid_row_inwrap * THREAD_SIZE_ROW) + (iter_row * TILE_SIZE_ROW);
            col_input_share_iter = (wrapid_col_inblock * WRAP_SIZE_COL) +
                                   (threadid_col_inwrap * THREAD_SIZE_COL) + (iter_col * TILE_SIZE_COL);
            int index_input_share_iter = get_1d_index(row_input_share_iter, col_input_share_iter, BLOCK_SIZE_COL);

            // 确认当前tile在输出矩阵中的起始位置
            int row_output_matrix_iter, col_output_matrix_iter;
            row_output_matrix_iter = output_block_row + (iter_col * TILE_SIZE_COL);
            col_output_matrix_iter = output_block_col + (iter_row * TILE_SIZE_ROW);
            int index_output_matrix_iter = get_1d_index(row_output_matrix_iter, col_output_matrix_iter, rows);

            // 从共享内存加载到寄存器内存(加载的时候按行加载, 按列存回, 实现转置)
            for (int i = 0; i < TILE_SIZE_ROW; i++)
            {
                for (int j = 0; j < TILE_SIZE_COL; j++)
                {
                    // 写回安全检查
                    // 确保共享内存读取不越界（针对BLOCK），且输出矩阵写入不越界（针对COLS/ROWS）
                    int cur_s_row = row_input_share_iter + i;
                    int cur_s_col = col_input_share_iter + j;
                    int cur_out_row = row_output_matrix_iter + j;
                    int cur_out_col = col_output_matrix_iter + i;

                    if (cur_s_row < BLOCK_SIZE_ROW && cur_s_col < BLOCK_SIZE_COL && 
                        cur_out_row < cols && cur_out_col < rows) 
                    {
                        output[cur_out_row * rows + cur_out_col] = input_s[cur_s_row * BLOCK_SIZE_COL + cur_s_col];
                    }
                }
            }
        }
    }
}

extern "C" void solve(const float *input, float *output, int rows, int cols)
{
    // Match block layout with kernel assumptions: x = 16, y = 8 (total 128 threads)
    constexpr int BLOCK_DIM_X = THREADS_PER_WRAP_COL * WARPS_PER_BLOCK_COL; // 8 * 2 = 16
    constexpr int BLOCK_DIM_Y = THREADS_PER_WRAP_ROW * WARPS_PER_BLOCK_ROW; // 4 * 2 = 8
    dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 blocksPerGrid((cols + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL,
                       (rows + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW);

    matrix_transpose<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

int main()
{
    const int rows = 128;
    const int cols = 256;
    const size_t bytes_in = static_cast<size_t>(rows) * cols * sizeof(float);
    const size_t bytes_out = static_cast<size_t>(cols) * rows * sizeof(float);

    // Host buffers
    std::vector<float> h_in(rows * cols);
    std::vector<float> h_out(cols * rows, 0.0f);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            h_in[r * cols + c] = static_cast<float>(r * 0.1f + c * 0.01f);
        }
    }

    // Device buffers
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes_in));
    CHECK_CUDA(cudaMalloc(&d_out, bytes_out));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes_in, cudaMemcpyHostToDevice));

    // Run transpose
    solve(d_in, d_out, rows, cols);

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes_out, cudaMemcpyDeviceToHost));

    // Verify: out[c, r] == in[r, c]
    float max_err = 0.0f;
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            float expect = h_in[r * cols + c];
            float got = h_out[c * rows + r];
            float err = std::fabs(expect - got);
            if (err > max_err)
                max_err = err;
        }
    }

    const float tol = 1e-5f;
    std::printf("Transpose max error: %g\n", max_err);
    std::printf("Sample out[0,0]=%g, out[last]=%g\n", h_out[0], h_out.back());
    if (max_err < tol)
    {
        std::puts("Transpose PASS");
    }
    else
    {
        std::puts("Transpose FAIL");
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}