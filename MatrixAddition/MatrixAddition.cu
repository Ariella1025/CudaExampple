#include <cuda_runtime.h>
#include <iostream>
using namespace std;
#include <vector>
#include <cmath>

// Error checking macro
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

__global__ void matrix_addition(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N * N; i += stride)
    {
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_addition<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

int main()
{
    const int N = 4096;
    const int total = N * N;

    vector<float> hA(total), hB(total), hC(total, 0.0f);
    for (int i = 0; i < total; ++i)
    {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(total - i);
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, total * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    solve(dA, dB, dC, N);

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, total * sizeof(float), cudaMemcpyDeviceToHost));

    bool pass = true;
    for (int i = 0; i < total; ++i)
    {
        const float expected = hA[i] + hB[i];
        if (fabs(hC[i] - expected) > 1e-5f)
        {
            pass = false;
            std::cerr << "Mismatch at index " << i << ": got " << hC[i]
                      << ", expected " << expected << std::endl;
            break;
        }
    }

    if (pass)
    {
        std::cout << "Matrix addition test passed for N=" << N << std::endl;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return pass ? 0 : 1;
}
