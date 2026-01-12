#include <cuda_runtime.h>
#include <iostream>
using namespace std;
#include <vector>

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

__global__ void vector_add(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride)
    {
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

int main()
{
    const int N = 1 << 20; // 1,048,576 elements
    const size_t bytes = N * sizeof(float);

    // Host buffers
    vector<float> hA(N), hB(N), hC(N);
    for (int i = 0; i < N; ++i)
    {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(N - i);
    }

    // Device buffers
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    solve(dA, dB, dC, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        float expected = hA[i] + hB[i];
        float err = std::abs(hC[i] - expected);
        if (err > maxError)
            maxError = err;
    }

    cout << "Vector addition max error: " << maxError << endl;
    cout << "Sample C[0]: " << hC[0] << " C[N-1]: " << hC[N - 1] << endl;

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return 0;
}