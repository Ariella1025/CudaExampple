import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # 执行矩阵乘法
    # :param a: (M, N)
    # :param b: (N, K)
    # :param c: (M, K)
    # :param stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck: 
    # 各个矩阵在对应维度上移动一个index实际存储位置需要移动的步长

    a_m_idx = tl.program_id(0)
    b_k_idx = tl.program_id(1)

    # 为每个实例创建处理的块指针
    a_block_ptr = tl.make_block_ptr(
        base=a,
        shape=(M, N),
        strides=(stride_am, stride_an),
        offsets=(a_m_idx * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(0, 1)         # 行优先
    )

    b_block_ptr = tl.make_block_ptr(
        base=b,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        offsets=(0, b_k_idx * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(0, 1)
    )
    
    c_block_ptr = tl.make_block_ptr(
        base=c,
        shape=(M, K),
        strides=(stride_cm, stride_ck),
        offsets=(a_m_idx * BLOCK_SIZE_M, b_k_idx * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(0, 1)
    )

    # 初始化输出块
    Ct = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for _ in range(tl.cdiv(N, BLOCK_SIZE_N)):
        # 加载当前实例当前循环处理的At和Bt
        At = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Bt = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        Ct = tl.dot(At, Bt, acc=Ct, allow_tf32=False)

        # 指针移动
        a_block_ptr = a_block_ptr.advance((0, BLOCK_SIZE_N))  
        b_block_ptr = b_block_ptr.advance((BLOCK_SIZE_N, 0))      

    # 存储
    tl.store(c_block_ptr, Ct.to(tl.float32), boundary_check=(0, 1))


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = a.stride()
    stride_bn, stride_bk = b.stride()
    stride_cm, stride_ck = c.stride()
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    # 检查输入矩阵的形状是否满足要求
    assert a.shape == (M, N)
    assert b.shape == (N, K)
    assert c.shape == (M, K)

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(K, BLOCK_SIZE_K)
    )
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

if __name__ == "__main__":
    M, N, K = 129, 257, 513

    # ========== 补充：固定随机种子+禁用TF32（保证精度） ==========
    torch.manual_seed(42)

    # 使用PyTorch新版本API禁用TF32, 保证都使用FP32精度
    # triton默认不使用TF32，但PyTorch的matmul和conv可能会使用TF32
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'
    
    a = torch.randn((M, N), device='cuda', dtype=torch.float32)
    b = torch.randn((N, K), device='cuda', dtype=torch.float32)
    c = torch.zeros((M, K), device='cuda', dtype=torch.float32)

    solve(a, b, c, M, N, K)

    print("Result matrix C shape:", c.shape)
    c_ref = torch.matmul(a, b)
    
    if torch.allclose(c, c_ref, rtol=1e-3, atol=1e-4):
        print("Matrix multiplication successful and verified!")
    else:
        print("There is an error in the matrix multiplication.")
        max_error = torch.max(torch.abs(c - c_ref)).item()
        print(f"Maximum error between c and c_ref: {max_error:.6f}")