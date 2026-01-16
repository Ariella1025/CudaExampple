import torch
import triton
import triton.language as tl


@triton.jit
def MatrixAddition(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    """执行矩阵加法"""
    pid = tl.program_id(0)

    # 加载块指针
    a_block_ptr = tl.make_block_ptr(
        base=a,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    c_block_ptr = tl.make_block_ptr(
        base=c,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )

    # 加载块矩阵
    a_block = tl.load(a_block_ptr, boundary_check=(0,), padding_option="zero")
    b_block = tl.load(b_block_ptr, boundary_check=(0,), padding_option="zero")

    # 执行加法
    c_block = a_block + b_block

    # 写回
    tl.store(c_block_ptr, c_block, boundary_check=(0,))


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    MatrixAddition[grid](a, b, c, n_elements, BLOCK_SIZE)


def _test_matrix_addition(N: int = 128, atol: float = 1e-4, rtol: float = 1e-4):
    """简单校验: 随机生成N×N矩阵, 调用triton核并与torch结果对比"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available; cannot run Triton kernel test")

    device = "cuda"
    a = torch.randn((N, N), device=device, dtype=torch.float32)
    b = torch.randn_like(a)
    c = torch.empty_like(a)

    # Triton核按一维展开的向量处理, 因此传入扁平视图
    solve(a.view(-1), b.view(-1), c.view(-1), N)

    expected = a + b
    max_diff = (c - expected).abs().max().item()
    if not torch.allclose(c, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"MatrixAddition failed: max diff={max_diff}")

    print(f"MatrixAddition test passed (N={N}, max diff={max_diff:.2e})")


if __name__ == "__main__":
    _test_matrix_addition(N=256)