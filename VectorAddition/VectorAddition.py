import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    # 实现矩阵a + b = c
    # :param a: (n_element, )
    # :param b: (n_element, )
    # :param c: (n_element, )
    # :param BLOCKSIZE: 每次迭代加载的特征维度块大小
    
    a_tile_ptr = tl.program_id(0)       # 将a在n_element维度进行划分

    # 创建block指针, 当前实例处理的block
    a_block_ptr = tl.make_block_ptr(
        base=a,       # 初始指针从a的第一个元素开始
        shape=(n_elements, ),
        strides=(1,),   # 两个维度上移动一个位置在实际内存上需要移动的步幅
        offsets=(a_tile_ptr*BLOCK_SIZE,),     # 当前实例处理的偏移
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )

    b_block_ptr = tl.make_block_ptr(
        base=b,       # 初始指针从b的第一个元素开始
        shape=(n_elements, ),
        strides=(1,),   # 两个维度上移动一个位置在实际内存上需要移动的步幅
        offsets=(a_tile_ptr*BLOCK_SIZE,),     # 当前实例处理的偏移
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )

    c_block_ptr = tl.make_block_ptr(
        base=c,       # 初始指针从c的第一个元素开始
        shape=(n_elements, ),
        strides=(1,),   # 两个维度上移动一个位置在实际内存上需要移动的步幅
        offsets=(a_tile_ptr*BLOCK_SIZE,),     # 当前实例处理的偏移
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )

    # 初始化当前实例输出的C_tile
    Ct = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    # 加载当前实例处理的A_tile和B_tile
    At = tl.load(a_block_ptr, boundary_check=(0,), padding_option="zero")
    Bt = tl.load(b_block_ptr, boundary_check=(0,), padding_option="zero")

    Ct = At + Bt

    # 写回Ct
    tl.store(c_block_ptr, Ct, boundary_check=(0,))


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)

if __name__ == "__main__":
    N = 1 << 20
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    c = torch.empty(N, device='cuda', dtype=torch.float32)

    solve(a, b, c, N)

    # 验证结果
    torch.testing.assert_close(c, a + b)
    if torch.allclose(c, a + b):
        print("Result is correct!")
    print("Result vector C shape:", c.shape)