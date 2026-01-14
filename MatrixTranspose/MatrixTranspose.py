import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, 
                            stride_ir, stride_ic, stride_or, stride_oc,
                            BLOCK_SIZE_R: tl.constexpr,
                            BLOCK_SIZE_C: tl.constexpr):
    """
    实现矩阵转置
    :param input: (rows, cols)
    :param output: (cols, rows)
    :param stride_ir, stride_ic: 输入矩阵在行和列维度上移动一个index实际存储位置需要移动的步长
    :param stride_or, stride_oc: 输出矩阵在行和列维度上移动一个index实际存储位置需要移动的步长
    """
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    # 加载块指针
    # 列优先读取, 行优先存储
    input_block_ptr_trans = tl.make_block_ptr(
        base=input,
        shape=(cols, rows),              # 视角转置
        strides=(stride_ic, stride_ir),  # 步长转置：第一维跳列步长，第二维跳行步长
        offsets=(col_idx * BLOCK_SIZE_C, row_idx * BLOCK_SIZE_R), 
        block_shape=(BLOCK_SIZE_C, BLOCK_SIZE_R), # 目标输出形状
        order=(0, 1)                     # 强制列优先读取，实现读取时转置
    )

    output_block_ptr = tl.make_block_ptr(
        base=output,
        shape=(cols, rows),
        strides=(stride_or, stride_oc),
        offsets=(col_idx * BLOCK_SIZE_C, row_idx * BLOCK_SIZE_R),
        block_shape=(BLOCK_SIZE_C, BLOCK_SIZE_R),
        order=(1, 0)         # 行优先存储
    )

    # 加载输入 ([BLOCK_SIZE_C, BLOCK_SIZE_R])
    input_tile = tl.load(input_block_ptr_trans, boundary_check=(0, 1), padding_option="zero")
    
    tl.store(output_block_ptr, input_tile, boundary_check=(0, 1))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1
    BLOCK_SIZE_R, BLOCK_SIZE_C = 16, 16

    grid = (triton.cdiv(rows, BLOCK_SIZE_R), triton.cdiv(cols, BLOCK_SIZE_C))
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc,
        BLOCK_SIZE_R , BLOCK_SIZE_C                     # type: ignore
    )

if __name__ == "__main__":
    rows, cols = 128, 256
    input = torch.randn((rows, cols), device='cuda', dtype=torch.float32)
    output = torch.zeros((cols, rows), device='cuda', dtype=torch.float32)

    solve(input, output, rows, cols)

    # 验证结果
    expected = input.t()
    if torch.allclose(output, expected):
        print("矩阵转置成功!")
    else:
        print("矩阵转置失败!")