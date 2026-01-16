import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d(input, kernel, output, input_size, kernel_size, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """
    执行1维卷积
    :param input: 输入张量，形状为 (input_size,)
    :param kernel: 卷积核张量，形状为 (kernel_size,)
    :param output: 输出张量，形状为 (input_size - kernel_size + 1,)
    :param input_size: 输入张量的大小
    :param kernel_size: 卷积核的大小
    :param BLOCK_SIZE: 每个块处理的输出元素数量(注意是输出元素)
    :param BLOCK_SIZE_K: 卷积核分片大小

    Example: input_size=10, kernel_size=3, BLOCK_SIZE=4
    则第一个块 (pid=0) 处理 output[0], output[1], output[2], output[3]
    需要访问 input[0:6] (0~5)
    第二个块 (pid=1) 处理 output[4], output[5], output[6], output[7]
    需要访问 input[4:10] (4~9)
    """
    pid = tl.program_id(0)

    # 计算当前块处理的输出元素的索引范围
    output_ptr = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 计算当前块处理时需要访问的输入元素的范围
    # input_ptr = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE + kernel_size - 1)       # 注意这里是 BLOCK_SIZE + kernel_size - 1

    # Example: BLOCK_SIZE=4, kernel_size=3
    # 第一个块 (pid=0) 需要处理 output[0], output[1], output[2], output[3]
    # 需要访问 input[0:6] (0~5)
    # 不在这里读入的原因是kernel分片读取

    # 输出结果
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # kernel_size同样分片
    for k in range(tl.cdiv(kernel_size, BLOCK_SIZE_K)):
        # 读取当前的卷积核分片
        kernel_ptr = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        kernel_data = tl.load(kernel + kernel_ptr, mask=kernel_ptr < kernel_size, other=0.0)

        # 查找当前循环需要处理的输入数据范围, 并重整为(BLOCK_SIZE, BLOCK_SIZE_K) 的形状
        # 例如：BLOCK_SIZE=4, BLOCK_SIZE_K=3
        # input_ptr = [[0,1,2], [1,2,3], [2,3,4], [3,4,5]]  对应 output[0], output[1], output[2], output[3]
        input_ptr = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] + k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]

        # 读取输入数据分片
        input_data = tl.load(input + input_ptr, mask=input_ptr < input_size, other=0.0)

        # 计算卷积结果
        acc += tl.sum(input_data * kernel_data[None, :], axis=1)
    
    # 将结果写回输出张量
    tl.store(output + output_ptr, acc, mask=output_ptr < (input_size - kernel_size + 1))



# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    BLOCK_SIZE = 1024
    BLOCK_SIZE_K = 64
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d[grid](input, kernel, output, input_size, kernel_size, BLOCK_SIZE, BLOCK_SIZE_K)  # type: ignore


def _test_conv1d(input_size: int = 32768, kernel_size: int = 511, atol: float = 1e-3, rtol: float = 1e-3) -> None:
    device = torch.device("cuda")
    x = torch.randn(input_size, device=device, dtype=torch.float32)
    w = torch.randn(kernel_size, device=device, dtype=torch.float32)
    y = torch.empty(input_size - kernel_size + 1, device=device, dtype=torch.float32)

    solve(x, w, y, input_size, kernel_size)

    ref = F.conv1d(x.view(1, 1, -1), w.view(1, 1, -1)).view(-1)

    if not torch.allclose(y, ref, atol=atol, rtol=rtol):
        max_err = (y - ref).abs().max().item()
        raise AssertionError(f"Conv1D test failed: max error {max_err}, atol={atol}, rtol={rtol}.")

    print(f"Conv1D test passed (input_size={input_size}, kernel_size={kernel_size}).")


if __name__ == "__main__":
    _test_conv1d()


