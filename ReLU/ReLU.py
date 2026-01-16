import torch
import triton
import triton.language as tl


@triton.jit
def relu(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    """ReLU activation kernel"""
    pid = tl.program_id(0)

    start_idx = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = start_idx + offsets

    # 读取数据
    input_data = tl.load(input + indices, mask=indices < n_elements, other=0.0)

    # 执行ReLU操作
    ouput_data = tl.maximum(input_data, 0.0)

    # 写回结果
    tl.store(output + indices, ouput_data, mask=indices < n_elements)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu[grid](input, output, N, BLOCK_SIZE)       # type: ignore


def _test_relu(N: int = 10_000):
    device = torch.device("cuda")
    x = torch.randn(N, device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    solve(x, y, N)

    ref = torch.maximum(x, torch.zeros_like(x))
    if not torch.equal(y, ref):
        diff = (y - ref).abs().max().item()
        raise AssertionError(f"ReLU failed: max diff {diff}")
    print(f"ReLU test passed for N={N}")


if __name__ == "__main__":
    _test_relu()


