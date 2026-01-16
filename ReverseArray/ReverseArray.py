import torch
import triton
import triton.language as tl


@triton.jit
def reverse_array(input, N, BLOCK_SIZE: tl.constexpr):
    """反转数组"""
    pid = tl.program_id(0)

    # 当前实例需要读取头尾各一块数据
    head_start_ptr = pid * BLOCK_SIZE
    tail_start_ptr = tl.maximum(0, N - (pid + 1) * BLOCK_SIZE)      # 保护N<BLOCK_SIZE的情况, 防止越界

    if head_start_ptr <= (N+1) // 2:
        # 只处理前半部分
        offset = tl.arange(0, BLOCK_SIZE)
        head_indices = head_start_ptr + tl.arange(0, BLOCK_SIZE)
        tail_indices = tail_start_ptr + tl.arange(0, BLOCK_SIZE)

        # 读取数据
        head_data = tl.load(input + head_indices, mask=head_indices < N, other=0)
        tail_data = tl.load(input + tail_indices, mask=tail_indices < N, other=0)

        # 计算当前的head_data和tail_data需要写入的位置
        head_write_indices = N - 1 - head_indices
        tail_write_indices = N - 1 - tail_indices
        head_mask = (head_write_indices < N) * (head_write_indices >=0)
        tail_mask = (tail_write_indices < N) * (tail_write_indices >=0)

        # 写回数据
        tl.store(input + head_write_indices, head_data, mask=head_mask)
        tl.store(input + tail_write_indices, tail_data, mask=tail_mask)

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    num_total_blocks = triton.cdiv(N, BLOCK_SIZE)
    grid = (triton.cdiv(num_total_blocks, 2),)

    reverse_array[grid](input, N, BLOCK_SIZE)  # type: ignore


def _test_reverse(N: int = 25000000):
    device = torch.device("cuda")
    x = torch.arange(N, device=device, dtype=torch.float32)
    y = x.clone()

    solve(y, N)

    ref = torch.flip(x, dims=[0])
    if not torch.equal(y, ref):
        diff = (y - ref).abs().max().item()
        raise AssertionError(f"Reverse failed: max diff {diff}")
    print(f"Reverse test passed for N={N}")


if __name__ == "__main__":
    _test_reverse()
