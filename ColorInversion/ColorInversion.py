import torch
import triton
import triton.language as tl

@triton.jit
def color_inversion(image, width, height, BLOCK_SIZE: tl.constexpr):
    """
    执行图像反转
    :param image: 需要反转得图像张量, 注意输入是一维张量(width * height * 4, )
    :param width: 图像宽度
    :param height: 图像高度
    :param BLOCK_SIZE: 每个线程块处理的像素数量
    """

    pid = tl.program_id(0)

    # 输入的input数组元素是uint8, 因此指针类型是tl.uint8

    # 如果正常读取逐个加载太慢了, 每个元素是8bit(0~255)即uint8
    # 因此可以4个一组进行处理, 合并成uint32进行处理

    # 转为uint32指针
    image = image.to(tl.pointer_type(tl.uint32))

    # 更改前, 移动一个元素跨8个bit, 更改后, 移动一个元素跨32个bit
    # 相当于把原来[width * height * 4]个uint8元素变成[width * height]个uint32元素

    # 构造读入块指针
    input = tl.make_block_ptr(
        base=image,
        shape=(height * width,),
        strides=(1,),
        block_shape=(BLOCK_SIZE,),
        offsets=(pid * BLOCK_SIZE,),
        order=(0,)
    )

    # 构造写出块指针
    output = tl.make_block_ptr(
        base=image,
        shape=(height * width,),
        strides=(1,),
        block_shape=(BLOCK_SIZE,),
        offsets=(pid * BLOCK_SIZE,),
        order=(0,)
    )


    # 加载当前块数据[width * height, ], 每个元素是uint32
    pixel_data = tl.load(input, boundary_check=(0, ), padding_option="zero")

    # 反转颜色通道 (R, G, B), 保持Alpha通道不变, 位运算反转
    # 0x00FFFFFF 对应小端序下的 R, G, B 通道 (0xFF, 0xFF, 0xFF, 0x00)
    # XOR运算: x^0xFF = ~x (取反)
    inverted_pixel_data = pixel_data ^ 0x00FFFFFF   # 逐个元素进行位运算反转

    # 写回反转数据
    tl.store(output, inverted_pixel_data, boundary_check=(0, ))


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    color_inversion[grid](image, width, height, BLOCK_SIZE)    # type:ignore

def main(width: int = 1024, height: int = 128):
    # Generate random RGBA data on GPU (uint8), flatten to match kernel expectation
    rgba = torch.randint(0, 256, (height * width * 4,), device="cuda", dtype=torch.uint8)

    expected = rgba.view(-1, 4).clone()
    expected[:, :3] = 255 - expected[:, :3]
    expected = expected.reshape(-1)

    solve(rgba, width, height)
    torch.cuda.synchronize()

    if torch.equal(rgba.cpu(), expected.cpu()):
        print("颜色反转成功!")
    else:
        print("颜色反转失败!")


if __name__ == "__main__":
    main()
    