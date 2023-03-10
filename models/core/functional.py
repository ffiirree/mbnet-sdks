import torch

__all__ = ['make_divisible', 'get_gaussian_kernel1d', 'get_gaussian_kernel2d', 'get_3x3_gaussian_weight2d',]


def make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor

    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor

    return new_value


def get_3x3_gaussian_weight2d(sigma: torch.Tensor):
    assert(len(sigma.size()) == 1)
    x = -0.5 * (-1 / sigma).pow(2)

    m = torch.tensor([[[
        [2, 1, 2],
        [1, 0, 1],
        [2, 1, 2]
    ]]], device=sigma.device).repeat(x.shape[0], 1, 1, 1)

    k = torch.exp(m * x.view(x.shape[0], 1, 1, 1))
    return k 


def get_gaussian_kernel1d(kernel_size, sigma: float):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf


def get_gaussian_kernel2d(kernel_size, sigma: float):
    kernel1d = get_gaussian_kernel1d(kernel_size, sigma)
    return torch.mm(kernel1d[:, None], kernel1d[None, :])
