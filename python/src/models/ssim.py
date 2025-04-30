from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def gaussian(window_size: int, sigma: float):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def uniform(window_size):
    uniform = torch.ones(window_size)
    return uniform / uniform.sum()


def create_window(window_size: int, channel: int, sigma: float, use_gaussian):
    _win = uniform(window_size)
    if use_gaussian:
        _win = gaussian(window_size, sigma)
        print("Gaussian window is created")
    else:
        print("Uniform window is created")

    _1D_window = _win.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(
    *,
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool,
    max_value: float,
):
    assert img1.ndim == img2.ndim == 4  # batch, channel, x, and y

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = (max_value * 0.01) ** 2
    C2 = (max_value * 0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


class SSIM(torch.nn.Module):
    def __init__(
        self,
        max_value: float,
        scale: float,
        bias: float,
        window_size: int = 11,
        sigma: float = 1.5,
        size_average: bool = True,
        use_gauss: bool = True,
        offset: float = 0.0,
    ):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.max_value = max_value
        self.scale = scale
        self.bias = bias
        self.use_gauss = use_gauss
        self.offset = offset

        self.channel = 1
        self.window = create_window(
            window_size=self.window_size,
            sigma=self.sigma,
            channel=self.channel,
            use_gaussian=self.use_gauss,
        )

    def scale_img(self, img: torch.Tensor) -> torch.Tensor:
        ret = (img - self.bias) / self.scale + self.offset
        return ret

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        assert img1.ndim == img2.ndim == 4  # batch, channel, x, and y
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            self.channel = channel
            window = create_window(
                window_size=self.window_size,
                sigma=self.sigma,
                channel=self.channel,
                use_gaussian=self.use_gauss,
            )

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window

        return _ssim(
            img1=self.scale_img(img1),
            img2=self.scale_img(img2),
            window=self.window,
            window_size=self.window_size,
            channel=self.channel,
            size_average=self.size_average,
            max_value=self.max_value,
        )
