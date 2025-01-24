import torch
import torch.nn as nn
from itertools import pairwise, chain


class EncoderDecoder(nn.Module):
    def __init__(self, input_channels: int = 3,
                 output_channels: int = 1,
                 dropout: float | None = None,
                 channels: list[int] | None = None, **kwargs) -> None:
        super().__init__()

        if channels is None:
            channels = [64, 128, 256]

        # Encoder
        self.encoder = nn.Sequential(
            *list(chain.from_iterable(
                self._encoder_block(*pair, dropout)
                for pair in pairwise((input_channels, *channels[:-1]))))
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(*channels[-2:], kernel_size=3, padding=1),
            *self._relu_drop(dropout),
        )

        # Decoder
        self.decoder = nn.Sequential(
            *list(chain.from_iterable(
                self._decoder_block(*pair, dropout)
                for pair in pairwise(channels[::-1]))),
            nn.Conv2d(channels[0], output_channels, kernel_size=1),
        )

    def _relu_drop(self, dropout: float | None = None) -> tuple[nn.Module]:
        if dropout is None:
            return (nn.ReLU(),)
        return (nn.ReLU(), nn.Dropout(p=dropout))

    def _encoder_block(self, in_channels: int, out_channels: int,
                       dropout: float | None = None) -> list[nn.Module]:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            *self._relu_drop(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

    def _decoder_block(self, in_channels: int, out_channels: int,
                       dropout: float | None = None) -> list[nn.Module]:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            *self._relu_drop(dropout),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = EncoderDecoder(input_channels=3, output_channels=1, dropout=0.1)
    print(model)

    input_tensor = torch.randn(1, 3, 32, 32)
    output = model(input_tensor)
    print(f"{input_tensor.shape=} - {output.shape=}")

    input_tensor = torch.randn(1, 3, 256, 384)
    output = model(input_tensor)
    print(f"{input_tensor.shape=} - {output.shape=}")
