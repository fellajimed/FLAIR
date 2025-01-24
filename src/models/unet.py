import torch
import torch.nn as nn
from itertools import pairwise


class UNet(nn.Module):
    def __init__(self, input_channels: int = 3,
                 output_channels: int = 1,
                 batchnorm: bool = False,
                 dropout: float | None = None,
                 channels: list[int] | None = None, **kwargs) -> None:
        super().__init__()

        if channels is None:
            channels = [64, 128, 256, 512]

        # the last element of the list will be the bottleneck
        self.encoders = [self.conv_block(input_channels, channels[0],
                                         batchnorm, dropout)]
        self.upconvs = []
        self.decoders = []

        for (c1, c2) in pairwise(channels):
            self.encoders.append(
                nn.Sequential(nn.MaxPool2d(2),
                              self.conv_block(c1, c2, batchnorm, dropout)))
            self.upconvs.insert(0, nn.ConvTranspose2d(c2, c1, kernel_size=2,
                                                      stride=2))
            self.decoders.insert(0, self.conv_block(c2, c1, batchnorm,
                                                    dropout))

        self.encoders = nn.ModuleList(self.encoders)
        self.upconvs = nn.ModuleList(self.upconvs)
        self.decoders = nn.ModuleList(self.decoders)

        self.final_conv = nn.Conv2d(channels[0], output_channels,
                                    kernel_size=1)

    def conv_block(self, in_channels: int, out_channels: int,
                   batchnorm: bool = False,
                   dropout: float | None = None) -> nn.Module:
        """
        Convolutional Block: Two convolutional layers
        with ReLU and batch normalization.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            *self._relu_drop_batchnorm(dropout, batchnorm, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            *self._relu_drop_batchnorm(dropout, batchnorm, out_channels),
        )

    def _relu_drop_batchnorm(self, dropout: float | None,
                             batchnorm: bool, out_channels: int
                             ) -> list[nn.Module]:
        _layers = [nn.ReLU()]
        if dropout is not None:
            _layers.append(nn.Dropout(p=dropout))
        if batchnorm:
            _layers.append(nn.BatchNorm2d(out_channels))
        return _layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        encs = []
        for encoder in self.encoders:
            y = encoder(y)
            encs.insert(0, y)

        dec = encs.pop(0)
        for enc, upconv, decoder in zip(encs, self.upconvs, self.decoders):
            dec = decoder(torch.cat((enc, upconv(dec)), dim=1))

        return self.final_conv(dec)


if __name__ == "__main__":
    model = UNet(input_channels=3, output_channels=1, dropout=0.1)
    print(model)

    input_tensor = torch.randn(1, 3, 32, 32)
    output = model(input_tensor)
    print(f"{input_tensor.shape=} - {output.shape=}")

    input_tensor = torch.randn(1, 3, 256, 384)
    output = model(input_tensor)
    print(f"{input_tensor.shape=} - {output.shape=}")
