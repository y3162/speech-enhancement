from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchaudio.transforms import Resample


def feature_loss(
    fmap_r: list[list[torch.Tensor]],
    fmap_g: list[list[torch.Tensor]],
) -> torch.Tensor:
    loss = 0
    for real_maps, fake_maps in zip(fmap_r, fmap_g):
        for real, fake in zip(real_maps, fake_maps):
            loss += torch.mean(torch.abs(real - fake))
    return loss * 2


def lsgan_discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> torch.Tensor:
    loss = 0
    for real, fake in zip(disc_real_outputs, disc_generated_outputs):
        loss = loss + torch.mean((1 - real) ** 2) + torch.mean(fake ** 2)
    return loss


def lsgan_generator_loss(disc_outputs: list[torch.Tensor]) -> torch.Tensor:
    loss = 0
    for fake in disc_outputs:
        loss = loss + torch.mean((1 - fake) ** 2)
    return loss


def _pair_outputs(discriminators, y: torch.Tensor, y_hat: torch.Tensor):
    y_d_rs = []
    y_d_gs = []
    fmap_rs = []
    fmap_gs = []
    for disc in discriminators:
        y_d_r, fmap_r = disc(y)
        y_d_g, fmap_g = disc(y_hat)
        y_d_rs.append(y_d_r)
        fmap_rs.append(fmap_r)
        y_d_gs.append(y_d_g)
        fmap_gs.append(fmap_g)
    return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def conv2d_padding(
    kernel_size: tuple[int, int],
    dilation: tuple[int, int] = (1, 1),
) -> tuple[int, int]:
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class DiscriminatorCQT(nn.Module):
    def __init__(self, hop_length: int, n_octaves: int, bins_per_octave: int) -> None:
        super().__init__()
        filters = 128
        kernel_size = (3, 9)
        dilations = [1, 2, 4]
        stride = (1, 2)
        in_channels = 2
        fs = 16000
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        from nnAudio import features

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=fs * 2,
            hop_length=hop_length,
            n_bins=bins_per_octave * n_octaves,
            bins_per_octave=bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )
        self.conv_pres = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=conv2d_padding(kernel_size),
                )
                for _ in range(n_octaves)
            ]
        )
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Conv2d(
                in_channels,
                filters,
                kernel_size=kernel_size,
                padding=conv2d_padding(kernel_size),
            )
        )
        for dilation in dilations:
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        filters,
                        filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=(dilation, 1),
                        padding=conv2d_padding(kernel_size, (dilation, 1)),
                    )
                )
            )
        square_kernel = (kernel_size[0], kernel_size[0])
        self.convs.append(
            weight_norm(
                nn.Conv2d(
                    filters,
                    filters,
                    kernel_size=square_kernel,
                    padding=conv2d_padding(square_kernel),
                )
            )
        )
        self.conv_post = weight_norm(
            nn.Conv2d(
                filters,
                1,
                kernel_size=square_kernel,
                padding=conv2d_padding(square_kernel),
            )
        )
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=fs, new_freq=fs * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        x = self.resample(x)
        z = self.cqt_transform(x)
        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)
        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))
        latent_z = []
        for i in range(self.n_octaves):
            start = i * self.bins_per_octave
            end = (i + 1) * self.bins_per_octave
            latent_z.append(self.conv_pres[i](z[:, :, :, start:end]))
        latent_z = torch.cat(latent_z, dim=-1)
        for conv in self.convs:
            latent_z = self.activation(conv(latent_z))
            fmap.append(latent_z)
        latent_z = self.conv_post(latent_z)
        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hop_lengths = [512, 256, 256]
        n_octaves = [8, 8, 8]
        bins_per_octaves = [12, 24, 36]
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    hop_length=hop_length,
                    n_octaves=n_octave,
                    bins_per_octave=bins_per_octave,
                )
                for hop_length, n_octave, bins_per_octave in zip(
                    hop_lengths, n_octaves, bins_per_octaves
                )
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        return _pair_outputs(self.discriminators, y, y_hat)


class DiscriminatorR(nn.Module):
    def __init__(self, resolution: list[int]) -> None:
        super().__init__()
        if len(resolution) != 3:
            raise ValueError(f"MRD layer requires list with len=3, got {resolution}")
        self.resolution = resolution
        self.lrelu_slope = 0.1
        channels = 32
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))
                ),
                weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length, device=x.device),
            center=False,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        return torch.norm(x, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        x = self.spectrogram(x).unsqueeze(1)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(resolution)
                for resolution in [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        return _pair_outputs(self.discriminators, y, y_hat)


class SEMambaPPDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cqt = MultiScaleSubbandCQTDiscriminator()
        self.mrd = MultiResolutionDiscriminator()

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> SimpleNamespace:
        y_dq_r, y_dq_g, fmap_q_r, fmap_q_g = self.cqt(y, y_hat)
        y_dr_r, y_dr_g, fmap_r_r, fmap_r_g = self.mrd(y, y_hat)
        return SimpleNamespace(
            cqt_real=y_dq_r,
            cqt_fake=y_dq_g,
            cqt_fmap_real=fmap_q_r,
            cqt_fmap_fake=fmap_q_g,
            mrd_real=y_dr_r,
            mrd_fake=y_dr_g,
            mrd_fmap_real=fmap_r_r,
            mrd_fmap_fake=fmap_r_g,
        )
