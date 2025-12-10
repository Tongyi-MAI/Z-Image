"""AutoencoderKL implementation compatible with diffusers weights."""

# Modified from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AutoencoderKLOutput:
    sample: torch.Tensor


class AutoencoderConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __getattr__(self, name):
        return self.__dict__.get(name)


def swish(x):
    return x * torch.sigmoid(x)


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        eps=1e-6,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=groups, num_channels=out_channels, eps=eps, affine=True
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.nonlinearity = swish

        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.conv_shortcut = None

    def forward(self, input_tensor, temb=None):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / 1.0
        return output_tensor


class Attention(nn.Module):
    def __init__(self, in_channels, heads=1, dim_head=None, groups=32, eps=1e-6):
        super().__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.group_norm = nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True
        )

        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_k = nn.Linear(in_channels, in_channels)
        self.to_v = nn.Linear(in_channels, in_channels)
        self.to_out = nn.ModuleList([nn.Linear(in_channels, in_channels)])

    def forward(self, hidden_states):
        b, c, h, w = hidden_states.shape
        residual = hidden_states
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(b, c, -1).transpose(1, 2)  # (B, H*W, C)

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        import torch.nn.functional as F

        hidden_states = F.scaled_dot_product_attention(query, key, value)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(b, c, h, w)

        return residual + hidden_states


class Downsample2D(nn.Module):
    def __init__(self, channels, with_conv=True, out_channels=None, padding=1):
        super().__init__()
        out_channels = out_channels or channels
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(
                channels, out_channels, kernel_size=3, stride=2, padding=padding
            )

    def forward(self, hidden_states):
        if self.with_conv:
            return self.conv(hidden_states)
        else:
            return torch.nn.functional.avg_pool2d(
                hidden_states, kernel_size=2, stride=2
            )


class Upsample2D(nn.Module):
    def __init__(self, channels, with_conv=True, out_channels=None):
        super().__init__()
        out_channels = out_channels or channels
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(
                channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, hidden_states):
        hidden_states = torch.nn.functional.interpolate(
            hidden_states, scale_factor=2.0, mode="nearest"
        )
        if self.with_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=1,
        resnet_eps=1e-6,
        resnet_groups=32,
        add_downsample=True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(in_c, out_channels, eps=resnet_eps, groups=resnet_groups)
            )
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        with_conv=True,
                        out_channels=out_channels,
                        padding=0,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                pad = (0, 1, 0, 1)
                hidden_states = torch.nn.functional.pad(
                    hidden_states, pad, mode="constant", value=0
                )
                hidden_states = downsampler(hidden_states)

        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=1,
        resnet_eps=1e-6,
        resnet_groups=32,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(in_c, out_channels, eps=resnet_eps, groups=resnet_groups)
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, with_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UNetMidBlock2D(nn.Module):
    def __init__(
        self, in_channels, resnet_eps=1e-6, resnet_groups=32, attention_head_dim=None
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels, in_channels, eps=resnet_eps, groups=resnet_groups
                ),
                ResnetBlock2D(
                    in_channels, in_channels, eps=resnet_eps, groups=resnet_groups
                ),
            ]
        )
        self.attentions = nn.ModuleList(
            [Attention(in_channels, heads=1, groups=resnet_groups, eps=resnet_eps)]
        )

    def forward(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        for attn in self.attentions:
            hidden_states = attn(hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        double_z=True,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )

        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, block_out_channel in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channel
            is_final_block = i == len(block_out_channels) - 1

            block = DownEncoderBlock2D(
                input_channel,
                output_channel,
                num_layers=layers_per_block,
                resnet_groups=norm_num_groups,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(block)

        self.mid_block = UNetMidBlock2D(
            block_out_channels[-1],
            resnet_groups=norm_num_groups,
        )

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], conv_out_channels, 3, padding=1
        )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1
        )

        self.mid_block = UNetMidBlock2D(
            block_out_channels[-1],
            resnet_groups=norm_num_groups,
        )

        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, block_out_channel in enumerate(reversed_block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channel
            is_final_block = i == len(block_out_channels) - 1
            block = UpDecoderBlock2D(
                input_channel,
                output_channel,
                num_layers=layers_per_block + 1,
                resnet_groups=norm_num_groups,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(block)

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        force_upcast: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.config = AutoencoderConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            latent_channels=latent_channels,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
        )

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

        self.quant_conv = (
            nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(latent_channels, latent_channels, 1)
            if use_post_quant_conv
            else None
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> AutoencoderKLOutput:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return AutoencoderKLOutput(sample=dec)
