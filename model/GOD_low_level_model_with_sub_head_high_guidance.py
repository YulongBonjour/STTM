from diffusers.models.vae import Decoder
import torch
import torch.nn as nn
from functools import partial
class Voxel2StableDiffusionModel(torch.nn.Module):
    def __init__(self, voxel_num_dict= {}, h=4096, n_blocks=4,
                 input_n_blocks=1,ups_mode='4x', norm_type='ln', act_first=False,):
        super().__init__()
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm,
                                                                                              normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        # assert in_dim==h
        self.input_proj = {}
        self.input_res_block = {}
        self.input_n_blocks=input_n_blocks
        self.null_high_embeds = nn.Parameter(torch.randn(h))
        self.null_high_embeds_1 = nn.Parameter(torch.randn(h))
        for k, v in voxel_num_dict.items():
            self.input_proj['head{}'.format(k)] = nn.Sequential(
                nn.Linear(v,h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.65),
                )
            self.input_res_block['head{}'.format(k)]=nn.ModuleList([nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(0.15)
                ) for _ in range(input_n_blocks)
            ])
        self.input_proj=nn.ModuleDict(self.input_proj)
        self.input_res_block=nn.ModuleDict(self.input_res_block)
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15)
            ) for _ in range(n_blocks)
        ])
        self.ups_mode = ups_mode
        self.merge_block= nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15)
            )
        if ups_mode == '4x':
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 64)

            self.upsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )
        if ups_mode == '8x':  # prev best
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 256)

            self.upsampler = Decoder(
                in_channels=256,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=1,
            )

        if ups_mode == '16x':
            self.lin1 = nn.Linear(h, 8192, bias=False)
            self.norm = nn.GroupNorm(1, 512)

            self.upsampler = Decoder(
                in_channels=512,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D",
                                "UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256, 512],
                layers_per_block=1,
            )

    def get_subject_proj(self, x, sub_id):
        x = self.input_proj['head{}'.format(sub_id)](x)
        residual = x
        for i in range(self.input_n_blocks):
            x = self.input_res_block['head{}'.format(sub_id)][i](x)
            x += residual
            residual = x
        return x
    def freeze_pretrained(self):
        self.requires_grad_(False)
        self.input_proj.requires_grad_(True)
        self.input_res_block.requires_grad_(True)
    def forward(self, x,sub_id,high_feature=None,only_high_features=False):
        if only_high_features:
            x = self.null_high_embeds_1
            x = x + high_feature
            residual = x
            x = self.merge_block(x)
            x = residual + x
        else:
            if high_feature == None:
                high_feature = self.null_high_embeds
            x = self.get_subject_proj(x, sub_id)
            residual = x
            for res_block in self.mlp:
                x = res_block(x)
                x = x + residual
                residual = x
            x = x.reshape(len(x), -1)

            x = x + high_feature
            residual = x
            x = self.merge_block(x)
            x = residual + x

        x = self.lin1(x)  # bs, 4096

        if self.ups_mode == '4x':
            side = 16
        if self.ups_mode == '8x':
            side = 8
        if self.ups_mode == '16x':
            side = 4
        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, side, side).contiguous())
        return self.upsampler(x)
