import torch
from torch import nn, einsum
from timm.models.layers import DropPath, trunc_normal_
from typing import Optional, Callable, Any
from model.nets.SS4D import SS4D
from model.mamba2.mamba import *
from thop import profile
from model.nets.TSIM import *

class T_Mamba(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_dim: int = 0,
            patch_size=1,
            drop_path: float = 0,
            norm_layer = "LN",  # "BN", "LN2D"
            channel_first=False,
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            patch_norm=True,
            ssm_init="v0",
            forward_type="v0",
            patchembed_version: str = "v1",
            SS4D: type = SS4D
    ):
        super().__init__()

        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.SS4D = SS4D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            # ==========================
            forward_type=forward_type,
            channel_first=channel_first,
        )
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(hidden_dim, hidden_dim, patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        self.pre_post = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        

    @staticmethod
    def _make_patch_embed(dim, embed_dim, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                          channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    def forward(self, pre, post):
        diff = abs(post-pre)
        x = diff
        diff = self.down(diff)
        diff = self.SS4D(self.patch_embed(diff))
        diff = diff.permute(0, 3, 1, 2).contiguous()
        diff = self.up(diff)

        diff = diff + x

        pre = self.sigmoid(self.pre_post(pre))
        post = self.sigmoid(self.pre_post(post))

        diff = diff * pre * post

        return diff


if __name__ == '__main__':
    x = torch.randn(1, 64, 128, 128)
    y = torch.randn(1, 64, 128, 128)
    model = T_Mamba(64,64)  # 创建 ScConv 模型
    print(model(x, y).shape)  # 打印模型输出的形状
    model.cuda()
    flops, params = profile(model, (x.cuda(), y.cuda()))
    print('flops: %.2f GFLOPS, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))



