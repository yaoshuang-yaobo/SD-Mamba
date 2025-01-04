import torch
from torch import nn, einsum
from collections import OrderedDict
from model.nets import *
import torch.nn.functional as F
from thop import profile


class S_Net(nn.Module):
    def __init__(self, in_channels_pre, in_channels_post, nclass):
        super(S_Net, self).__init__()
        self.dim = 64
        self.in_pre = in_channels_pre
        self.in_post = in_channels_post
        self.nclass = nclass
        self.inchannels = [64, 128, 256, 512]


        self.TSIM_1 = TSIM(self.inchannels[0], self.dim).cuda()
        self.TSIM_2 = TSIM(self.inchannels[1], self.dim).cuda()
        self.TSIM_3 = TSIM(self.inchannels[2], self.dim).cuda()
        self.TSIM_4 = TSIM(self.inchannels[3], self.dim).cuda()

        self.MAPC_1 = MAPC(self.in_pre, self.in_post, self.inchannels[0]).cuda()
        self.MAPC_2 = MAPC(self.inchannels[0], self.inchannels[0], self.inchannels[1]).cuda()
        self.MAPC_3 = MAPC(self.inchannels[1], self.inchannels[1], self.inchannels[2]).cuda()
        self.MAPC_4 = MAPC(self.inchannels[2], self.inchannels[2], self.inchannels[3]).cuda()
        # Bottleneck

        self.t_mamba = T_Mamba(self.inchannels[0],self.inchannels[1],self.inchannels[2],self.inchannels[3], self.dim, self.nclass).cuda()

    def forward(self, pre, post):
        pre = pre.cuda()
        post = post.cuda()
        pre1, post1 = self.MAPC_1(pre, post)
        pre1, post1 = self.TSIM_1(pre1, post1)

        pre2, post2 = self.MAPC_2(pre1, post1)
        pre2, post2 = self.TSIM_2(pre2, post2)


        pre3, post3 = self.MAPC_3(pre2, post2)
        pre3, post3 = self.TSIM_3(pre3, post3)

        pre4, post4 = self.MAPC_4(pre3, post3)
        pre4, post4 = self.TSIM_4(pre4, post4)

        change = self.t_mamba(pre1, post1, pre2, post2, pre3, post3, pre4, post4)

        outputs = [change]

        return change


if __name__ == '__main__':
    pre = torch.randn(1, 3, 256, 256)
    post = torch.randn(1, 1, 256, 256)
    model = S_Net(in_channels_pre=3, in_channels_post=1, nclass=2)  # 创建 ScConv 模型
    out = model(pre, post)
    print(out.shape)  # 打印模型输出的形状
    model.cuda()
    flops, params = profile(model, (pre.cuda(), post.cuda()))
    print('flops: %.2f GFLOPS, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
