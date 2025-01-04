import torch
from torch import nn, einsum
from thop import profile


class TSIM(nn.Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim

        self.pre_q = nn.Sequential(nn.Conv2d(self.in_channels, self.dim, kernel_size=1),
                                   nn.BatchNorm2d(self.dim),
                                   nn.ReLU(inplace=True)).cuda()
        self.pre_k = nn.Sequential(nn.Conv2d(self.in_channels, self.dim, kernel_size=1),
                                   nn.BatchNorm2d(self.dim),
                                   nn.ReLU(inplace=True)).cuda()
        self.pre_v = nn.Sequential(nn.Conv2d(self.in_channels, self.dim, kernel_size=1),
                                   nn.BatchNorm2d(self.dim),
                                   nn.ReLU(inplace=True)).cuda()

        self.priors = nn.AdaptiveAvgPool2d(output_size=(10, 10)).cuda()
        self.softmax = nn.Softmax(dim=-1)

        self.conv = nn.Conv2d(self.dim, self.in_channels, kernel_size=1).cuda()

        self.pre_post = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        ).cuda()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pre, post):
        pre = pre.cuda()
        post = post.cuda()
        pre_0 = self.pre_post(pre)
        post_0 = self.pre_post(post)
        b, c, h, w = pre.size()

        pre_q = self.priors(self.pre_q(pre))[:, :, 1:-1, 1:-1].reshape(b, self.dim, -1).permute(0, 2, 1)
        pre_k = self.pre_k(pre).view(b, -1, w * h)
        pre_v = self.pre_v(pre).view(b, -1, w * h)

        post_q = self.priors(self.pre_q(post))[:, :, 1:-1, 1:-1].reshape(b, self.dim, -1).permute(0, 2, 1)
        post_k = self.pre_k(post).view(b, -1, w * h)
        post_v = self.pre_v(post).view(b, -1, w * h)

        # print(pre_q.shape, pre_k.shape, pre_v.shape, post_k.shape)

        pre_energy = torch.bmm(pre_q, post_k)
        pre_attention = self.softmax(pre_energy)
        pre = torch.mul(pre_v, pre_attention)
        pre = pre.view(b, self.dim, h, w)

        post_energy = torch.bmm(post_q, pre_k)
        post_attention = self.softmax(post_energy)
        post = torch.mul(post_v, post_attention)
        post = post.view(b, self.dim, h, w)

        diff = abs(post - pre)
        diff = self.conv(diff)

        G = self.sigmoid(diff)
        pre = pre_0 * G
        post = post_0 * (1 - G)
        return pre, post


if __name__ == "__main__":
    model = TSIM(in_channels=256, dim=64)
    model.eval()
    pre = torch.randn(1, 256, 32, 32)
    post = torch.randn(1, 256, 32, 32)
    with torch.no_grad():
        pre, post = model.forward(pre,post)
    print(pre.shape, post.shape)
    model.cuda()
    flops, params = profile(model, (pre.cuda(), post.cuda()))
    print('flops: %.2f GFLOPS, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))