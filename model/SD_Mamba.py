
from model.S_Net import *
from model.D_Net import *
from model.S_Net_IBCM import *
from model.S_Net_TSCA import *
from model.S_Net_ConMB import *
from model.S_Net_CroMB import *


class SD_Mamba(nn.Module):
    def __init__(self, in_channels_pre, in_channels_post, nclass):
        super(SD_Mamba, self).__init__()
        self.in_channels_pre = in_channels_pre
        self.in_channels_post = in_channels_post
        self.nclass = nclass

        self.s_net = S_Net(in_channels_pre=self.in_channels_pre, in_channels_post=self.in_channels_post, nclass=self.nclass).cuda()
        self.d_net = D_Net(in_dim=self.nclass, nclass_pre=self.in_channels_pre, nclass_post=self.in_channels_post).cuda()

    def forward(self, pre, post):

        change = self.s_net(pre, post)
        pre1, post1 = self.d_net(change)

        out = []
        out.append(change)


        return out, pre1, post1


if __name__ == '__main__':

    model = SD_Mamba(in_channels_pre=3, in_channels_post=1, nclass=2)
    model.eval()
    pre = torch.randn(1, 3, 256, 256)
    post = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        x, pre, post = model.forward(pre, post)
    print(x[0].shape,pre.shape, post.shape)
    flops, params = profile(model, (pre, post,))
    print('flops: %.2f GFLOPS, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

