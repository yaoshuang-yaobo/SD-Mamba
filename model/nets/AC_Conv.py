import torch
import torch.nn as nn
from thop import profile
import torch.nn.modules.conv as conv


class AddCoords(nn.Module):
    def __init__(self, rank=2):
        super(AddCoords, self).__init__()
        self.rank = rank

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """

        if self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)


            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            input_tensor = input_tensor.cuda()
            xx_channel = xx_channel.cuda()
            yy_channel = yy_channel.cuda()

            out_x = torch.cat([input_tensor, xx_channel], dim=1)
            out_y = torch.cat([input_tensor, yy_channel], dim=1)

        return out_x, out_y


class AC_Conv_down(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super(AC_Conv_down, self).__init__()

        self.addcoords = AddCoords(2)

        pad = int((kernel_size - 1) / 2)
        # 如果设置pad=(kernel_size-1)/2,则运算后，宽度和高度不变
        # kernel size had better be odd number so as to avoid alignment error
        self.conv_x_i = nn.Conv2d(in_dim + 1, out_dim, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=2).cuda()

        self.conv_x_j = nn.Conv2d(out_dim + 1, out_dim, kernel_size=(1, kernel_size),  padding=(0, pad)).cuda()

        self.conv_y_i = nn.Conv2d(out_dim + 1, out_dim, kernel_size=(kernel_size, 1), padding=(pad, 0)).cuda()

        self.conv_y_j = nn.Conv2d(in_dim + 1, out_dim, kernel_size=(1, kernel_size),padding=(0, pad), stride=2).cuda()

    def forward(self, raw):
        x, y = self.addcoords(raw)
        x = self.conv_x_i(x)
        y = self.conv_y_j(y)

        x_i, x_j = self.addcoords(x)
        y_i, y_j = self.addcoords(y)

        x = self.conv_x_j(x_j)
        y = self.conv_y_i(y_i)

        return x, y


class AC_Conv_up(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super(AC_Conv_up, self).__init__()

        self.addcoords = AddCoords(2)

        pad = int((kernel_size - 1) / 2)
        # 如果设置pad=(kernel_size-1)/2,则运算后，宽度和高度不变
        # kernel size had better be odd number so as to avoid alignment error
        self.conv_x_i = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_dim + 1, out_dim, kernel_size=(kernel_size, 1), padding=(pad, 0)),
        ).cuda()

        self.conv_x_j = nn.Conv2d(out_dim + 1, out_dim, kernel_size=(1, kernel_size),  padding=(0, pad)).cuda()

        self.conv_y_i = nn.Conv2d(out_dim + 1, out_dim, kernel_size=(kernel_size, 1), padding=(pad, 0)).cuda()

        self.conv_y_j = nn.Sequential(
            nn.Conv2d(in_dim + 1, out_dim, kernel_size=(1, kernel_size),padding=(0, pad)),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        ).cuda()

    def forward(self, raw):
        x, y = self.addcoords(raw)
        x = self.conv_x_i(x)
        y = self.conv_y_j(y)

        x_i, x_j = self.addcoords(x)
        y_i, y_j = self.addcoords(y)

        x = self.conv_x_j(x_j)
        y = self.conv_y_i(y_i)

        return x, y


if __name__ == '__main__':
    model = AC_Conv_up(in_dim=2, out_dim=64)
    model.eval()
    image = torch.randn(1, 2, 256, 256)
    with torch.no_grad():
        x, y = model.forward(image)
    print(x.shape, y.shape)
    flops, params = profile(model, (image,))
    print('flops: %.2f GFLOPS, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

