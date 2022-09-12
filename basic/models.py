from torch.nn import Module, ModuleList
from torch.nn import Conv2d, InstanceNorm2d, Dropout, Dropout2d
from torch.nn import ReLU
from torch.nn.functional import pad
import random


class DepthSepConv2D(Module):  # Depthwise Separable Convolutions 分离表现在哪？？
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1,1), dilation=(1,1)):  # 步长保持为1 膨胀值为1
        super(DepthSepConv2D, self).__init__()
        self.padding = None
        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)


        else:
            padding = (0, 0)
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1, 1))
        self.activation = activation


    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        return x

"""
Dropout对所有元素中每个元素按照概率proba更改为零
Dropout2d的赋值对象是彩色的图像数据（batch N，通道 C，高度 H，宽 W）的一个通道里的每一个数据，即输入为 Input: (N, C, H, W) 时，对每一个通道维度 C 按概率赋值为 0。
"""
class MixDropout(Module):
    def __init__(self, dropout_proba=0.4, dropout2d_proba=0.2):   # 注意这里的dropout和dropout2d
        super(MixDropout, self).__init__()

        self.dropout = Dropout(dropout_proba)
        self.dropout2d = Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:   # 随机取 dropout和dropout2d
            return self.dropout(x)
        return self.dropout2d(x)


"""
A CB is a succession of two convolutional layers, followed
by an Instance Normalization layer. A third convolutional layer
is applied at the end of this block. Each convolution layer uses
3 × 3 kernels and is followed by a ReLU activation function; zero
padding is introduced to remove the kernel edge effect. 
"""
class ConvBlock(Module):

    def __init__(self, in_, out_, stride=(1,1), k=3, activation=ReLU, dropout=0.4):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = Conv2d(in_channels=in_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv2 = Conv2d(in_channels=out_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv3 = Conv2d(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)  # TODO:这里的参数或许可以改改
        """
        在样本N和通道C两个维度上滑动，对Batch中的N个样本里的每个样本n，和C个通道里的每个样本c，其组合[n, c]求对应的所有值的均值和方差，所以得到的是N⋅C个均值和方差
        eps: 为保证数值稳定性（分母不能趋近或取0），给分母加上的值，默认为1e-5
        momentum：动态均值和动态方差所使用的动量，默认为0.1
        affine： 布尔值， 当为True时，给该层添加可学习的仿射变换参数
        track_running_stats: 布尔值，当设为True，记录训练过程中的均值和方差

        """
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x


class DSCBlock(Module):   # DSCBlock结构

    def __init__(self, in_, out_, pool=(2, 1), activation=ReLU, dropout=0.4):
        super(DSCBlock, self).__init__()
        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_, out_, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_, out_, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=pool)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)
        return x


class FCN_Encoder(Module):
    def __init__(self, params):   # 好好看看params
        super(FCN_Encoder, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
        ])
        self.blocks = ModuleList([
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 256, pool=(1, 1), dropout=self.dropout),
        ])

    def forward(self, x):
        for b in self.init_blocks:   # block内的循环
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt   # 残差连接
        return x
