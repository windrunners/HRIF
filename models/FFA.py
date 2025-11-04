import torch.nn as nn
import torch


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DualChannelAttention(nn.Module):
    def __init__(self, channel):
        super(DualChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SingleChannelAttention(nn.Module):
    def __init__(self, channel):
        super(SingleChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


# 基础Block，无任何注意力机制
class BaseBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(BaseBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res += x
        return res


# 单通道注意力Block
class SingleAttentionBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(SingleAttentionBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.ca = SingleChannelAttention(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.ca(res)
        res = self.palayer(res)
        res += x
        return res


# 双通道注意力Block
class DualAttentionBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DualAttentionBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.dca = DualChannelAttention(dim)
        self.sa = SpatialAttention()
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        dca_out = self.dca(res)
        sa_out = self.sa(res)
        res = res * dca_out * sa_out
        res = self.palayer(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks, block_type='dual'):
        super(Group, self).__init__()
        if block_type == 'base':
            block_class = BaseBlock
        elif block_type == 'single':
            block_class = SingleAttentionBlock
        else:  # dual
            block_class = DualAttentionBlock

        modules = [block_class(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


# 原始FFA网络（双通道注意力+空间注意力）
class FFA(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='dual')
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='dual')
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='dual')

        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.sa = SpatialAttention()
        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3

        sa_weight = self.sa(out)
        out = out * sa_weight

        out = self.palayer(out)
        x = self.post(out)
        return x + x1


# 消融实验1：无通道注意力机制，只有空间注意力
class FFA1(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA1, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='base')
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='base')
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='base')

        self.sa = SpatialAttention()
        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        # 平均融合
        out = (res1 + res2 + res3) / 3

        # 只使用空间注意力
        sa_weight = self.sa(out)
        out = out * sa_weight

        out = self.palayer(out)
        x = self.post(out)
        return x + x1


# 消融实验2：单通道注意力机制，无空间注意力
class FFA2(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA2, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='single')
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='single')
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='single')

        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3

        out = self.palayer(out)
        x = self.post(out)
        return x + x1


# 消融实验3：无任何注意力机制
class FFA3(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA3, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='base')
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='base')
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='base')

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        # 简单平均融合
        out = (res1 + res2 + res3) / 3

        x = self.post(out)
        return x + x1


# 消融实验4：双通道注意力，无空间注意力
class FFA4(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA4, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='dual')
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='dual')
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks, block_type='dual')

        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3

        out = self.palayer(out)
        x = self.post(out)
        return x + x1


if __name__ == "__main__":
    # 原始版本
    net = FFA(gps=3, blocks=3)
    print("Original FFA Network:")
    print(net)

    # 消融实验版本
    net1 = FFA1(gps=3, blocks=3)  # 无通道注意力，只有空间注意力
    net2 = FFA2(gps=3, blocks=3)  # 单通道注意力，无空间注意力
    net3 = FFA3(gps=3, blocks=3)  # 无任何注意力机制
    net4 = FFA4(gps=3, blocks=3)  # 双通道注意力，无空间注意力

    print("\nAblation Networks:")
    print("FFA1 (No Channel Attention):", net1)
    print("FFA2 (Single Channel Attention):", net2)
    print("FFA3 (No Attention):", net3)
    print("FFA4 (Dual Channel Attention, No Spatial):", net4)