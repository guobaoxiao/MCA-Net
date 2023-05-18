import torch
import torch.nn as nn
from loss import batch_episym


class PointCN(nn.Module):
    def __init__(self, channels, out_channels):
        nn.Module.__init__(self)

        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv(x)

        out = out + x

        return out


class Channel(nn.Module):
    def __init__(self, channels, out_channels):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
        )
        self.weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_1.data.fill_(0.1)

    def forward(self, x):
        out = self.conv(x)
        # out = out + x
        return out * self.weight_1


class Down(nn.Module):
    def __init__(self, channels, out_channels):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        out = self.conv(x)
        # out = out + x
        return out


class MCABlock(nn.Module):
    def __init__(self, net_channels, input_channel):
        nn.Module.__init__(self)
        channels = net_channels
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

        self.s1_1 = PointCN(channels, channels)
        self.s1_2 = PointCN(channels, channels)
        self.s1_3 = PointCN(channels, channels)
        self.s1_4 = PointCN(channels, channels)
        self.s1_5 = PointCN(channels, channels)
        self.s1_6 = PointCN(channels, channels)

        self.s2_1 = PointCN(channels // 2, channels // 2)
        self.s2_2 = PointCN(channels // 2, channels // 2)
        self.s2_3 = PointCN(channels // 2, channels // 2)
        self.s2_4 = PointCN(channels // 2, channels // 2)
        self.s2_5 = PointCN(channels // 2, channels // 2)

        self.s3_1 = PointCN(channels // 4, channels // 4)

        self.down1 = Down(channels, channels // 2)
        self.down2 = Down(channels // 2, channels // 4)

        
        self.info1_2 = Channel(channels, channels // 2)
        self.info2_1 = Channel(channels // 2, channels)

        self.iinfo1_2 = Channel(channels, channels // 2)
        self.iinfo2_1 = Channel(channels // 2, channels)

        self.iiinfo2_1 = Channel(channels // 2, channels)
        self.iiinfo3_1 = Channel(channels // 4, channels)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, data, xs):
        # data: b*c*n*1
        batch_size, num_pts = data.shape[0], data.shape[2]
        x_row = self.conv1(data)

        x1_1 = self.s1_1(x_row)
        x2_0 = self.down1(x1_1)

        x1_2 = self.s1_2(x1_1)
        x2_1 = self.s2_1(x2_0)

        # ��һ����Ϣ����
        x1_3 = self.s1_3(x1_2 + self.info2_1(x2_1))
        x2_2 = self.s2_2(x2_1 + self.info1_2(x1_2))
        # ���

        x1_4 = self.s1_4(x1_3)
        x2_3 = self.s2_3(x2_2)

        #�ڶ�����Ϣ����
        x1_5 = self.s1_5(x1_4 + self.iinfo2_1(x2_3))
        x2_4 = self.s2_4(x2_3 + self.iinfo1_2(x1_4))
        #���

        x3_0 = self.down2(x2_4)

        x1_6 = self.s1_6(x1_5)
        x2_5 = self.s2_5(x2_4)
        x3_1 = self.s3_1(x3_0)

        out = x1_6 + self.iiinfo2_1(x2_5) + self.iiinfo3_1(x3_1)

        logits = torch.squeeze(torch.squeeze(self.output(out), 3), 1)
        e_hat = weighted_8points(xs, logits)

        x1, x2 = xs[:, 0, :, :2], xs[:, 0, :, 2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        return logits, e_hat, residual


class MCANet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        depth_each_stage = config.net_depth // (config.iter_num + 1)
        self.side_channel = (config.use_ratio == 2) + (config.use_mutual == 2)
        self.weights_init = MCABlock(config.net_channels, 4 + self.side_channel)
        self.weights_iter = [MCABlock(config.net_channels, 6 + self.side_channel) for
                             _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # data: b*1*n*c
        input = data['xs'].transpose(1, 3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1, 2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat = [], []
        logits, e_hat, residual = self.weights_init(input, data['xs'])
        res_logits.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat, residual = self.weights_iter[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()],
                          dim=1),
                data['xs'])
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

