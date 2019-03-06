import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

from tcn import TemporalConvNet as tcn

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class Model(torch.nn.Module):
    def __init__(self, n_feature, n_class):
        super(Model, self).__init__()

        self.fc = nn.Linear(n_feature, n_feature)
        # self.fc1 = nn.Linear(n_feature, n_feature)
        self.classifier = nn.Linear(n_feature, n_class)
        self.dropout = nn.Dropout(0.7)

        self.apply(weights_init)

    def forward(self, inputs, is_training=True):

        x = F.relu(self.fc(inputs))
        if is_training:
            x = self.dropout(x)
        #x = F.relu(self.fc1(x))
        #if is_training:
        #    x = self.dropout(x)
        return x, self.classifier(x)


class Model_attn(torch.nn.Module):
    def __init__(self, n_feature, n_class):
        super(Model_attn, self).__init__()

        self.fc = nn.Linear(n_feature, n_feature)
        # self.fc1 = nn.Linear(n_feature, n_feature)

        # attention
        self.fc_a1 = nn.Linear(n_feature, 512)
        self.fc_a2 = nn.Linear(512, 1)


        # classifier
        self.classifier = nn.Linear(n_feature, n_class)
        self.dropout = nn.Dropout(0.6)

        self.apply(weights_init)

    def forward(self, inputs, is_training=True):

        x = F.relu(self.fc(inputs))
        if is_training:
            x = self.dropout(x)

        # attention
        x_a = F.relu(self.fc_a1(x))
        if is_training:
            x_a = self.dropout(x_a)
        x_a = self.fc_a2(x_a)

        # classifier
        x_class = self.classifier(x)

        return x_a, x_class


class TemporalAttention(nn.Module):
    def __init__(self, n_feat, dropout_rate=0.5):
        super(TemporalAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(n_feat, 1)

    def forward(self, input):
        # assume input in (N, L, C) shape
        # x = self.dropout(input)
        x = self.fc(input)  # (N, L, 1)
        x_a = torch.mean(x * input, dim=-2)  # (N, C)
        return x_a


class SpatialAttention(nn.Module):
    def __init__(self, in_feature, dropout_rate=0.5):
        super(SpatialAttention, self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_feature, in_feature)

    def forward(self, input):
        # input shape: (N, Cin, L)
        x = self.pool(input)  # (N, Cin, 1)
        x = self.dropout(x)
        x_flat = x.squeeze(-1)  # (N, Cin)

        y = self.fc(x_flat)  # (N, Cin)
        y = y.unsqueeze(-1)  # (N, Cin, 1)

        return input * y  # (N. Cin, L)


class AdaptiveBlock(nn.Module):
    def __init__(self, n_feature, L, dropout_rate=0.5):
        super(AdaptiveBlock, self).__init__()
        self.tcn = tcn(num_inputs=n_feature,
                        num_channels=[n_feature],
                        kernel_size=2, dropout=dropout_rate)
        self.pool = nn.AdaptiveMaxPool1d(L//2)

    def forward(self, x):
        x = self.tcn(x)
        x = self.pool(x)
        return x


class ATTN_wt(nn.Module):
    def __init__(self, n_feature, kernel=2, dropout_rate=0.5):
        super(ATTN_wt, self).__init__()
        self.conv1d = nn.Conv1d(n_feature, kernel, kernel_size=kernel,
                                stride=kernel)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # input N, L, Cin
        xt = x.transpose(-1, -2)  # N, Cin, L
        wt = self.relu(self.conv1d(xt))  # N, 2, L//2
        w = wt.transpose(-1, -2)  # N, L//2, 2
        w = self.softmax(w)  # N, L//2, 2
        return w


class Model_detect(nn.Module):
    def __init__(self, n_feature, n_class, down_rate=2, dropout_rate=0.5):
        super(Model_detect, self).__init__()
        self.down_rate = down_rate

        self.init_fc = nn.Linear(n_feature, 512)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.attn_block = ATTN_wt(512, kernel=down_rate)
        self.avg_pool = nn.AvgPool1d(kernel_size=down_rate, stride=down_rate,
                                     ceil_mode=True)

        self.refine_fc = nn.Linear(512, 512)
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc_class = nn.Linear(512, n_class)

    def forward(self, inputs):
        # N, L, Cin
        N, L, _ = inputs.shape
        x = self.relu(self.init_fc(inputs))
        x = self.drop1(x)  # N, L, 512

        x = self.drop2(self.relu(self.refine_fc(x)))
        x_class_init = self.fc_class(x)  # N, L, cls

        w_mean = 0
        while L != 1:
            # x = self.drop2(self.relu(self.refine_fc(x)))
            x = self.drop2(x)
            wt = self.attn_block(x)  # N, L//2, 2
            wt = wt.view(N, -1, 1)  # N, L//2 * 2, 1
            x[:, :L//2 * 2, :] = x[:, :L//2 * 2, :].clone() * wt

            x = x.transpose(-1, -2)
            x = self.avg_pool(x)
            x = x.transpose(-1, -2)
            L = x.shape[-2]
            w_mean += torch.mean(wt.view(-1)**2)

        w_mean /= (int(np.log2(inputs.shape[-2]))+1)

        x_class = self.fc_class(x)  # N, 1, cls
        return x_class, x_class_init, w_mean



class Model_tcn(torch.nn.Module):
    def __init__(self, n_feature, n_class, dropout_rate=0.5, tlen=750):
        super(Model_tcn, self).__init__()
        self.n_class = n_class
        self.n_feature = n_feature

        # self.tcn1 = tcn(num_inputs=512,
        #                 num_channels=[512, 512],
        #                 kernel_size=2, dropout=dropout_rate)

        self.test_fc = nn.Linear(n_feature, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

        self.tcn = tcn(512, [512], kernel_size=1, dropout=0.5)

        # self.pool = nn.AdaptiveMaxPool1d(1)
        self.spatial_pool = SpatialAttention(512)

        self.temp_pool = TemporalAttention(512)

        self.conv_class = nn.Linear(512, n_class)

        self.drop = nn.Dropout(dropout_rate)

    def forward(self, inputs, is_training=True):

        # input shape : (N, L, Cin)
        x = self.test_fc(inputs)  # (N, L, 512)
        x = self.relu(x)
        x = x.transpose(-1, -2)  # (N, 512, L)

        L = x.shape[-1]
        _len = int(np.log2(L))
        for i in range(_len):
            x = self.tcn(x)
            x = nn.AdaptiveMaxPool1d(L//2)(x)
        x = nn.AdaptiveMaxPool1d(1)(x)

        x = x.squeeze(-1)  # N, 512
        x = self.conv_class(x)

        return x