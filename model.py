import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.nn.utils import weight_norm
from torch.autograd import Variable

from tcn import TemporalConvNet as tcn

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            torch_init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


class Model_orig(torch.nn.Module):
    def __init__(self, n_feature, n_class):
        super(Model_orig, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        self.classifier = nn.Linear(n_feature, n_class)
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):

        x = F.relu(self.fc(inputs))
        if is_training:
            x = self.dropout(x)
        return x, self.classifier(x)


def tri_filter(size=3):
    k = size//2
    if k==0:
        return np.array([1])
    x = np.arange(0, k+1)
    y = 1 - x / (k+1)
    rev = y[1:]
    rev = rev[::-1]
    y = np.concatenate((rev, y))
    return y / np.sum(y)


class FilterBlock(torch.nn.Module):
    def __init__(self, in_channel=2048, size=3, stride=1):
        super(FilterBlock, self).__init__()
        size = size - (size % 2 - 1)  # only odd size
        self.size = size
        self.stride = stride
        self.in_channel = in_channel

        self.pad = nn.ReplicationPad1d(self.size//2)

        _filter = torch.FloatTensor(tri_filter(self.size))
        self.filter = Variable(_filter.view(1, 1, -1).repeat(
                            (in_channel, 1, 1))).cuda()
        self.filter.requires_grad = False

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        x = self.pad(x)
        x = F.conv1d(x, self.filter, stride=self.stride,
                     groups=self.in_channel)
        x = x.transpose(1, 2)
        x = x.squeeze(0)
        return x

class Model(torch.nn.Module):
    def __init__(self, n_feature, n_class):
        super(Model, self).__init__()

        self.filter_block = FilterBlock(n_feature, size=5)

        # self.init_fc = nn.Linear(n_feature, n_feature)
        # self.init_drop = nn.Dropout(0.6)

        self.fc = nn.Linear(n_feature, n_feature)
        self.classifier = nn.Linear(n_feature, n_class, bias=True)
        self.dropout = nn.Dropout(0.7)

        self.apply(weights_init)

    def forward(self, inputs, is_training=True, is_tmp=False):
        # inputs = F.relu(self.init_fc(inputs))
        # if is_training:
        #     inputs = self.init_drop(inputs)
        inputs = self.filter_block(inputs)
        x = F.relu(self.fc(inputs))
        if is_training:
            x = self.dropout(x)
        if is_tmp:
            return x
        return x, self.classifier(x)


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


class Model_detect(nn.Module):
    def __init__(self, n_feature, n_class, down_rate=2, dropout_rate=0.7):
        super(Model_detect, self).__init__()
        # self.down_rate = down_rate
        # self.dropout_rate = dropout_rate

        self.init_fc = nn.Linear(n_feature, n_feature)

        self.drop1 = nn.Dropout(dropout_rate)

        self.tcn = tcn(n_feature, [n_feature//4], kernel_size=2, dropout=0.5)

        self.classifier = nn.Linear(n_feature//4, n_class)

        self.apply(weights_init)

    def forward(self, inputs, is_training=True, is_tmp=False):
        # N, L, Cin
        if len(inputs.shape) < 3:
            inputs = inputs.unsqueeze(0)
        # N, L, _ = inputs.shape
        inputs = self.drop1(F.relu(self.init_fc(inputs)))
        x_in = self.tcn(inputs.transpose(-1, -2))
        x = x_in.transpose(-1, -2)

        x_class = self.classifier(x)

        x = x.squeeze(0)
        x_class = x_class.squeeze(0)

        return x, x_class


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        else:
            return x[:, :, :-self.chomp_size].contiguous()


class Model_tcn(nn.Module):
    def __init__(self, n_feature, n_class, down_rate=2, dropout_rate=0.7):
        super(Model_tcn, self).__init__()

        self.init_fc = nn.Linear(n_feature, n_feature)

        self.drop1 = nn.Dropout(dropout_rate)

        # self.tcn = tcn(n_feature, [n_feature//4], kernel_size=2, dropout=0.5)

        self.classifier = nn.Conv1d(n_feature, n_class, kernel_size=3,
                                    stride=1, padding=1, dilation=1)
        # self.chomp = Chomp1d(1)

        self.apply(weights_init)

    def forward(self, inputs, is_training=True, is_tmp=False):
        # N, L, Cin
        if len(inputs.shape) < 3:
            inputs = inputs.unsqueeze(0)
        x = self.drop1(F.relu(self.init_fc(inputs)))

        x_class = self.classifier(x.transpose(-2, -1))
        # x_class = self.chomp(x_class)
        x_class = x_class.transpose(-2, -1)

        x = x.squeeze(0)
        x_class = x_class.squeeze(0)

        return x, x_class
