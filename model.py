import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import math

torch.set_default_tensor_type("torch.cuda.FloatTensor")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
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


class Custom(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tscale = args.max_seqlen
        self.feat_dim = args.feature_size

        self.hidden_dim_1d = 512
        self.n_class = args.num_class

        # Base Module
        self.conv_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

        # classification module
        self.conv_class = nn.Conv1d(self.hidden_dim_1d, self.n_class+1, 1, bias=False)

        # attention module
        self.conv_atn = nn.Conv1d(self.hidden_dim_1d, 1, 3, padding=1)

        self.apply(weights_init)

    def forward(self, x, is_training=True):
        x = x.permute(0, 2, 1)  # B, C, T
        B, C, T = x.shape
        x_feature = self.conv_1d_b(x)  # --> B, C, T

        y_class = self.conv_class(x_feature)  # --> B, cls, T
        y_atn = torch.sigmoid(self.conv_atn(x_feature))  # --> B, 1, T
        return y_class, y_atn

