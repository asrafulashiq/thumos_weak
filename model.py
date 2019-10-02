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

        # self.weight = torch.nn.Parameter(
        #     data = torch.eye(n_feature).unsqueeze(0).expand(
        #         n_class, n_feature, n_feature
        #     )
        # )

        self.weight = torch.eye(n_feature).unsqueeze(0).expand(
                n_class, n_feature, n_feature
            )

        # self.weight = nn.Parameter(torch.randn(
        #     n_class, n_feature
        # ))


    def get_weight(self):
        # wt = self.weight
        # ww = torch.bmm(wt.unsqueeze(-1), wt.unsqueeze(-1).permute(0, 2, 1))
        # return ww

        return self.weight

    def forward(self, inputs, is_training=True):

        x = F.relu(self.fc(inputs))
        if is_training:
            x = self.dropout(x)
        return x, self.classifier(x)