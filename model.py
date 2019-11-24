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


class Custom_BMN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tscale = args.max_seqlen
        self.prop_boundary_ratio = 0
        self.num_sample = args.num_sample
        self.num_sample_perbin = 1
        self.feat_dim = args.feature_size

        self.hidden_dim_1d = 512
        self.hidden_dim_2d = 512

        self.n_class = args.num_class

        self.sample_mask = self._get_interp1d_mask()

        # Base Module
        self.conv_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

        # classification module
        self.conv_class = nn.Conv1d(self.hidden_dim_1d, self.n_class, 1, bias=False)

        # attention module
        self.conv_atn = nn.Conv1d(self.hidden_dim_1d, 1, 3, padding=1)


        # Proposal Evaluation Module
        # self.conv_2d_p = nn.Sequential(
        #     nn.Conv2d(
        #         3 * self.hidden_dim_1d,
        #         3 * self.hidden_dim_2d,
        #         kernel_size=(1, 1),
        #         groups=3,
        #     ),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.6),
        # )

        # self.conv_conf = nn.Sequential(
        #     nn.Conv2d(3 * self.hidden_dim_2d, self.n_class, kernel_size=1)
        # )

        # self.conv_conf = nn.Sequential(
        #     nn.Conv2d(3 * self.hidden_dim_2d, self.n_class, kernel_size=1)
        # )

        # self.conv_attn = nn.Sequential(
        #     nn.Conv2d(3 * self.hidden_dim_2d, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )

        self.apply(weights_init)

    def forward(self, x, is_training=True):
        x = x.permute(0, 2, 1)  # B, C, T
        B, C, T = x.shape
        x_feature = self.conv_1d_b(x)  # --> B, C, T

        # consists of x_start, x_mid, x_end
        # --> B, 3*C, T, T
        # x_p = self._boundary_matching_layer(x_feature)

        # # B, 3 * C, T, T --> B, 3 * C, T, T
        # x_pp = self.conv_2d_p(x_p)

        # confidence_map = self.conv_conf(x_pp)  # --> B, cls, T, T

        # # attention_map = self.conv_attn(x_pp)
        y_class = self.conv_class(x_feature)  # --> B, cls, T
        y_atn = self.conv_atn(x_feature)  # --> B, 1, T

        # bmn class --> B, 3*cls, T, T
        if is_training:
            bmn_class = self._boundary_matching_layer(y_class, self.sample_mask, self.tscale)
        else:
            # sample_mask = self._get_interp1d_mask(T)
            bmn_class = self._boundary_matching_layer(y_class, self.sample_mask, self.tscale)

        # bmn_strt, bmn_mid, bmn_end = (
        #     bmn_class[:, :self.n_class],
        #     bmn_class[:, self.n_class: -self.n_class],
        #     bmn_class[:, -self.n_class:]
        # )  # --> B, cls, T, T
        # bmn_cls_score = torch.abs(bmn_mid-bmn_strt) + torch.abs(bmn_mid-bmn_end)

        # --> B, C, 1
        x_fg = (torch.sigmoid(y_atn) * x_feature).sum(-1, keepdim=True) / (
            torch.sigmoid(y_atn).sum(-1, keepdim=True) + 1e-8
        )

        y_fg = self.conv_class(x_fg)  # --> B, cls, 1

        return y_class, y_fg, bmn_class

    def _boundary_matching_layer(self, x, sample_mask, tscale):
        input_size = x.size()  # B, C, T

        # (B, C, T) x (T, N x T x T) --> B, C, N , T , T
        out = torch.matmul(x, sample_mask).reshape(
            input_size[0], input_size[1], self.num_sample, tscale, tscale
        )

        out_start = torch.mean(out[:, :, : self.num_sample // 4], dim=2)
        out_mid = torch.mean(
            out[:, :, self.num_sample // 4 : -self.num_sample // 4], dim=2
        )
        out_end = torch.mean(out[:, :, -self.num_sample // 4 :], dim=2)
        # each dim: B, C, T, T

        out = torch.cat((out_start, out_mid, out_end), dim=1)  # B, 3*C, T, T
        return out

    def _get_interp1d_bin_mask(
        self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin
    ):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[
                idx * num_sample_perbin : (idx + 1) * num_sample_perbin
            ]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self, tscale=None):
        # generate sample mask for each point in Boundary-Matching Map
        if tscale is None:
            tscale = self.tscale
        mask_mat = []
        for start_index in range(tscale):
            mask_mat_vector = []
            for end_index in range(tscale):
                if end_index >= start_index:
                    p_xmin = start_index
                    p_xmax = end_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin,
                        sample_xmax,
                        tscale,
                        self.num_sample,
                        self.num_sample_perbin,
                    )
                else:
                    p_mask = np.zeros([tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        mask_mat = mask_mat.transpose(0, 1, 3, 2)
        sample_mask = torch.Tensor(mask_mat).view(tscale, -1)
        return sample_mask


class BMN(nn.Module):
    def __init__(self):
        super(BMN, self).__init__()
        self.tscale = 100
        self.prop_boundary_ratio = 0.5
        self.num_sample = 32
        self.num_sample_perbin = 3
        self.feat_dim = 2048

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(
                self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                self.hidden_dim_1d,
                self.hidden_dim_1d,
                kernel_size=3,
                padding=1,
                groups=4,
            ),
            nn.ReLU(inplace=True),
        )

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(
                self.hidden_dim_1d,
                self.hidden_dim_1d,
                kernel_size=3,
                padding=1,
                groups=4,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(
                self.hidden_dim_1d,
                self.hidden_dim_1d,
                kernel_size=3,
                padding=1,
                groups=4,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(
                self.hidden_dim_1d,
                self.hidden_dim_3d,
                kernel_size=(self.num_sample, 1, 1),
            ),
            nn.ReLU(inplace=True),
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        base_feature = self.x_1d_b(x)
        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)
        confidence_map = self.x_1d_p(base_feature)
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, start, end

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(
            input_size[0], input_size[1], self.num_sample, self.tscale, self.tscale
        )
        return out

    def _get_interp1d_bin_mask(
        self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin
    ):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[
                idx * num_sample_perbin : (idx + 1) * num_sample_perbin
            ]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for start_index in range(self.tscale):
            mask_mat_vector = []
            for duration_index in range(self.tscale):
                if start_index + duration_index < self.tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin,
                        sample_xmax,
                        self.tscale,
                        self.num_sample,
                        self.num_sample_perbin,
                    )
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(
            torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False
        )


if __name__ == "__main__":
    mod = Custom_BMN()
    device = torch.device("cuda:1")
    mod.to(device)
    x = torch.rand(2, 2048, 100)
    y, y_feat = mod(x.to(device))

    print(y.shape)
    print(y_feat.shape)
