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
        y_atn = (self.conv_atn(x_feature))  # --> B, 1, T
        return y_class, y_atn



class Custom_BMN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.max_seqlen
        self.tscale = min(args.max_seqlen // 4, 50)
        self.prop_boundary_ratio = 0.25
        self.num_sample = args.num_sample
        self.num_sample_perbin = 1
        self.feat_dim = args.feature_size

        self.hidden_dim_1d = 512
        self.hidden_dim_2d = 512

        self.n_class = args.num_class

        self.sample_mask = self._get_interp1d_mask(self.seq_len, self.tscale)

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

        y_class = self.conv_class(x_feature)  # --> B, cls, T
        y_atn = self.conv_atn(x_feature)  # --> B, 1, T
        
        # if is_training:
        #     bmn_class = self._boundary_matching_layer(y_class, self.sample_mask, self.seq_len, self.tscale)
        # else:
        #     tscale = min(T // 4, 50)
        #     sample_mask = self._get_interp1d_mask(T, tscale)
        #     bmn_class = self._boundary_matching_layer(y_class, sample_mask, T, tscale)
        # bmn_class = None
        bmn_class = self._boundary_matching_layer(y_class, self.sample_mask, self.seq_len, self.tscale)

        return y_class, y_atn, bmn_class

    def _boundary_matching_layer(self, x, sample_mask, seq_len, tscale):
        input_size = x.size()  # B, C, T

        # (B, C, T) x (T, N x D x T) --> B, C, N , D , T
        out = torch.matmul(x, sample_mask).reshape(
            input_size[0], input_size[1], self.num_sample, tscale, seq_len
        )

        out_start = torch.mean(out[:, :, : self.num_sample // 4], dim=2)
        out_mid = torch.mean(
            out[:, :, self.num_sample // 4 : -self.num_sample // 4], dim=2
        )
        out_end = torch.mean(out[:, :, -self.num_sample // 4 :], dim=2)
        # each dim: B, C, D, T

        out = torch.cat((out_start, out_mid, out_end), dim=1)  # B, 3*C, D, T
        B, _, D, T = out.shape
        out = out.reshape(B, 3, -1, D, T)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
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

    def _get_interp1d_mask(self, seq_len, tscale, with_tensor=True):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for start_index in range(seq_len):
            mask_mat_vector = []
            for duration_index in range(tscale):
                if start_index + duration_index < seq_len:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, seq_len, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([seq_len, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)  # (T x N) x D x T
        if with_tensor:
            sample_mask = torch.Tensor(mask_mat)
            sample_mask = sample_mask.view(seq_len, -1)
        else:
            sample_mask = mask_mat
        return sample_mask


if __name__ == '__main__':
    import options
    opt = options.parser.parse_args()
    model = Custom_BMN(opt)
    # print(model.sample_mask.shape)


    # save sample_mask for different shapes
    from dataset import Dataset
    from tqdm import tqdm

    # dataset = Dataset(opt, mode='both')
    # dict_sample_mask = {}
    # for counter, (features, labels, idx) in tqdm(enumerate(dataset.load_test())):
    #     seq_len = features.shape[0]
    #     tscale = min(seq_len//4, 100)
    #     sample_mask = model._get_interp1d_mask(seq_len, tscale, with_tensor=False)
    #     if seq_len not in dict_sample_mask:
    #         dict_sample_mask[seq_len] = sample_mask
    #     if counter > 5:
    #         break

    # np.save("sample_mask.npy", dict_sample_mask)