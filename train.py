import torch
import torch.nn.functional as F
import numpy as np
from soft_nms import soft_nms_pytorch


def MILL(element_logits, seq_len, labels, device):
    k = np.ceil(seq_len / 8).astype("int32")
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(element_logits.shape[0]):
        tmp, _ = torch.topk(element_logits[i][: seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat(
            [instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0
        )
    milloss = -torch.mean(
        torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0
    )
    return milloss


def smooth_tensor(x, dim=-1, sigma=3):
    b, c, l = x.shape
    gx = np.arange(-sigma, sigma + 1)
    k_gauss = np.exp(-gx ** 2 / (2.0 * (sigma / 3.0) ** 2))
    k_gauss = k_gauss / k_gauss.sum()
    kernel = torch.FloatTensor(k_gauss).reshape(1, 1, -1).repeat(c, 1, 1)
    # Create input
    x_smooth = F.conv1d(x, kernel.to(x.device), groups=c, padding=sigma)
    return x_smooth


def MILL_atn(elements_cls, elements_atn, seq_len, labels, device):
    labels = labels / torch.sum(labels, dim=1, keepdim=True)

    # atn_fg = torch.softmax(elements_atn, -1)  # --> B, 1, T
    atn_fg = elements_atn.sigmoid() / elements_atn.sigmoid().sum(-1, keepdim=True)

    # --> B, cls+1
    x_fg_cls = (atn_fg * elements_cls).sum(-1)  # / elements_cls.shape[-1]

    milloss_fg = -torch.mean(
        torch.sum(labels * F.log_softmax(x_fg_cls, dim=1)[..., 1:], dim=1), dim=0
    )

    # atn_bg = F.softmin(elements_atn, -1)
    _bg = 1 - elements_atn.sigmoid()
    atn_bg = _bg / _bg.sum(-1, keepdim=True)

    x_bg_cls = (atn_bg * elements_cls).sum(-1)  # / elements_cls.shape[-1]
    milloss_bg = -torch.mean(F.log_softmax(x_bg_cls, 1)[..., 0])

    loss = milloss_fg + 0.1 * milloss_bg
    return loss


def t_val(x):
    if not isinstance(x, torch.Tensor):
        return x
    else:
        if x.is_cuda:
            return x.data.cpu().numpy()
        else:
            return x.data.numpy()


def refine_bmn_map(bmn_class, elements_cls, labels, device, args):
    # (B, 3, cls+1, D, T), (B, cls+1, T)

    # gaussian smooth element logit
    elements_cls_smooth = smooth_tensor(elements_cls, dim=-1)

    label_ij = labels.nonzero()
    for b, cls in label_ij:
        elements_smooth = elements_cls_smooth[b, cls + 1]  # --> (T,)
        conf_map = bmn_class[b, :, cls + 1]  # --> 3, D, T

        # get valid locations
        _mask = torch.flip(torch.triu(torch.ones_like(conf_map[0]), diagonal=0), dims=[-1])

        threshold = (
            elements_smooth.max() - elements_smooth.min()
        ) * 0.5 + elements_smooth.min()

        tmp_conf_s = conf_map[0]  # --> D, T
        tmp_conf_m = conf_map[1]  # --> D, T
        tmp_conf_e = conf_map[2]  # --> D, T

        tmp_conf_score_left = tmp_conf_m - tmp_conf_s
        tmp_conf_score_right = tmp_conf_m - tmp_conf_e
        tmp_conf_score = 0.5 * (tmp_conf_score_left + tmp_conf_score_right)

        tmp_mask = (tmp_conf_m > threshold) & (
            (tmp_conf_s < threshold) | (tmp_conf_e < threshold)
        )

        score_list = []
        segment_list = []

        for each_ind in tmp_mask.nonzero():
            dur, start = each_ind
            ending = start + dur
            if (
                _mask[dur, start] > 0
                and tmp_conf_score_left[dur, start] > 0
                and tmp_conf_score_right[dur, start] > 0
            ):
                score = tmp_conf_score[dur, start] + elements_smooth[start: ending+1].mean()
                segment_list.append(torch.FloatTensor([start, ending]).cuda())
                score_list.append(score)
        segment_list = torch.stack(segment_list, 0)
        score_list = torch.stack(score_list, 0)

        keep = soft_nms_pytorch(segment_list, score_list, thresh=threshold)



def train_bmn(itr, dataset, args, model, optimizer, logger, device):
    model.train()
    features, labels = dataset.load_data(
        n_similar=args.num_similar, similar_size=args.similar_size
    )
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    # features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    elements_cls, elements_atn, bmn_class = model(features)
    # --> (B, cls+1, T), (B, 1, T), (B, 3, cls+1, D, T)
    milloss = MILL_atn(elements_cls, elements_atn, seq_len, labels, device)

    refine_bmn_map(bmn_class, elements_cls, labels, device, args)

    total_loss = milloss  # + args.gamma * metric_loss + args.gamma2 * L1loss

    print("Iteration: %d, Loss: %.4f" % (itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.data.cpu().numpy()
