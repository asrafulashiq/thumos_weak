import torch
import torch.nn.functional as F
import numpy as np
from soft_nms import soft_nms_pytorch


def batch_per_dis(X1, X2, w):
    X1 = X1.permute(2, 1, 0).unsqueeze(2)
    X2 = X2.permute(2, 1, 0).unsqueeze(1)

    X_d = X1 - X2
    X_diff = X_d.view(X_d.shape[0], X_d.shape[1] * X_d.shape[2], -1)

    # w = w.unsqueeze(-1)
    # dis_mat = torch.bmm(X_diff, w).squeeze(-1)
    # dis_mat = dis_mat.view(dis_mat.shape[0], X_d.shape[1], X_d.shape[2])
    # dis_mat = torch.pow(dis_mat, 2)

    dis_mat = 1 - torch.cosine_similarity(X1, X2, dim=-1)

    return dis_mat


def get_unit_vector(x, dim=0):
    # return x
    return x / torch.norm(x, 2, dim=dim, keepdim=True)


def get_per_dis(x1, x2, w):
    w = w.unsqueeze(1)
    x1 = x1.transpose(0, 1).unsqueeze(1)
    x2 = x2.transpose(0, 1).unsqueeze(0)
    x_diff = x1 - x2

    # dis_mat = torch.pow(torch.sum(x_diff * w, -1), 2)
    dis_mat = 1 - torch.cosine_similarity(x1, x2, dim=-1)

    return dis_mat


def WLOSS_orig(x, element_logits, weight, labels, seq_len, device, args, gt_all=None):

    sim_loss = 0.0
    n_tmp = 0.0

    if gt_all is not None:
        sim_loss_gt = 0.0

    element_logits = F.hardtanh(element_logits, -args.clip, args.clip)
    for i in range(0, args.num_similar * args.similar_size, args.similar_size):
        lab = labels[i, :]
        for k in range(i + 1, i + args.similar_size):
            lab = lab * labels[k, :]

        common_ind = lab.nonzero().squeeze(-1)

        Xh = torch.Tensor()
        Xl = torch.Tensor()

        for k in range(i, i + args.similar_size):
            elem = element_logits[k][: seq_len[k], common_ind]
            atn = F.softmax(elem, dim=0)
            n1 = torch.FloatTensor([np.maximum(seq_len[k] - 1, 1)]).to(device)
            atn_l = (1 - atn) / n1

            xh = torch.mm(torch.transpose(x[k][: seq_len[k]], 1, 0), atn)
            xl = torch.mm(torch.transpose(x[k][: seq_len[k]], 1, 0), atn_l)
            xh = xh.unsqueeze(1)
            xl = xl.unsqueeze(1)
            Xh = torch.cat([Xh, xh], dim=1)
            Xl = torch.cat([Xl, xl], dim=1)

        Xh = get_unit_vector(Xh, dim=0)
        Xl = get_unit_vector(Xl, dim=0)

        D1 = batch_per_dis(Xh, Xh, weight[common_ind, :])

        D1 = torch.triu(D1, diagonal=1)
        D1 = D1.view(D1.shape[0], -1)
        d1 = torch.sum(D1, -1) / (args.similar_size*(args.similar_size-1)/2)

        D2 = batch_per_dis(Xh, Xl, weight[common_ind, :])

        D2 = D2 * (1-torch.eye(D2.shape[1])).unsqueeze(0)
        D2 = D2.view(D2.shape[0], -1)

        d2 = torch.sum(D2, -1) / (args.similar_size*(args.similar_size-1))

        loss = torch.sum(torch.max(
            d1 - d2 + args.dis, torch.FloatTensor([0.0]).to(device)
        ))

        sim_loss += loss

        n_tmp = n_tmp + torch.sum(lab)

    sim_loss = sim_loss / n_tmp
    if gt_all is not None:
        sim_loss_gt = sim_loss_gt / x.shape[0]
        return sim_loss, sim_loss_gt

    return sim_loss


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


def get_proposal(b, cls, bmn_class, bmn_complete, elements_cls_smooth, labels, device, args):
    elements_smooth = elements_cls_smooth[b, cls + 1]  # --> (T,)
    conf_map = bmn_class[b, :, cls + 1]  # --> 3, D, T
    conf_com = bmn_complete[b, cls + 1]  # --> D, T

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

    # tmp_mask = (tmp_conf_m > threshold) & (
    #     (tmp_conf_s < threshold) | (tmp_conf_e < threshold)
    # )
    tmp_mask = (_mask > 0)

    score_list = []
    segment_list = []

    for each_ind in tmp_mask.nonzero():
        dur, start = each_ind
        ending = start + dur
        # if (
        #     _mask[dur, start] > 0
        #     and tmp_conf_score_left[dur, start] > 0
        #     and tmp_conf_score_right[dur, start] > 0
        # ):
        score = tmp_conf_score[dur, start] + elements_smooth[start: ending+1].mean()
        segment_list.append(torch.FloatTensor([start, ending]).cuda())
        score_list.append(score)
    if len(segment_list) == 0:
        return None
    segment_list = torch.stack(segment_list, 0)
    score_list = torch.stack(score_list, 0)
    keep, orig_ind = soft_nms_pytorch(segment_list, score_list, thresh=threshold)
    if keep.nelement == 0:
        keep = orig_ind[0]
    inds = segment_list[keep].long()
    inds[:, 1] = inds[:, 1] - inds[:, 0]
    return inds   # statrt, duration


def refine_bmn_map(bmn_features, bmn_class, bmn_complete, elements_cls, labels, device, args):
    # (B, 3, cls+1, D, T),  (B, cls+1, D, T), (B, cls+1, T)

    # gaussian smooth element logit
    # elements_cls_smooth = smooth_tensor(elements_cls, dim=-1)

    sim_loss = 0.0
    n_tmp = 0.0

    for i in range(0, args.num_similar * args.similar_size, args.similar_size):
        lab = labels[i, :]
        for k in range(i + 1, i + args.similar_size):
            lab = lab * labels[k, :]

        common_ind = lab.nonzero().squeeze(-1)
        rand_ind = torch.randperm(common_ind.shape[0])[0]
        common_ind = common_ind[[rand_ind]]

        Xh = torch.Tensor()
        Xl = torch.Tensor()

        for k in range(i, i + args.similar_size):
            ind_h = get_proposal(i, common_ind[0], bmn_class, bmn_complete,
                elements_cls, labels, device, args)
            xh = bmn_features[i, :, ind_h[:, 1], ind_h[:, 0]]  # 3*c, ...
            xh = xh.unsqueeze(1)

            ind_l = get_proposal(i, -1, bmn_class, bmn_complete,
                elements_cls, labels, device, args)
            xl = bmn_features[i, :, ind_l[:, 1], ind_l[:, 0]]  # 3*c, ...
            xl = xl.unsqueeze(1)

            Xh = torch.cat([Xh, xh], dim=1)
            Xl = torch.cat([Xl, xl], dim=1)

        Xh = get_unit_vector(Xh, dim=0)
        Xl = get_unit_vector(Xl, dim=0)

        D1 = batch_per_dis(Xh, Xh, None)

        D1 = torch.triu(D1, diagonal=1)
        D1 = D1.view(D1.shape[0], -1)
        d1 = torch.sum(D1, -1) / (args.similar_size*(args.similar_size-1)/2)

        D2 = batch_per_dis(Xh, Xl, None)

        tt = torch.ones_like(D2)
        D2 = D2 * (torch.triu(tt) - torch.triu(tt, 1)).unsqueeze(0)
        D2 = D2.view(D2.shape[0], -1)

        d2 = torch.sum(D2, -1) / (args.similar_size*(args.similar_size-1))

        loss = torch.sum(torch.max(
            d1 - d2 + args.dis, torch.FloatTensor([0.0]).to(device)
        ))

        sim_loss += loss

        n_tmp = n_tmp + torch.sum(lab)

    sim_loss = sim_loss / n_tmp

    return sim_loss
        

def train_bmn(itr, dataset, args, model, optimizer, logger, device):
    model.train()
    features, labels = dataset.load_data()
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    # features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    elements_cls, elements_atn, bmn_features, bmn_class, bmn_complete = model(features)
    # --> (B, cls+1, T), (B, 1, T), (B, 3, cls+1, D, T)
    milloss = MILL_atn(elements_cls, elements_atn, seq_len, labels, device)

    weight = model.conv_class.weight
    final_features = final_features.permute(0, 2, 1)
    element_logits = elements_cls.permute(0, 2, 1)[..., 1:]
    # casloss = WLOSS_orig(
    #     final_features, element_logits, weight, labels, seq_len, device, args, None
    # )

    loss_dis = refine_bmn_map(bmn_features, bmn_class, bmn_complete, elements_cls, labels, device, args)

    total_loss = milloss

    print(f"{itr: >10d}: {t_val(milloss):.4f} + {t_val(0.): .4f} = {t_val(total_loss): .4f}")

    # print("Iteration: %d, Loss: %.4f" % (itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.data.cpu().numpy()
