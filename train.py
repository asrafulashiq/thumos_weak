import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def MILL(element_logits, seq_len, labels, device):
    """ element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over,
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) """

    k = np.ceil(seq_len / 8).astype("int32")
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(element_logits.shape[0]):
        tmp, _ = torch.topk(element_logits[i][: seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat(
            [instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0
        )
    milloss = -torch.mean(
        torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1),
        dim=0,
    )
    return milloss


def train(itr, dataset, args, model, optimizer, logger, device, scheduler=None):

    features, labels = dataset.load_data(
        n_similar=args.num_similar, similar_size=args.similar_size
    )
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    final_features, element_logits = model(features)

    milloss = MILL(element_logits, seq_len, labels, device)

    total_loss = milloss  # + (1 - args.Lambda) * casloss

    logger.add_scalar("milloss", milloss, itr)
    # logger.log_value("casloss", casloss, itr)
    logger.add_scalar("total_loss", total_loss, itr)

    print("Iteration: %d, Loss: %.3f" % (itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if scheduler:
        scheduler.step()


def MIL_BMN(conf_map, attention_map, seq_len, labels, device):
    eps = 1e-8
    B, C, *_ = conf_map.shape

    conf_map = torch.softmax(conf_map, dim=1)
    conf_map_atn = torch.triu(attention_map * conf_map, diagonal=1)
    conf_reduced = conf_map_atn.reshape(B, C, -1).max(dim=-1)[0]

    milloss = - torch.sum(labels * torch.log(conf_reduced + eps))

    if torch.isnan(milloss):
        import pdb
        pdb.set_trace()

    return milloss


def metric_function_sim_class(X1, X2):
    # input: N_sim, C
    N_sim, C = X1.shape
    X1_expand = X1.unsqueeze(1).expand(N_sim, N_sim, C)
    X2_expand = X2.unsqueeze(0).expand(N_sim, N_sim, C)
    # X1, X2: N_sim, N_sim, C

    dis_mat_s = 1 - torch.cosine_similarity(
        X1_expand[..., : C // 4], X2_expand[..., : C // 4], dim=-1
    )
    dis_mat_m = 1 - torch.cosine_similarity(
        X1_expand[..., C // 4 : -C // 4], X2_expand[..., C // 4 : -C // 4], dim=-1
    )
    dis_mat_e = 1 - torch.cosine_similarity(
        X1_expand[..., -C // 4 :], X2_expand[..., -C // 4 :], dim=-1
    )
    dis_mat = (dis_mat_e + dis_mat_s + dis_mat_m) / 3
    return dis_mat  # --> N_sim, N_sim, 1


def metric_function(X1, X2, labels):

    # X1: B1, Nc1, C
    # X2: B2, Nc2, C
    B1, Nc1, C = X1.shape
    B2, Nc2, C = X2.shape

    # reshape
    # X1_expand : B1, B2, Nc1, Nc2, C
    # X2_expand : B1, B2, Nc1, Nc2, C
    X1_expand = X1[:, None, :, None].expand(B1, B2, Nc1, Nc2, C)
    X2_expand = X2[None, :, None, :].expand(B1, B2, Nc1, Nc2, C)

    # dis : B1, B2, Nc1, Nc2
    dis_mat_s = 1 - torch.cosine_similarity(
        X1_expand[..., : C // 4], X2_expand[..., : C // 4], dim=-1
    )
    dis_mat_m = 1 - torch.cosine_similarity(
        X1_expand[..., C // 4 : -C // 4], X2_expand[..., C // 4 : -C // 4], dim=-1
    )
    dis_mat_e = 1 - torch.cosine_similarity(
        X1_expand[..., -C // 4 :], X2_expand[..., -C // 4 :], dim=-1
    )
    dis_mat = (dis_mat_e + dis_mat_s + dis_mat_m) / 3

    # mask
    mask = torch.ones(dis_mat.shape)
    mask = mask.permute(2, 3, 0, 1).triu(diagonal=1).permute(2, 3, 0, 1)
    mask_lab = labels.view(B1, 1, Nc1, 1) * labels.view(1, B2, 1, Nc2)
    mask = mask * mask_lab
    # --> B1, B2, Nc1, Nc2

    mask_same = (mask * torch.eye(Nc1)[None, None, ...]) > 0
    mask_diff = (mask * torch.ones(Nc1, Nc2).triu(diagonal=1)) > 0

    dis_similar = torch.mean(torch.masked_select(dis_mat, mask_same.to(dis_mat.device)))
    dis_dissimilar = torch.mean(
        torch.masked_select(dis_mat, mask_diff.to(dis_mat.device))
    )

    return dis_similar, dis_dissimilar


def Sim_Loss(conf_map, attention_map, x_feat, labels, device, args):
    pass
    # conf_map: (B, cls, T, T)
    # x_feat: (B, 3*C, T, T)

    eps = 1e-8
    B, N_c, *_ = conf_map.shape
    _, C, *_ = x_feat.shape

    conf_map_1 = torch.triu(torch.softmax(conf_map, dim=-2), diagonal=1)
    conf_map_2 = torch.triu(torch.softmax(conf_map, dim=-1), diagonal=1)
    conf_map_mul = conf_map_1 * conf_map_2 * attention_map

    Conf = conf_map_mul.reshape(B, N_c, -1)
    x_feat = x_feat.reshape(B, C, -1)

    x_avg_feat = (Conf.unsqueeze(2) * x_feat.unsqueeze(1)).sum(-1) / (
        Conf.unsqueeze(2).sum(dim=-1) + eps
    )  # B, N_c, 1, -1 x B, 1, C, -1 --> B, N_c, C

    dis_sim, dis_dissim = metric_function(x_avg_feat, x_avg_feat, labels)

    loss = torch.sum(
        torch.max(dis_sim - dis_dissim + args.dis, torch.tensor(0.0).to(device))
    )
    return loss


def t_val(x):
    if not isinstance(x, torch.Tensor):
        return x
    else:
        if x.is_cuda:
            return x.data.cpu().numpy()
        else:
            return x.data.numpy()


def train_bmn(itr, dataset, args, model, optimizer, logger, device):
    model.train()
    features, labels = dataset.load_data(
        n_similar=args.num_similar, similar_size=args.similar_size
    )
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    # features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    conf_map, attention_map, x_feat = model(features)
    # --> (B, cls, T, T), (C, 3*C, T, T)

    milloss = MIL_BMN(conf_map, attention_map, seq_len, labels, device)
    metric_loss = 0 #Sim_Loss(conf_map, attention_map, x_feat, labels, device, args)

    L1loss = torch.sum(attention_map) / (
        attention_map.shape[0]
        * attention_map.shape[1]
        * attention_map.shape[2]
        * attention_map.shape[3]
    )

    total_loss = milloss + args.gamma * metric_loss + args.gamma2 * L1loss

    logger.add_scalar("milloss", milloss, itr)
    logger.add_scalar("total_loss", total_loss, itr)

    # print("Iteration: %d, Loss: %.6f" % (itr, total_loss.data.cpu().numpy()))
    print(
        f"Iteration: {itr:>10d} -  {t_val(milloss): .6f} + {t_val(metric_loss):.4f} "
        + f"+ {t_val(L1loss):.4f}"
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.data.cpu().numpy()
