import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

torch.set_default_tensor_type("torch.cuda.FloatTensor")


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

def peak_find(x, win_size):
    x = x.transpose(0, 1).unsqueeze(0)  # 1, C, L
    out_size = int(np.ceil(x.shape[-1] / win_size))
    x_top = F.adaptive_max_pool1d(x, out_size)
    x_top = x_top.squeeze(0).transpose(0, 1)
    return x_top

def n_peak_find(x, args, win_size):
    # x : L, C
    split_list = [win_size]*(x.shape[0]//win_size-1) + \
        [x.shape[0] - max(0, win_size*(x.shape[0]//win_size-1))]
    splits = torch.split(x, split_list, dim=0)
    peak = torch.Tensor()
    for x_split in splits:
        sp_topk = torch.topk(x_split, int(min(x.shape[0], args.topk2)),
                             dim=0)[0]
        mean_sp = torch.mean(sp_topk, 0, keepdim=True)
        peak = torch.cat((peak, mean_sp))
    return peak


def MILL_test(element_logits, seq_len, labels, device, args):
    """ element_logits should be torch tensor of dimension
        (B, n_element, n_class),
        k should be numpy array of dimension (B,) indicating the top k
        locations to average over,
        labels should be a numpy array of dimension (B, n_class) of 1 or 0
        return is a torch tensor of dimension (B, n_class) """

    k = np.ceil(seq_len / args.topk).astype("int32")
    # labels = labels / torch.sum(labels, dim=1, keepdim=True)
    # instance_logits = torch.zeros(0).to(device)
    eps = 1e-8
    loss = 0
    element_logits = F.hardtanh(element_logits, -args.clip, args.clip)
    for i in range(element_logits.shape[0]):
        # tmp, _ = torch.topk(element_logits[i][: seq_len[i]], k=int(k[i]), dim=0)
        # topk = torch.sigmoid(torch.mean(tmp, 0))

        peaks = n_peak_find(element_logits[i][: seq_len[i]], args,
                          win_size=int(args.topk))

        block_peak = torch.sigmoid(peaks)

        prob = 1 - torch.prod( 1-block_peak, 0)

        lab = Variable(labels[i])
        loss1 = -torch.sum(lab * torch.log(prob + eps)) / torch.sum(lab)
        loss2 = -torch.sum((1 - lab) * torch.log(1 - prob + eps)) / torch.sum(1 - lab)


        # tmp = torch.topk(peaks, k=int(np.ceil(peaks.shape[0]/args.topk2)), dim=0)[0]
        # # tmp = torch.topk(peaks, k=int(min(peaks.shape[0], args.topk2)), dim=0)[0]
        # topk = torch.sigmoid(torch.mean(tmp, 0))
        # #neg_topk = torch.sigmoid(
        # #    peak_find(element_logits[i][: seq_len[i]], int(k[i]))
        # #)

        # lab = Variable(labels[i])
        # loss1 = -torch.sum(lab * torch.log(topk + eps)) / torch.sum(lab)
        # loss2 = -torch.sum((1 - lab) * torch.log(1 - topk + eps)) / torch.sum(1 - lab)

        # lab = Variable(labels[i])
        # loss1 = -torch.sum(lab * torch.log(topk + eps)) / torch.sum(lab)
        # loss2 = -torch.sum((1 - lab) * torch.log(1 - topk + eps)) / torch.sum(1 - lab)

        loss += 1 / 2 * (loss1 + loss2)
        if torch.isnan(loss):
            import pdb

            pdb.set_trace()

    milloss = loss / element_logits.shape[0]
    return milloss



def MILL_all(element_logits, seq_len, labels, device, args):
    """ element_logits should be torch tensor of dimension
        (B, n_element, n_class),
        k should be numpy array of dimension (B,) indicating the top k
        locations to average over,
        labels should be a numpy array of dimension (B, n_class) of 1 or 0
        return is a torch tensor of dimension (B, n_class) """

    k = np.ceil(seq_len / args.topk).astype("int32")
    # labels = labels / torch.sum(labels, dim=1, keepdim=True)
    # instance_logits = torch.zeros(0).to(device)
    eps = 1e-8
    loss = 0
    element_logits = F.hardtanh(element_logits, -args.clip, args.clip)
    for i in range(element_logits.shape[0]):
        tmp, _ = torch.topk(element_logits[i][: seq_len[i]], k=int(k[i]), dim=0)
        topk = torch.sigmoid(torch.mean(tmp, 0))

        lab = Variable(labels[i])
        loss1 = -torch.sum(lab * torch.log(topk + eps)) / torch.sum(lab)
        loss2 = -torch.sum((1 - lab) * torch.log(1 - topk + eps)) / torch.sum(1 - lab)

        loss += 1 / 2 * (loss1 + loss2)
        if torch.isnan(loss):
            import pdb

            pdb.set_trace()

    milloss = loss / element_logits.shape[0]
    return milloss


def get_unit_vector(x, dim=0):
    # return x
    return x / torch.norm(x, 2, dim=dim, keepdim=True)


def max_like(a, b, beta=10):
    return 1/beta * torch.log(torch.exp(beta*a) + torch.exp(beta*b))
    # return torch.max(a, b)

def min_like(a, b, beta=10):
    return - max_like(-a, -b, beta)


def list_max_like(x, beta=100):
    return 1/beta * torch.logsumexp(beta*x, -1)
    #return torch.max(x, dim=-1)[0]


def list_min_like(x, beta=100):
    return -list_max_like(-x, beta=beta)
    #return torch.min(x, dim=-1)[0]


def get_per_dis(x1, x2, w):
    w = w.unsqueeze(1)
    x1 = x1.transpose(0, 1).unsqueeze(1)
    x2 = x2.transpose(0, 1).unsqueeze(0)
    x_diff = x1 - x2

    dis_mat = torch.pow(torch.sum(x_diff * w, -1), 2)
    # dis_mat = 1 - torch.cosine_similarity(x1, x2, dim=-1)

    return dis_mat


def batch_per_dis(X1, X2, w):
    X1 = X1.permute(2, 1, 0).unsqueeze(2)
    X2 = X2.permute(2, 1, 0).unsqueeze(1)

    X_d = X1 - X2
    X_diff = X_d.view(X_d.shape[0], X_d.shape[1] * X_d.shape[2], -1)

    w = w.unsqueeze(-1)
    dis_mat = torch.bmm(X_diff, w).squeeze(-1)
    dis_mat = dis_mat.view(dis_mat.shape[0], X_d.shape[1], X_d.shape[2])

  #  dis_mat = 1 - torch.cosine_similarity(X1, X2, dim=-1)

    return torch.pow(dis_mat, 2)


def WLOSS_orig(x, element_logits, weight, labels, seq_len, device, args, gt_all=None):

    sim_loss = 0.0
    labels = Variable(labels)
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

            # atn_l = F.softmin(elem, dim=0)

            #_atn = F.sigmoid(element_logits[k][:seq_len[k], common_ind])
            #atn = _atn / torch.sum(_atn, 0, keepdim=True)

            #atn_l = (1 - _atn) / torch.sum(1-_atn, 0, keepdim=True)

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
        #d1 = torch.max(D1, -1)[0]
        # D1 = D1.reshape(args.similar_size**2)
        # d1 = list_max_like(D1, beta=args.beta1)

        D2 = batch_per_dis(Xh, Xl, weight[common_ind, :])

        D2 = D2 * (1-torch.eye(D2.shape[1])).unsqueeze(0)
        D2 = D2.view(D2.shape[0], -1)

        d2 = torch.sum(D2, -1) / (args.similar_size*(args.similar_size-1))
        # d2 = list_min_like(D2, beta=args.beta1)

        # check with gt
        if gt_all is not None:
            gt = gt_all[common_ind].unsqueeze(1)
            gt = get_unit_vector(gt)
            D1_gt = get_per_dis(Xh, gt, weight[[common_ind], :])
            D1_gt = D1_gt.reshape(args.similar_size)
            d1_gt = list_max_like(D1_gt, beta=args.beta1)

            D2_gt = get_per_dis(Xl, gt, weight[[common_ind], :])
            D2_gt = D2_gt.reshape(args.similar_size)
            d2_gt = list_min_like(D2_gt, beta=args.beta1)
            loss_gt = max_like(
                d1_gt - d2_gt + sig,
                torch.FloatTensor([0.0]).to(device),
                beta=args.beta2,
            )
            sim_loss_gt += loss_gt

        # loss = max_like(d1-d2+args.dis, torch.FloatTensor([0.0]).to(device),
        #                 beta=args.beta2)
        loss = torch.sum(torch.max(
            d1 - d2 + args.dis, torch.FloatTensor([0.0]).to(device)
        ))
        # loss2 = max_like(
        #     d1 - d3 + args.dis, torch.FloatTensor([0.0]).to(device), beta=args.beta2
        # )
        # loss = 0.5 * (loss1 + loss2)

        sim_loss += loss

        n_tmp = n_tmp + torch.sum(lab)

    sim_loss = sim_loss / n_tmp
    if gt_all is not None:
        sim_loss_gt = sim_loss_gt / x.shape[0]
        return sim_loss, sim_loss_gt

    return sim_loss


def CASL(x, element_logits, weight, labels, seq_len, device, args, gt_all=None):
    """ x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 """

    sim_loss = 0.0
    n_tmp = 0.0
    for i in range(0, args.num_similar * 2, 2):
        atn1 = F.softmax(element_logits[i][: seq_len[i]], dim=0)
        atn2 = F.softmax(element_logits[i + 1][: seq_len[i + 1]], dim=0)

        n1 = torch.FloatTensor([np.maximum(seq_len[i] - 1, 1)]).to(device)
        n2 = torch.FloatTensor([np.maximum(seq_len[i + 1] - 1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), atn1)
        Hf2 = torch.mm(torch.transpose(x[i + 1][: seq_len[i + 1]], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), (1 - atn1) / n1)
        Lf2 = torch.mm(
            torch.transpose(x[i + 1][: seq_len[i + 1]], 1, 0), (1 - atn2) / n2
        )

        xh1 = get_unit_vector(Hf1, 0)
        xh2 = get_unit_vector(Hf2, 0)
        xl1 = get_unit_vector(Lf1, 0)
        xl2 = get_unit_vector(Lf2, 0)
        # weight = get_unit_vector(weight, -1)

        d1 = torch.pow(torch.sum((xh1 - xh2) * weight.transpose(0, 1), 0), 2)
        d2 = torch.pow(torch.sum((xh1 - xl2) * weight.transpose(0, 1), 0), 2)
        d3 = torch.pow(torch.sum((xh2 - xl1) * weight.transpose(0, 1), 0), 2)

        # d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
        #     torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0)
        # )
        # d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (
        #     torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0)
        # )
        # d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (
        #     torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0)
        # )

        # d2 = torch.min(d2, d3)
        d2 = (d2 + d3)/2

        sim_loss = sim_loss + 0.5 * 2  * torch.sum(
            torch.max(d1 - d2 + args.dis, torch.FloatTensor([0.0]).to(device))
            * Variable(labels[i, :])
            * Variable(labels[i + 1, :])
        )
        # sim_loss = sim_loss + 0.5 * torch.sum(
        #     torch.max(d1 - d3 + args.dis, torch.FloatTensor([0.0]).to(device))
        #     * Variable(labels[i, :])
        #     * Variable(labels[i + 1, :])
        # )
        n_tmp = n_tmp + torch.sum(Variable(labels[i, :]) * Variable(labels[i + 1, :]))
    sim_loss = sim_loss / n_tmp
    return sim_loss


def continuity_loss(element_logits, labels, seq_len, device):
    """ element_logits should be torch tensor of dimension (B, n_element, n_class),
    return is a torch tensor of dimension (B, n_class) """

    labels_var = Variable(labels)

    logit_masked = element_logits * labels_var.unsqueeze(1)  # B, n_el, n_cls
    logit_masked = logit_masked.to(device)
    logit_s = torch.sum(
        torch.abs((logit_masked[:, 1:, :] - logit_masked[:, :-1, :])), 1
    )
    logit_s = logit_s / torch.from_numpy(seq_len.astype(np.float32)).unsqueeze(-1).to(
        device
    )
    c_loss = torch.sum(logit_s) / element_logits.shape[0]
    return c_loss


def train(itr, dataset, args, model, optimizer, logger, device, scheduler=None):

    # #### gt #####
    # features = dataset.load_partial(is_random=True)
    # features = torch.from_numpy(features).float().to(device)
    # gt_features = model(Variable(features), is_tmp=True)

    features, labels = dataset.load_data(
        n_similar=args.num_similar, similar_size=args.similar_size
    )
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    final_features, element_logits = model(Variable(features))

    milloss = MILL_test(element_logits, seq_len, labels, device, args)

    weight = model.classifier.weight
    casloss = WLOSS_orig(
        final_features, element_logits, weight, labels, seq_len, device, args, None
    )
    # casloss = CASL(
    #     final_features, element_logits, weight, labels, seq_len, device, args, None
    # )

    # closs = continuity_loss(element_logits, labels, seq_len, device)

    total_loss = args.Lambda * milloss + (1 - args.Lambda) * casloss

    if torch.isnan(total_loss):
        import pdb

        pdb.set_trace()

    logger.log_value("weight", torch.norm(weight), itr)

    logger.log_value("milloss", milloss, itr)
    # logger.log_value("casloss", casloss, itr)
    logger.log_value("total_loss", total_loss, itr)

    # print("Iteration: %d, Loss: %.3f" % (itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    if scheduler:
        scheduler.step()
