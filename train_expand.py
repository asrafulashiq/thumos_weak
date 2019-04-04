import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

torch.set_default_tensor_type("torch.cuda.FloatTensor")


def MILL_all(element_logits, seq_len, batch_size, labels, device):
    """ element_logits should be torch tensor of dimension
        (B, n_element, n_class),
        k should be numpy array of dimension (B,) indicating the top k
        locations to average over,
        labels should be a numpy array of dimension (B, n_class) of 1 or 0
        return is a torch tensor of dimension (B, n_class) """

    k = np.ceil(seq_len / 8).astype("int32")
    # labels = labels / torch.sum(labels, dim=1, keepdim=True)
    # instance_logits = torch.zeros(0).to(device)
    eps = 1e-8
    loss = 0
    for i in range(batch_size):
        tmp, _ = torch.topk(
            element_logits[i][: seq_len[i]], k=int(k[i]), dim=0)

        topk = torch.sigmoid(torch.mean(tmp, 0))
        lab = Variable(labels[i])
        loss1 = -torch.sum(lab * torch.log(topk+eps)) / torch.sum(lab)
        loss2 = -torch.sum((1-lab) * torch.log(1-topk+eps)) / torch.sum(1-lab)

        loss += 1/2 * (loss1 + loss2)

        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

    milloss = loss / batch_size
    return milloss


def get_unit_vector(x):
    # return x
    return x / torch.norm(x, 2, dim=0, keepdim=True)


def max_like(a, b, beta=1):
    return 1/beta * torch.log(torch.exp(beta*a) + torch.exp(beta*b))
    # return torch.max(a, b)


def list_max_like(x, beta=1000):
    return 1/beta * torch.logsumexp(beta*x, -1)


def list_min_like(x, beta=100):
    return -list_max_like(-x, beta=beta)
    # return torch.min(x, dim=-1)[0]


def get_per_dis(x1, x2, w):
    w = w.unsqueeze(1)
    x1 = x1.transpose(0, 1).unsqueeze(1)
    x2 = x2.transpose(0, 1).unsqueeze(0)
    x_diff = x1 - x2

    dis_mat = torch.pow(torch.sum(x_diff * w, -1), 2)
    # dis_mat = dis_mat.reshape(w.shape[0]**2)

    return dis_mat


def WLOSS_orig(x, element_logits, weight, labels,
               n_similar, seq_len, device, args):

    sim_loss = 0.0
    sig = args.dis
    labels = Variable(labels)
    for i in range(0, n_similar * 2, 2):

        lab = labels[i, :] * labels[i + 1, :]

        common_ind = lab.nonzero().squeeze(-1)

        atn1 = F.softmax(element_logits[i][:seq_len[i], common_ind], dim=0)
        atn2 = F.softmax(element_logits[i+1][:seq_len[i+1], common_ind], dim=0)

        atn1_l = F.softmin(element_logits[i][:seq_len[i], common_ind], dim=0)
        atn2_l = F.softmin(element_logits[i+1][:seq_len[i+1],
                                               common_ind], dim=0)

        xh1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), atn1)
        xh2 = torch.mm(torch.transpose(x[i + 1][: seq_len[i + 1]], 1, 0), atn2)
        xl1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), atn1_l)
        xl2 = torch.mm(
            torch.transpose(x[i + 1][: seq_len[i + 1]], 1, 0), atn2_l)

        xh1_h = get_unit_vector(xh1)
        xh2_h = get_unit_vector(xh2)
        xl1_h = get_unit_vector(xl1)
        xl2_h = get_unit_vector(xl2)

        # d1 = torch.pow(torch.mm(weight[common_ind, :], xh1_h - xh2_h), 2)

        # d2 = torch.pow(torch.mm(weight[common_ind, :], xh1_h - xl2_h), 2)

        d1 = torch.pow(torch.sum(weight[common_ind, :] *
                       (xh1_h - xh2_h).transpose(0, 1), -1), 2)
        dis_neg_1 = torch.pow(torch.sum(
            weight[common_ind, :] *
            (xh1_h - xl2_h).transpose(0, 1), -1, keepdim=True), 2)
        dis_neg_2 = torch.pow(torch.sum(
            weight[common_ind, :] *
            (xh2_h - xl1_h).transpose(0, 1), -1, keepdim=True), 2)

        # get negetive instance
        for j in range(x.shape[0]):
            if j in [i, i + 1]:
                continue
            cur_lab = labels[j, :]
            uncommon_ind = (1 - lab) * cur_lab
            if torch.sum(uncommon_ind) != 0:
                ind = uncommon_ind.nonzero().squeeze(-1)
                _atn = F.softmax(element_logits[j][: seq_len[j], ind],
                                 dim=0)
                xtmp = torch.mm(torch.transpose(x[j][:seq_len[j]], 1, 0),
                                _atn)
                xtmp = get_unit_vector(xtmp)
                _dis = get_per_dis(xh1_h, xtmp, weight[common_ind, :])
                dis_neg_1 = torch.cat(
                    [dis_neg_1, _dis], dim=1
                )
                _dis = get_per_dis(xh2_h, xtmp, weight[common_ind, :])
                dis_neg_2 = torch.cat(
                    [dis_neg_2, _dis], dim=1
                )
        d2 = list_min_like(dis_neg_1)
        d3 = list_min_like(dis_neg_2)

        # d3 = torch.pow(torch.mm(weight[common_ind, :], xh2_h - xl1_h), 2)

        sim_loss = sim_loss + 0.5 * torch.sum(
            max_like(d1 - d2 + sig, torch.FloatTensor([0.0]).to(device))
        ) / torch.sum(lab)
        sim_loss = sim_loss + 0.5 * torch.sum(
            max_like(d1 - d3 + sig, torch.FloatTensor([0.0]).to(device))
        ) / torch.sum(lab)

    sim_loss = sim_loss / x.shape[0]
    return sim_loss


def train(itr, dataset, args, model, optimizer,
          logger, device, scheduler=None):

    #####
    # features = dataset.load_partial(is_random=True)
    # features = torch.from_numpy(features).float().to(device)
    # # model.train(False)
    # gt_features = model(Variable(features), is_tmp=True)
    # # model.train(True)

    features, labels = dataset.load_data(
        n_similar=args.num_similar,
        similar_size=args.similar_size)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    final_features, element_logits = model(Variable(features))

    milloss = MILL_all(element_logits, seq_len,
                       args.batch_size, labels, device)

    weight = model.classifier.weight
    casloss = WLOSS_orig(final_features, element_logits, weight,
                         labels, args.num_similar,
                         seq_len, device, args)
    total_loss = args.Lambda * milloss + (1 - args.Lambda) * (casloss)

    if torch.isnan(total_loss):
        import pdb
        pdb.set_trace()

    logger.log_value("milloss", milloss, itr)
    logger.log_value('casloss', casloss, itr)
    logger.log_value("total_loss", total_loss, itr)

    print("Iteration: %d, Loss: %.3f" % (itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if scheduler:
        scheduler.step()
