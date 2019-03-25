import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Model
from video_dataset import Dataset
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
from classificationMAP import getClassificationMAP as cmAP
import time

torch.set_default_tensor_type("torch.cuda.FloatTensor")


def MILL(element_logits, seq_len, batch_size, labels, device):
    """ element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over,
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) """

    k = np.ceil(seq_len / 8).astype("int32")
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        tmp, _ = torch.topk(element_logits[i][: seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat(
            [instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0
        )
    milloss = -torch.mean(
        torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1),
        dim=0,
    )
    return milloss


def MILL_2(element_logits, seq_len, batch_size, labels, device, atn):
    """ element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over,
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) """

    k = np.ceil(seq_len / 8).astype("int32")
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    element_logits = element_logits * torch.sigmoid(atn)
    for i in range(batch_size):
        # atnb = F.softmax(atn[i][:seq_len[i]], 0)
        # # tmp, _ = torch.topk(element_logits[i][: seq_len[i]], k=int(k[i]), dim=0)
        # tmp = element_logits[i][:seq_len[i]] * atnb
        # instance_logits = torch.cat(
        #     [instance_logits, torch.sum(tmp, 0, keepdim=True)], dim=0
        # )

        tmp, _ = torch.topk(
            element_logits[i][: seq_len[i]],
            k=int(k[i]),
            dim=0,
        )
        instance_logits = torch.cat(
            [instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0
        )
    milloss = -torch.mean(
        torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1),
        dim=0,
    )
    return milloss


def CASL(x, element_logits, seq_len, n_similar, labels, device, atn):
    """ x is the torch tensor of feature from the last layer of model
        of dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 """

    sim_loss = 0.0
    n_tmp = 0.0
    element_logits = element_logits * torch.sigmoid(atn)
    for i in range(0, n_similar * 2, 2):
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

        d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
            torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0)
        )
        d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (
            torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0)
        )
        d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (
            torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0)
        )

        sim_loss = sim_loss + 0.5 * torch.sum(
            torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.0]).to(device))
            * Variable(labels[i, :])
            * Variable(labels[i + 1, :])
        )
        sim_loss = sim_loss + 0.5 * torch.sum(
            torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.0]).to(device))
            * Variable(labels[i, :])
            * Variable(labels[i + 1, :])
        )
        n_tmp = n_tmp + torch.sum(Variable(labels[i, :]) * Variable(labels[i + 1, :]))
    sim_loss = sim_loss / n_tmp
    return sim_loss


def distance(x, y, m):
    """ x : n_class, n_feat
    y : n_class, n_feat """
    xb = x.unsqueeze(1).repeat((1, x.shape[0], 1))
    yb = y.unsqueeze(1).repeat((1, y.shape[0], 1)).transpose(0, 1)

    dis = torch.sqrt(torch.sum(torch.pow(xb - yb, 2), -1))

    ndis = (1+np.exp(-m)) / (1 + torch.exp(dis - m))

    return ndis


def cosine_dis(x, y):
    """ x : n_class, n_feat
    y : n_class, n_feat """
    xb = x.unsqueeze(1).repeat((1, x.shape[0], 1))
    yb = y.unsqueeze(1).repeat((1, y.shape[0], 1)).transpose(0, 1)

    ndis = 1 - torch.cosine_similarity(xb, yb, dim=-1, eps=1e-8)

    return ndis


def CASL_2(x, element_logits, seq_len, labels, device, gt_feat_t, atn):
    """ x is the torch tensor of feature from the last layer of model of
        dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 """

    sim_loss = 0.0
    # n_tmp = 0.0
    gt_feat = torch.transpose(gt_feat_t, 0, 1)
    element_logits = element_logits * F.sigmoid(atn)
    for i in range(0, x.shape[0]):
        atn1 = F.softmax(element_logits[i][: seq_len[i]], dim=0)

        n1 = torch.FloatTensor([np.maximum(seq_len[i] - 1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), atn1)
        Lf1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), (1 - atn1) / n1)
        Hf2 = torch.transpose(gt_feat, 0, 1)

        d1 = 1 - torch.sum(Hf1 * gt_feat, dim=0) / (
            torch.norm(Hf1, 2, dim=0) * torch.norm(gt_feat, 2, dim=0)
        )

        eye = (1 - torch.eye(20)).to(device)
        mat = (
            torch.mm(gt_feat_t, Hf1)
            / (
                torch.sqrt(torch.mm(gt_feat_t, gt_feat))
                * torch.sqrt(torch.mm(torch.transpose(Hf1, 0, 1), Hf1))
                + 1e-8
            )
            * eye
        )

        # d2 = 1 - torch.max(mat, 0)[0]
        d2 = 1 - torch.sum(mat, 0) / torch.sum(eye, 0)

        # d2 = 1 - torch.mean(
        #     torch.mm(gt_feat_t, Hf1)
        #     / (
        #         torch.sqrt(torch.mm(gt_feat_t, gt_feat))
        #         * torch.sqrt(torch.mm(torch.transpose(Hf1, 0, 1), Hf1))
        #     ),
        #     0,
        # )

        d3 = 1 - torch.sum(gt_feat * Lf1, dim=0) / (
            torch.norm(gt_feat, 2, dim=0) * torch.norm(Lf1, 2, dim=0)
        )

        sim_loss = sim_loss + 0.5 * torch.sum(
            torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.0]).to(device))
            * Variable(labels[i, :])
        ) / torch.sum(Variable(labels[i, :]))

        sim_loss = sim_loss + 0.5 * torch.sum(
            torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.0]).to(device))
            * Variable(labels[i, :])
        ) / torch.sum(Variable(labels[i, :]))

        # n_tmp = n_tmp + torch.sum(Variable(labels[i, :]))
    sim_loss = sim_loss / x.shape[0]
    return sim_loss


def CASL_2_like(x, element_logits, seq_len, labels, device, gt_feat_t, atn):
    """ x is the torch tensor of feature from the last layer of model of
        dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 """

    sim_loss = 0.0
    # n_tmp = 0.0
    gt_feat = torch.transpose(gt_feat_t, 0, 1)
    element_logits = element_logits * F.sigmoid(atn)

    bceloss = torch.nn.BCELoss()

    for i in range(0, x.shape[0]):
        atn1 = F.softmax(element_logits[i][: seq_len[i]], dim=0)

        n1 = torch.FloatTensor([np.maximum(seq_len[i] - 1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), atn1)
        Lf1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), (1 - atn1) / n1)
        Hf2 = torch.transpose(gt_feat, 0, 1)

        dis = cosine_dis(Hf1.transpose(0, 1), gt_feat_t)

        lab = Variable(labels[i, :]).view(-1, 1)

        mask1 = lab * torch.eye(20)
        mask2 = lab * (1 - torch.eye(20))

        d1 = torch.sum(mask1 * dis) / torch.sum(mask1)
        d2 = torch.sum(mask2 * dis) / torch.sum(mask2)

        loss1 = torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.0]).to(device))

        dislf = cosine_dis(Lf1.transpose(0, 1), gt_feat_t)
        d3 = torch.diag(dislf)

        loss3 = torch.sum(
            torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.0]).to(device))
            * Variable(labels[i, :])
        ) / torch.sum(Variable(labels[i, :]))

        sim_loss += loss1 + loss3

        # sim_loss += 1/3 * (loss1 + loss2 + loss3)

    sim_loss = sim_loss / x.shape[0]
    return sim_loss


def CASL_3(x, element_logits, seq_len, labels, device, gt_feat_t, atn, args):
    """ x is the torch tensor of feature from the last layer of model of
        dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 """

    m = args.dis
    sim_loss = 0.0
    # n_tmp = 0.0
    gt_feat = torch.transpose(gt_feat_t, 0, 1)
    element_logits = element_logits * F.sigmoid(atn)

    bceloss = torch.nn.BCELoss()

    for i in range(0, x.shape[0]):
        atn1 = F.softmax(element_logits[i][: seq_len[i]], dim=0)

        n1 = torch.FloatTensor([np.maximum(seq_len[i] - 1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), atn1)
        Lf1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), (1 - atn1) / n1)
        Hf2 = torch.transpose(gt_feat, 0, 1)

        dis = distance(Hf1.transpose(0, 1), gt_feat_t, m)

        lab = Variable(labels[i, :]).view(-1, 1)

        mask1 = lab * torch.eye(20)
        mask2 = lab * (1 - torch.eye(20))

        loss1 = - torch.sum(mask1 * torch.log(dis)) / torch.sum(mask1)
        loss2 = - torch.sum(mask2 * torch.log(1 - dis)) / torch.sum(mask2)

        dislf = distance(Lf1.transpose(0, 1), gt_feat_t, m)
        d3 = torch.diag(dislf)

        loss3 = -torch.sum(torch.log(1 - d3) * Variable(labels[i, :])) \
            / torch.sum(Variable(labels[i, :]))

        sim_loss += 1/3 * (loss1 + loss2 + loss3)

    sim_loss = sim_loss / x.shape[0]
    return sim_loss


def continuity_loss(element_logits, labels, seq_len, batch_size, device):
    """ element_logits should be torch tensor of dimension (B, n_element, n_class),
    return is a torch tensor of dimension (B, n_class) """

    labels_var = Variable(labels)

    logit_masked = element_logits * labels_var.unsqueeze(1)  # B, n_el, n_cls
    logit_masked = logit_masked.to(device)
    logit_diff = torch.sum(
        torch.abs((logit_masked[:, 1:, :] - logit_masked[:, :-1, :])), 1
    )  # B, n_cls

    # labels_sum = torch.sum(labels_var, -1, keepdim=True) + 1e-8  # B, 1

    logit_s = logit_diff  # / labels_sum  # B, n_cls
    logit_s = logit_s / torch.from_numpy(seq_len.astype(np.float32)).unsqueeze(-1).to(
        device
    )
    c_loss = torch.sum(logit_s) / batch_size
    return c_loss


def l1loss(atn, seq_len):
    loss = 0
    atn = torch.sigmoid(atn)
    for i in range(atn.shape[0]):
        loss += torch.sum(atn[i][:seq_len[i]]) / seq_len[i]
    loss = loss / atn.shape[0]
    return loss


def train(itr, dataset, args, model, optimizer, logger, device, scheduler=None):

    #####
    features = dataset.load_partial(is_random=False)
    features = torch.from_numpy(features).float().to(device)
    model.train(False)
    gt_features = model(Variable(features), is_tmp=True)
    model.train(True)

    features, labels = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    final_features, element_logits, atn = model(Variable(features))

    milloss = MILL_2(element_logits, seq_len, args.batch_size, labels, device, atn)
    # casloss = CASL(
    #     final_features, element_logits, seq_len, args.num_similar, labels, device, atn
    # )
    # casloss2 = CASL_3(
    #     final_features, element_logits, seq_len, labels, device, gt_features, atn, args
    # )

    casloss2 = CASL_2_like(final_features, element_logits, seq_len,
                           labels, device, gt_features, atn)

    # contloss = continuity_loss(element_logits, labels, seq_len, args.batch_size, device)

    # total_loss = milloss

    total_loss = args.Lambda * milloss + (1 - args.Lambda) * (casloss2)

    if torch.isnan(total_loss):
        import pdb
        pdb.set_trace()

    logger.log_value("milloss", milloss, itr)
    logger.log_value('casloss', casloss2, itr)
    logger.log_value("total_loss", total_loss, itr)

    # print(f'{itr} : loss : ', [total_loss.data.cpu(), milloss.data.cpu(), casloss.data.cpu()])

    print("Iteration: %d, Loss: %.3f" % (itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if scheduler:
        scheduler.step()

