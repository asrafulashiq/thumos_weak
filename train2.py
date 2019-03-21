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


def CASL(x, element_logits, seq_len, n_similar, labels, device):
    """ x is the torch tensor of feature from the last layer of model
        of dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 """

    sim_loss = 0.0
    n_tmp = 0.0
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


def CASL_2(x, element_logits, seq_len, labels, device, gt_feat_t):
    """ x is the torch tensor of feature from the last layer of model of
        dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 """

    sim_loss = 0.0
    # n_tmp = 0.0
    gt_feat = torch.transpose(gt_feat_t, 0, 1)
    for i in range(0, x.shape[0]):
        atn1 = F.softmax(element_logits[i][: seq_len[i]], dim=0)

        n1 = torch.FloatTensor([np.maximum(seq_len[i] - 1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), atn1)
        Lf1 = torch.mm(torch.transpose(x[i][: seq_len[i]], 1, 0), (1 - atn1) / n1)
        Hf2 = torch.transpose(gt_feat, 0, 1)

        d1 = 1 - torch.sum(Hf1 * gt_feat, dim=0) / (
            torch.norm(Hf1, 2, dim=0) * torch.norm(gt_feat, 2, dim=0)
        )

        mat = torch.mm(gt_feat_t, Hf1) / (
            torch.sqrt(torch.mm(gt_feat_t, gt_feat))
            * torch.sqrt(torch.mm(torch.transpose(Hf1, 0, 1), Hf1))
            + 1e-8
        ) * (1 - torch.eye(20)).to(device)

        d2 = 1 - torch.max(mat, 0)[0]

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


def continuity_loss(element_logits, labels, seq_len, batch_size, device):
    """ element_logits should be torch tensor of dimension (B, n_element, n_class),
    return is a torch tensor of dimension (B, n_class) """

    labels_var = Variable(labels)

    logit_masked = element_logits * labels_var.unsqueeze(1)  # B, n_el, n_cls
    logit_masked = logit_masked.to(device)
    logit_diff = torch.sum(
        torch.abs((logit_masked[:, 1:, :] - logit_masked[:, :-1, :])),
        1)  # B, n_cls

    # labels_sum = torch.sum(labels_var, -1, keepdim=True) + 1e-8  # B, 1

    logit_s = logit_diff  # / labels_sum  # B, n_cls
    logit_s = logit_s / torch.from_numpy(
        seq_len.astype(np.float32)
    ).unsqueeze(-1).to(device)
    c_loss = torch.sum(logit_s) / batch_size
    return c_loss


def train(itr, dataset, args, model, optimizer, logger, device, scheduler=None):

    #####
    features = dataset.load_partial(is_random=True)
    features = torch.from_numpy(features).float().to(device)
    model.train(False)
    gt_features = model(Variable(features), is_tmp=True)
    model.train(True)

    features, labels = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    final_features, element_logits = model(Variable(features))

    milloss = MILL(element_logits, seq_len, args.batch_size, labels, device)
    casloss = CASL(final_features, element_logits, seq_len, args.num_similar, labels, device)
    casloss2 = CASL_2(
        final_features, element_logits, seq_len, labels, device, gt_features
    )

    contloss = continuity_loss(element_logits, labels, seq_len, args.batch_size, device)

    if torch.isnan(casloss).any() or torch.isnan(gt_features).any() or \
            torch.isnan(contloss).any():
        import pdb
        pdb.set_trace()

    # total_loss = milloss

    total_loss = args.Lambda * milloss + (1 - args.Lambda)/2 * (casloss + casloss2)

    logger.log_value("milloss", milloss, itr)
    # logger.log_value('casloss', casloss, itr)
    logger.log_value("total_loss", total_loss, itr)

    # print(f'{itr} : loss : ', [total_loss.data.cpu(), milloss.data.cpu(), casloss.data.cpu()])

    print("Iteration: %d, Loss: %.3f" % (itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    if scheduler:
        scheduler.step()

