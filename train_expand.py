import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

torch.set_default_tensor_type("torch.cuda.FloatTensor")


def MILL_all(element_logits, seq_len, labels, device):
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
    for i in range(element_logits.shape[0]):
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

    milloss = loss / element_logits.shape[0]
    return milloss


def get_unit_vector(x):
    # return x
    return x / torch.norm(x, 2, dim=0, keepdim=True)


def max_like(a, b, beta=10):
    return 1/beta * torch.log(torch.exp(beta*a) + torch.exp(beta*b))
    # return torch.max(a, b)


def list_max_like(x, beta=100):
    return 1/beta * torch.logsumexp(beta*x, -1)
    # return torch.max(x, dim=-1)[0]


def list_min_like(x, beta=100):
    return -list_max_like(-x, beta=beta)
    # return torch.min(x, dim=-1)[0]


def get_per_dis(x1, x2, w):
    w = w.unsqueeze(1)
    x1 = x1.transpose(0, 1).unsqueeze(1)
    x2 = x2.transpose(0, 1).unsqueeze(0)
    x_diff = x1 - x2

    dis_mat = torch.pow(torch.sum(x_diff * w, -1), 2)
    # dis_mat = torch.norm(x_diff, 2, -1)
    # dis_mat = dis_mat.reshape(w.shape[0]**2)

    return dis_mat


def WLOSS_orig(x, element_logits, weight, labels,
               n_similar, seq_len, device, args):

    sim_loss = 0.0
    sig = args.dis
    labels = Variable(labels)
    for i in range(0, n_similar * args.similar_size, args.similar_size):

        lab = labels[i, :]
        for k in range(i+1, i+args.similar_size):
            lab = lab * labels[k, :]

        common_ind = lab.nonzero().squeeze(-1)[0]

        Xh = torch.Tensor()
        Xl = torch.Tensor()

        for k in range(i, i+args.similar_size):
            atn = F.softmax(
                element_logits[k][:seq_len[k], [common_ind]], dim=0
            )
            atn_l = F.softmin(
                element_logits[k][:seq_len[k], [common_ind]], dim=0
            )
            xh = torch.mm(torch.transpose(x[k][:seq_len[k]], 1, 0),
                          atn)
            xl = torch.mm(torch.transpose(x[k][:seq_len[k]], 1, 0),
                          atn_l)
            Xh = torch.cat([Xh, xh], dim=1)
            Xl = torch.cat([Xl, xl], dim=1)

        Xh = get_unit_vector(Xh)
        Xl = get_unit_vector(Xl)

        D1 = get_per_dis(Xh, Xh, weight[[common_ind], :])
        D1 = D1.reshape(args.similar_size**2)

        D2 = get_per_dis(Xh, Xl, weight[[common_ind], :])
        D2 = D2.reshape(args.similar_size**2)

        d1 = list_max_like(D1, beta=args.beta1)
        d2 = list_min_like(D2, beta=args.beta1)

        loss = max_like(d1-d2+sig, torch.FloatTensor([0.0]).to(device),
                        beta=args.beta2)

        sim_loss += loss

    sim_loss = sim_loss / x.shape[0]
    return sim_loss


def train(itr, dataset, args, model, optimizer,
          logger, device, scheduler=None):

    features, labels = dataset.load_data(
        n_similar=args.num_similar,
        similar_size=args.similar_size)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    final_features, element_logits = model(Variable(features))

    milloss = MILL_all(element_logits, seq_len, labels, device)

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
