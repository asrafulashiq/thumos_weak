import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def MILL(element_logits, seq_len, labels, device):
    k = np.ceil(seq_len / 8).astype("int32")
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(element_logits.shape[0]):
        tmp, _ = torch.topk(
            element_logits[i][: seq_len[i]], k=int(k[i]), dim=0
        )
        instance_logits = torch.cat(
            [instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0
        )
    milloss = -torch.mean(
        torch.sum(
            labels * F.log_softmax(instance_logits, dim=1), dim=1
        ),
        dim=0,
    )
    return milloss

def MILL_atn(elements_cls, elements_atn, seq_len, labels, device):
    labels = labels / torch.sum(labels, dim=1, keepdim=True)

    # --> B, cls+1
    x_fg_cls = (elements_atn * elements_cls).sum(-1) / elements_cls.shape[-1]

    milloss = -torch.mean(
        torch.sum(
            labels * F.log_softmax(x_fg_cls, dim=1)[:, 1:], dim=1
        ),
        dim=0,
    )
    return milloss


def t_val(x):
    if not isinstance(x, torch.Tensor):
        return x
    else:
        if x.is_cuda:
            return x.data.cpu().numpy()
        else:
            return x.data.numpy()


def train(itr, dataset, args, model, optimizer, logger, device):
    model.train()
    features, labels = dataset.load_data(
        n_similar=args.num_similar, similar_size=args.similar_size
    )
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    elements_cls, elements_atn = model(features)
    # --> (B, cls+1, T), (B, 1, T)
    milloss = MILL_atn(elements_cls, elements_atn, seq_len, labels, device)

    total_loss = milloss  #+ args.gamma * metric_loss + args.gamma2 * L1loss

    print("Iteration: %d, Loss: %.4f" % (itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.data.cpu().numpy()
