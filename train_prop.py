import torch
import torch.nn.functional as F
import numpy as np


def smooth_tensor(x, dim=-1, sigma=3):
    b, c, l = x.shape
    gx = np.arange(-sigma, sigma + 1)
    k_gauss = np.exp(-gx ** 2 / (2.0 * (sigma / 3.0) ** 2))
    k_gauss = k_gauss / k_gauss.sum()
    kernel = torch.FloatTensor(k_gauss).reshape(1, 1, -1).repeat(c, 1, 1)
    # Create input
    x_smooth = F.conv1d(x, kernel.to(x.device), groups=c, padding=sigma)
    return x_smooth


def MILL_atn(elements_cls, seq_len, labels, device):
    
    labels = labels / labels.sum(-1)
    # make T x C gate
    # --> B, T, cls
    gate = torch.sigmoid(elements_cls[..., 1:] - elements_cls[..., [0]])

    # gated temporal average pooling
    # --> B, cls
    gated_tap = torch.sum(gate * elements_cls[..., 1:], dim=-2) / (torch.sum(gate, dim=-2)+1e-4)
    # --> B, 1
    back_tap = elements_cls[..., [0]].sum(-2) / elements_cls.shape[-2]
    tap = torch.cat((back_tap, gated_tap), dim=-1)  # --> B, cls + 1

    wb = 1. / labels.sum(-1, keepdim=True)  # B, 1
    milloss = torch.mean(-torch.sum(labels * torch.log(tap[..., 1:]+1e-4) + wb * torch.log(tap[..., [0]]+1e-4)))

    # select maximum score at each snippet for positive class
    loss_cont = 0
    for i in range(labels.shape[0]):  # process each batch
        elem = elements_cls[i, :, 1:]
        elem = elem[:, labels[i] > 0]  # T, pos_class
        elem_max = torch.max(elem, -1, keepdim=True)  # T, 1
        elem_back = elem[:, [0]]  # T, 1

        pp = torch.sum(torch.max(elem_max * elem_back + 0.5, torch.zeros_like(elem_back)), dim=0)
        pp_norm = (elem_max.norm(dim=0) * elem_back.norm(dim=0) + 1e-4)
        loss_cont += pp / pp_norm
    loss_cont = loss_cont / labels.shape[0]
    return milloss, loss_cont


def get_bg_feat(x, elements_atn):
    _bg = 1 - elements_atn.sigmoid()
    atn_bg = _bg / _bg.sum(-1, keepdim=True)
    x_bg = (atn_bg * x).sum(-1)

    return x_bg



def t_val(x):
    if not isinstance(x, torch.Tensor):
        return x
    else:
        if x.is_cuda:
            return x.data.cpu().numpy()
        else:
            return x.data.numpy()




def train_lpat(itr, dataset, args, model, optimizer, logger, device):
    model.train()
    features, labels = dataset.load_data()
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    # features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    element_cls = model(features)

    milloss, loss_cont = MILL_atn(element_cls, seq_len, labels, device)
    total_loss = milloss + args.gamma * loss_cont

    print(f"{itr: >10d}: {t_val(milloss):.4f} + {t_val(loss_cont): .4f} = {t_val(total_loss): .4f}")
    # print("Iteration: %d, Loss: %.4f" % (itr, total_loss.data.cpu().numpy()))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.data.cpu().numpy()
