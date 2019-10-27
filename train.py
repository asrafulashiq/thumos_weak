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

    final_features, element_logits = model(Variable(features))

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
