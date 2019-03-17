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
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def topk_loss(element_logits, seq_len, batch_size, labels, device):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over,
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''

    k = np.ceil(seq_len/8).astype('int32')
    # labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    # milloss = F.binary_cross_entropy_with_logits(
    #     instance_logits, labels
    # )
    milloss = -torch.mean(
                torch.sum(Variable(labels) *
                F.log_softmax(instance_logits, dim=1), dim=-1), dim=0)
    return milloss


def milloss(element_logits, batch_size, labels, device):
    # labels = labels / torch.sum(labels, dim=1, keepdim=True)
    milloss = -torch.mean(
                torch.sum(Variable(labels) *
                F.log_softmax(element_logits, dim=1), dim=-1), dim=0)
    # milloss = F.binary_cross_entropy_with_logits(
    #     element_logits.squeeze(), labels
    # )
    return milloss


def train(itr, dataset, args, model, optimizer, logger, device,
          valid=False):

    features, labels = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, :np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    batch_size = labels.shape[0]

    model.train()

    x_class, _ = model(Variable(features))

    loss_mil = milloss(x_class, batch_size, labels, device)

    total_loss = loss_mil
    # logger.log_value("train_milloss", loss_mil, itr)
    logger.log_value('train_total_loss', total_loss, itr)

    train_loss = total_loss.data.cpu().numpy()

    if not valid:
        print('Iteration: %d, Loss: %.3f' % (itr, train_loss))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if valid:
        model.eval()
        with torch.no_grad():
            features, labels = dataset.load_valid()
            seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
            features = features[:, :np.max(seq_len), :]
            features = torch.from_numpy(features).float().to(device)
            labels = torch.from_numpy(labels).float().to(device)
            x_class, _ = model(Variable(features))

            loss_mil = milloss(x_class, batch_size, labels, device)
            val_total_loss = loss_mil

            logger.log_value("val_mil_loss", val_total_loss, itr)

            val_loss = val_total_loss.data.cpu().numpy()
            print('Iteration: %d, Train Loss: %.4f  Valid Loss: %.4f' %
                  (itr, train_loss, val_loss))




"""
    x_class, x_class_init, w_mean = model(Variable(features))

    # sparsity loss
    loss_sparse = 0.5 - w_mean

    # difference loss
    init_loss = topk_loss(x_class_init, seq_len, batch_size, labels, device)
    final_loss = milloss(x_class, batch_size, labels, device)

    loss_diff = F.relu(final_loss - init_loss + 0.1)

    a, b = 0.1, 0.001

    total_loss = final_loss + init_loss + a * loss_diff + b * loss_sparse

    logger.log_value('train_diff_loss', loss_diff, itr)
    logger.log_value('train_final_loss', final_loss, itr)
    logger.log_value('train_total_loss', total_loss, itr)
    logger.log_value('train_sparse_loss', loss_sparse, itr)

    train_loss = total_loss.data.cpu().numpy()

    if not valid:
        print('Iteration: %d, Loss: %.3f' % (itr, train_loss))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if valid:
        model.eval()
        with torch.no_grad():
            features, labels = dataset.load_valid()
            seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
            features = features[:, :np.max(seq_len), :]
            features = torch.from_numpy(features).float().to(device)
            labels = torch.from_numpy(labels).float().to(device)
            x_class, x_class_init, w_mean = model(Variable(features))

            # sparsity loss
            loss_sparse = 0.5 - w_mean

            # difference loss
            init_loss = topk_loss(x_class_init, seq_len, batch_size, labels, device)
            final_loss = milloss(x_class, batch_size, labels, device)

            loss_diff = F.relu(final_loss - init_loss + 0.1)

            val_total_loss = final_loss + init_loss + a * loss_diff + b * loss_sparse

            if torch.isnan(val_total_loss).any():
                print("nan found")
                raise Exception("Nan value found")

            logger.log_value('val_diff_loss', loss_diff, itr)
            logger.log_value('val_final_loss', final_loss, itr)
            logger.log_value('val_sparse_loss', loss_sparse, itr)
            logger.log_value('val_total_loss', total_loss, itr)

            logger.log_value('val_total_loss', val_total_loss, itr)

            val_loss = val_total_loss.data.cpu().numpy()
            print('Iteration: %d, Train Loss: %.4f  Valid Loss: %.4f' %
                  (itr, train_loss, val_loss))

    # logger.log_value('total_loss', total_loss, itr)
"""

