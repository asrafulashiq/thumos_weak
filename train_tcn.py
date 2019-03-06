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


def train(itr, dataset, args, model, optimizer, logger, device,
          valid=False):

    features, labels = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, :np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    model.train()
    element_logits = model(Variable(features))

    ml_loss = -torch.mean(torch.sum(Variable(labels) *
                          F.log_softmax(element_logits, dim=-1), dim=-1), dim=0)

    # ml_loss = F.binary_cross_entropy_with_logits(
    #     element_logits, labels
    # )

    total_loss = ml_loss
    logger.log_value('train_total_loss', total_loss, itr)

    train_loss = total_loss.data.cpu().numpy()

    if not valid:
        print('Iteration: %d, Loss: %.3f' % (itr, train_loss))

    optimizer.zero_grad()
    total_loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    if valid:
        model.eval()
        with torch.no_grad():
            features, labels = dataset.load_valid()
            seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
            features = features[:, :np.max(seq_len), :]
            features = torch.from_numpy(features).float().to(device)
            labels = torch.from_numpy(labels).float().to(device)
            element_logits = model(Variable(features), is_training=False)

            val_milloss = -torch.mean(torch.sum(Variable(labels) *
                          F.log_softmax(element_logits, dim=-1), dim=-1), dim=0)
            # val_milloss = F.binary_cross_entropy_with_logits(
            #     element_logits, labels
            # )
            val_total_loss = val_milloss

            logger.log_value('val_total_loss', val_total_loss, itr)

            val_loss = val_total_loss.data.cpu().numpy()
            print('Iteration: %d, Train Loss: %.4f  Valid Loss: %.4f' %
                  (itr, train_loss, val_loss))

    # logger.log_value('total_loss', total_loss, itr)


