# -*- coding:utf-8 -*-
# Author:Richard Fang

import time
import numpy as np
import torch


def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[x1, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [x1,x2]
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = box_scores
    areas = (x2 - x1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        xx1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        inter = torch.tensor(w).cuda() if cuda else torch.tensor(w)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 2][scores > thresh].int()

    return keep


def speed():
    boxes = 1000*torch.rand((1000, 100, 2), dtype=torch.float)
    boxscores = torch.rand((1000, 100), dtype=torch.float)

    # cuda flag
    cuda = 1 if torch.cuda.is_available() else 0
    if cuda:
        boxes = boxes.cuda()
        boxscores = boxscores.cuda()

    start = time.time()
    for i in range(1000):
        soft_nms_pytorch(boxes[i], boxscores[i], cuda=cuda)
    end = time.time()
    print("Average run time: %f ms" % (end-start))


def test():
    # boxes and boxscores
    boxes = torch.tensor([[200, 400],
                          [220, 420],
                          [240, 440],
                          [200, 400],
                          [1, 2]], dtype=torch.float)
    boxscores = torch.tensor([0.8, 0.7, 0.6, 0.5, 0.9], dtype=torch.float)

    # cuda flag
    cuda = 1 if torch.cuda.is_available() else 0
    if cuda:
        boxes = boxes.cuda()
        boxscores = boxscores.cuda()

    print(soft_nms_pytorch(boxes, boxscores, cuda=cuda))


if __name__ == '__main__':
    test()
    # speed()





