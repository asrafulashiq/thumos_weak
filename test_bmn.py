import torch
import torch.nn.functional as F
import utils
import numpy as np
from torch.autograd import Variable
from classificationMAP import getClassificationMAP as cmAP
from detectionMAP import getDetectionMAP as dmAP
import scipy.io as sio
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from eval_detection import ANETdetection
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed

import torch
import torch.nn.functional as F
import utils
import numpy as np
from torch.autograd import Variable
from classificationMAP import getClassificationMAP as cmAP
from detectionMAP import getDetectionMAP as dmAP
import scipy.io as sio
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from eval_detection import ANETdetection


@torch.no_grad()
def test_bmn(itr, dataset, args, model, logger, device):

    model.eval()
    done = False
    instance_logits_stack = []
    element_logits_stack = []
    labels_stack = []
    ind_stack = []
    segment_predict =  []

    iou = [0.1, 0.3, 0.5, 0.7]
    dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args)

    for counter, (features, labels, idx) in tqdm(enumerate(dataset.load_test(shuffle=True))):

        if features.shape[0] > args.max_seqlen:
            continue

        features, labels, done = dataset.load_data(is_training=False)
        # features, flag = utils.len_extract(features, args.max_seqlen)

        features = torch.from_numpy(features).float().to(device)
        features = features.unsqueeze(0)
        element_cls, element_atn, conf_map = model(features, is_training=False)
        # atn_fg = torch.softmax(elements_atn, -1)
        atn_fg = element_atn.sigmoid() / element_atn.sigmoid().sum(-1, keepdim=True)
        y_cls = (element_cls * atn_fg).sum(-1) #/ element_atn.shape[-1]
        y_cls = y_cls.squeeze()
        element_logits = (element_cls).permute(0, 2, 1)
        element_logits = element_logits.squeeze(0)[..., 1:]
        tmp = (
            torch.softmax(
                y_cls, -1
            )[1:]  # omit the background class
            .cpu()
            .data.numpy()
        )
        element_logits = element_logits.cpu().data.numpy()

        # if flag is not None:
        #     if flag[0] == "pad":
        #         _seq = flag[1]
        #         # conf_map = conf_map[..., :, :_seq]
        #         element_logits = element_logits[:_seq]
        #     seglen = flag[1]
        # else:
        #     seglen = conf_map.shape[-1]

        # seglen = conf_map.shape[-1]

        # get prediction
        # _mask = torch.flip(torch.tril(torch.ones_like(conf_map), diagonal=-1), dims=[-1])
        # conf_map[_mask > 0] = -10000

        # conf_map = conf_map[:, :, 1:].squeeze().data.cpu().numpy()
        # for c in range(args.num_class):
        #     if tmp[c] < 0.1:
        #         continue
        #     tmp_conf_s = conf_map[0, c]  # --> D, T
        #     tmp_conf_m = conf_map[1, c]  # --> D, T
        #     tmp_conf_e = conf_map[2, c]  # --> D, T

        #     threshold = (element_logits[:, c].max() - element_logits[:, c].min())*0.5 + \
        #             element_logits[:, c].min()
            
        #     tmp_conf_score_left = tmp_conf_m-tmp_conf_s
        #     tmp_conf_score_right = tmp_conf_m-tmp_conf_e
        #     tmp_conf_score = 0.5 * (tmp_conf_score_left + tmp_conf_score_right)

        #     # tmp_mask = (tmp_conf_m > threshold) & (tmp_conf_s < threshold) & \
        #     #     (tmp_conf_e < threshold)
        #     tmp_mask = (tmp_conf_m > threshold) & ((tmp_conf_s < threshold) | (tmp_conf_e < threshold))

        #     # tmp_conf_score = tmp_conf_score * tmp_mask
        #     tmp_conf_score[tmp_mask < 0] = -1000

        #     _len = tmp_conf_s.shape[-1]

        #     for each_ind in zip(*tmp_mask.nonzero()):
        #         d, s = each_ind
        #         e = s + d
        #         if d < 1:
        #             continue
                
        #         scr = float(0.1 * tmp_conf_score[d, s] + np.mean(element_logits[s: e+1, c]))
        #         scr = scr #* y_cls[c+1].sigmoid().data.cpu().numpy()
        #         if scr > 0 and tmp_conf_score_left[d, s]>0 and tmp_conf_score_right[d,s]>0:
        #             segment_predict.append(
        #                 [idx, s/_len, e/_len, scr, c, seglen]
        #             )

        instance_logits_stack.append(tmp)
        element_logits_stack.append(element_logits)
        labels_stack.append(labels)
        ind_stack.append(idx)

        if counter > 50:
            break

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)
    
    # dmap_detect._import_prediction_bmn(segment_predict)
    dmap_detect._import_prediction(element_logits_stack)
    dmap = dmap_detect.evaluate(ind_to_keep=ind_stack)
    # dmap_detect.save_info("info.pkl")

    if args.dataset_name == "Thumos14":
        test_set = sio.loadmat("test_set_meta.mat")["test_videos"][0]
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]["background_video"] == "YES":
                labels_stack[i, :] = np.zeros_like(labels_stack[i, :])

    cmap = cmAP(instance_logits_stack, labels_stack)
    print("Classification map %f" % cmap)
    for k in range(len(iou)):
        print("Detection map @ %f = %f" % (iou[k], dmap[k] * 100))

    logger.add_scalar("Test Classification mAP", cmap, itr)
    for item in list(zip(dmap, iou)):
        logger.add_scalar(
            "Test Detection mAP @ IoU = " + str(item[1]), item[0], itr
        )

    # utils.write_to_file(args.dataset_name, dmap, cmap, itr)