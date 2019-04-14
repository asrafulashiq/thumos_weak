import torch
import torch.nn.functional as F
import utils
import numpy as np
from torch.autograd import Variable
from classificationMAP import getClassificationMAP as cmAP
from detectionMAP2 import getDetectionMAP as dmAP
import scipy.io as sio
from sklearn.metrics import accuracy_score

from eval_detection import ANETdetection

torch.set_default_tensor_type("torch.cuda.FloatTensor")


def test(itr, dataset, args, model, logger, device):

    done = False
    instance_logits_stack = []
    element_logits_stack = []
    labels_stack = []
    while not done:
        if dataset.currenttestidx % 100 == 0:
            print(
                "Testing test data point %d of %d"
                % (dataset.currenttestidx, len(dataset.testidx))
            )

        features, labels, done = dataset.load_data(is_training=False)
        features = torch.from_numpy(features).float().to(device)

        with torch.no_grad():
            features = features.unsqueeze(0)
            _, element_logits = model(Variable(features), is_training=False)
            element_logits = element_logits.squeeze(0)
        tmp = (
            F.softmax(
                torch.mean(
                    torch.topk(
                        element_logits, k=int(np.ceil(len(features) / 8)), dim=0
                    )[0],
                    dim=0,
                ),
                dim=0,
            )
            .cpu()
            .data.numpy()
        )
        element_logits = element_logits.cpu().data.numpy()

        instance_logits_stack.append(tmp)
        element_logits_stack.append(element_logits)
        labels_stack.append(labels)

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)

    # dmap, iou = dmAP(element_logits_stack, dataset.path_to_annotations, args)

    iou = [0.1, 0.3, 0.5]

    dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args)
    dmap_detect._import_prediction(element_logits_stack)
    dmap = dmap_detect.evaluate()

    if args.dataset_name == "Thumos14":
        test_set = sio.loadmat("test_set_meta.mat")["test_videos"][0]
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]["background_video"] == "YES":
                labels_stack[i, :] = np.zeros_like(labels_stack[i, :])

    cmap = cmAP(instance_logits_stack, labels_stack)
    print("Classification map %f" % cmap)
    for k in range(len(iou)):
        print("Detection map @ %f = %f" % (iou[k], dmap[k]*100))

    logger.log_value("Test Classification mAP", cmap, itr)
    for item in list(zip(dmap, iou)):
        logger.log_value("Test Detection mAP @ IoU = " + str(item[1]), item[0], itr)

    # utils.write_to_file(args.dataset_name, dmap, cmap, itr)