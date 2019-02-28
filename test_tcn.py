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
from detectionMAP import getDetectionMAP as dmAP
import scipy.io as sio
from sklearn.metrics import accuracy_score
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def test(itr, dataset, args, model, logger, device, is_detect=True, is_score=True):

    done = False
    instance_logits_stack = []
    element_logits_stack = []
    labels_stack = []
    model.eval()
    while not done:
        if dataset.currenttestidx % 100 == 0:
            print('Testing test data point %d of %d' % (dataset.currenttestidx,
                                                        len(dataset.testidx)))

        features, labels, done = dataset.load_data(is_training=False)
        features = torch.from_numpy(features).float().to(device)

        with torch.no_grad():
            features = features.unsqueeze(0)
            element_logits = model(Variable(features),
                                   is_training=False)
            element_logits = element_logits.squeeze(0)

        tmp = F.softmax(element_logits, dim=-1).cpu().data.numpy()
        #tmp = multit.cpu().data.numpy()
        # tmp = F.softmax(torch.mean(torch.topk(element_logits, k=int(np.ceil(len(features)/8)), dim=0)[0], dim=0), dim=0).cpu().data.numpy()
        # element_logits = element_logits.cpu().data.numpy()

        instance_logits_stack.append(tmp)
        # element_logits_stack.append(element_logits)
        labels_stack.append(labels)

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)

    cmap = cmAP(instance_logits_stack, labels_stack)
    print('Classification map %f' % cmap)
    logger.log_value('Test Classification mAP', cmap, itr)

    if is_score:
        from sklearn.metrics import accuracy_score
        _gt = np.argmax(labels_stack, axis=-1)
        _pred = np.argmax(instance_logits_stack, axis=-1)
        accuracy = accuracy_score(_gt, _pred)
        print("Accuracy : {:.3f}".format(accuracy*100))
        logger.log_value('Test Accuracy', accuracy, itr)

    if not is_detect:
        utils.write_to_file(args.dataset_name, None, cmap, itr)
        return

    dmap, iou = dmAP(element_logits_stack, dataset.path_to_annotations)

    if args.dataset_name == 'Thumos14':
        test_set = sio.loadmat('test_set_meta.mat')['test_videos'][0]
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]['background_video'] == 'YES':
                labels_stack[i, :] = np.zeros_like(labels_stack[i,:])

    print('Detection map @ %f = %f' % (iou[0], dmap[0]))
    print('Detection map @ %f = %f' % (iou[1], dmap[1]))
    print('Detection map @ %f = %f' % (iou[2], dmap[2]))
    print('Detection map @ %f = %f' % (iou[3], dmap[3]))
    print('Detection map @ %f = %f' % (iou[4], dmap[4]))
    for item in list(zip(dmap, iou)):
        logger.log_value('Test Detection mAP @ IoU = ' + str(item[1]), item[0], itr)

    utils.write_to_file(args.dataset_name, dmap, cmap, itr)
