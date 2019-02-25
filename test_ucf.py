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
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import options_ucf
from ucf_dataset import Dataset
from model import Model
from sklearn.metrics import accuracy_score


def test(dataset, args, model, device):

    instance_logits_stack = []
    element_logits_stack = []
    labels_stack = []

    for features, labels, name in dataset.load_one_test():
        print(f"{name}")
        features = torch.from_numpy(features).float().to(device)

        with torch.no_grad():
            _, element_logits = model(Variable(features), is_training=False)
        tmp = F.softmax(torch.mean(torch.topk(element_logits, k=int(np.ceil(len(features)/8)), dim=0)[0], dim=0), dim=0).cpu().data.numpy()
        element_logits = element_logits.cpu().data.numpy()

        instance_logits_stack.append(tmp)
        # element_logits_stack.append(element_logits)
        labels_stack.append(labels)

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)

    _pred = np.argmax(instance_logits_stack, axis=-1)
    _gt = np.argmax(labels_stack, axis=-1)

    score = accuracy_score(_gt, _pred)
    print(f"Accuracy : {score*100}%")

if __name__ == "__main__":
    args = options_ucf.parser.parse_args()
    dataset = Dataset(args)
    device = torch.device("cuda")

    model_path = './ckpt/ucf.pkl'
    model = Model(args.feature_size, args.num_class).to(device)
    model.load_state_dict(torch.load(model_path))
    test(dataset, args, model, device)