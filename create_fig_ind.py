import torch
from model import Model
from dataset import Dataset
import utils
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, deque
import seaborn as sns
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import scipy.io as sio
from scipy.signal import savgol_filter
from matplotlib.backends.backend_pdf import PdfPages

torch.set_default_tensor_type('torch.cuda.FloatTensor')

IS_ORIGINAL = False

if IS_ORIGINAL:
    from model import Model as Model
    import options as options

    out_name = "./fig/fig.pdf"

else:
    # from model import Model_detect as Model
    import options_expand as options

    out_name = "./fig/fig.pdf"


def smooth(v, order=1):
    return v
    l = min(50, len(v))
    l = l - (1 - l % 2)
    if len(v) <= order:
        return v
    return savgol_filter(v, l, order)


def sigmoid(x):
    return 1/(1+np.exp(-x)+1e-10)


def test(features, model, device):

    features = torch.from_numpy(features).float().to(device)
    features = features.unsqueeze(0)
    with torch.no_grad():
        if IS_ORIGINAL:
            _, x_class = model(Variable(features), is_training=False)
        else:
            # features = features.unsqueeze(0)
            _, x_class = model(Variable(features), is_training=False)

    x_class = x_class.squeeze()

    # x_class = torch.sigmoid(x_class)
    element_logits = x_class.cpu().data.numpy()
    return element_logits  # vid_len, cls


def get_pred_loc(x, threshold=0.1):
    pred_loc = []
    vid_pred = np.concatenate(
        [np.zeros(1), (x > threshold).astype('float32'), np.zeros(1)],
        axis=0)
    vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1]
                     for idt in range(1, len(vid_pred))]
    s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
    e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
    for j in range(len(s)):
        if e[j] - s[j] >= 2:
            pred_loc.append((s[j], e[j]))
    return pred_loc


def plot_vspan(Y, ax, ymin=0, ymax=1, color=(1, 0, 0)):
    for i in range(len(Y)-1):
        xs = i
        xe = i+1
        clr = list(color) + [Y[i]]
        ax.axvspan(xmin=xs, xmax=xe+2, ymin=ymin, ymax=ymax, color=tuple(clr))
    # return ax


if __name__ == "__main__":
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    dataset = Dataset(args)

    model = Model(dataset.feature_size, dataset.num_class).to(device)

    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    # args.pretrained_ckpt = './ckpt/thumos/thumos_base.pkl'
    if args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt)
        if IS_ORIGINAL:
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise SystemExit
    model.eval()

    total_images = len(dataset.testidx)

    palette = sns.color_palette(None, len(dataset.classlist))

    cnt_ax = 0
    for feat, labs, seg, vname in tqdm(dataset.load_one_test_with_segment()):
        if len(labs) == 0:
            continue
        element_logits = test(feat, model, device)

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 3)
        # ax = axes[cnt_ax]
        cnt_ax += 1

        if cnt_ax >= 10:
            break

        for classname in np.unique(labs):
            cls_idx = utils.str2ind(classname, dataset.classlist)
            logit = element_logits[:, cls_idx]

            logit = smooth(logit)

            def softmax(x):
                x = x - np.max(x)
                return np.exp(x)/np.sum(np.exp(x))
            def softmin(x):
                x = x - np.min(x)
                return np.exp(-x)/np.sum(np.exp(-x))

            #logit = np.clip(logit, a_max=4, a_min=-4)
            #logit = softmax(logit)
            logit = (logit - np.min(logit))/(np.max(logit)-np.min(logit)+1e-10)

            if np.all(logit<0.5):
                import pdb
                pdb.set_trace()

            pred_loc = get_pred_loc(logit, threshold=0.5)
            pred = np.zeros(len(feat))

            for _loc in pred_loc:
                pred[_loc[0]:_loc[1]+1] = 1

            idx = np.where(labs == classname)[0]
            gt = np.zeros(len(feat))

            for _id in idx:
                _seg = seg[_id]
                s = int(round(_seg[0]*25/16))
                e = int(round(_seg[1]*25/16))
                gt[s:e+1] = 1

            # ax.plot(logit_orig, color=palette[cls_idx], linewidth=1, alpha=0.1)
            plot_vspan(pred, ax, ymin=0.33, ymax=0.63, color=(1., 0, 0))
            plot_vspan(gt, ax, ymin=0.68, ymax=0.98, color=(0, 1., 0))
            ax.plot(logit*0.28, color='red', linewidth=2)
            ax.hlines([0.30, 0.65], xmin=0, xmax=len(gt), linestyles='dashed', color='gray',
            linewidth=0.5)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_yticks([0.15, 0.45, 0.8])
            # ax.set_yticklabels(["score", "detection", "ground-truths"])
            
            fname = str(vname.decode("utf8"))+ " _" + str(classname)
            # ax.plot(pred, color=palette[cls_idx],
            #         linestyle='-.', linewidth=1, alpha=0.4)
            # ax.plot(logit, color=palette[cls_idx], linewidth=2)
            # ax.plot(gt, color=palette[cls_idx], linestyle='-',
            #         linewidth=2, alpha=0.4)
            break
        # ax.plot(atn, color=(0, 0, 0), alpha=0.6)
        ax.grid(False)
        fig.tight_layout()
        Path("./fig_wl/anet_cos").mkdir(parents=True, exist_ok=True)
        fig.savefig("fig_wl/anet_cos/{}_wl.pdf".format(fname), format="pdf", dpi=1200)
        plt.close('all')
        # ax.set_yticks([])
