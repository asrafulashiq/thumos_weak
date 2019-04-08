import torch
from model import Model
#from video_dataset import Dataset
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
    out_name = "./fig/test_figures_original.pdf"

else:
    # from model import Model_detect as Model
    import options_expand as options
    out_name = "./fig/test_detect_anet.pdf"



def smooth(v, order=3):
    # return v
    l = min(10, len(v))
    l = l - (1 - l % 2)
    if len(v) <= order:
        return v
    return savgol_filter(v, l, order)


def test(features, model, device):

    features = torch.from_numpy(features).float().to(device)
    with torch.no_grad():
        if IS_ORIGINAL:
            _, x_class, atn = model(Variable(features), is_training=False)
        else:
            # features = features.unsqueeze(0)
            _, x_class = model(Variable(features), is_training=False)

    x_class = x_class.squeeze()

    # x_class = torch.sigmoid(x_class)
    element_logits = x_class.cpu().data.numpy()
    return element_logits  # vid_len, cls


def sigmoid(x, beta=0.5):
    return 1 / (1+np.exp(-beta * x))


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
        if e[j] - s[j] >= 0:
            pred_loc.append((s[j], e[j]))
    return pred_loc


if __name__ == "__main__":
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    dataset = Dataset(args)

    model = Model(dataset.feature_size, dataset.num_class).to(device)

    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    if args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt)
        if IS_ORIGINAL:
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    plt.close('all')
    total_images = len(dataset.testidx)
    im_per_row = 4
    row_per_page = 8
    total_pages = int(np.ceil(total_images / (im_per_row*row_per_page)))

    pdf = PdfPages(out_name)

    figs = []
    axes = []
    for cnt in range(total_pages):
        _fig, _ax = plt.subplots(row_per_page, im_per_row)
        _fig.set_size_inches(10, 15)
        figs.append(_fig)
        axes.extend(_ax.flatten())
    # fig.set_size_inches(_mul * 2, _mul * 8)
    # axes = fig.axes

    palette = sns.color_palette(None, len(dataset.classlist))
    # palette = np.array(palette)

    cnt_ax = 0
    for feat, labs, seg in tqdm(dataset.load_one_test_with_segment()):
        if len(labs) == 0:
            continue
        element_logits = test(feat, model, device)

        ax = axes[cnt_ax]
        cnt_ax += 1

        for classname in np.unique(labs):
            cls_idx = utils.str2ind(classname, dataset.classlist)
            logit = element_logits[:, cls_idx]
            logit = smooth(logit)
            logit = sigmoid(logit)
            logit = (logit - np.min(logit))/(np.max(logit)-np.min(logit)+1e-10)

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

            ax.plot(gt, color=palette[cls_idx], linestyle='-',
                    linewidth=2, alpha=0.4)
            ax.plot(pred, color=palette[cls_idx], linestyle='-.', linewidth=2, alpha=0.4)
            ax.plot(logit, color=palette[cls_idx], linewidth=2)
        # ax.plot(atn, color=(0, 0, 0), alpha=0.6)
        ax.grid(True)
        ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8])
        ax.set_title(",".join([ii[:4] for ii in np.unique(labs)]))
        ax.set_ylim(-0.05, 1.1)
    for fig in figs:
        fig.tight_layout()
        pdf.savefig(fig)
    plt.close('all')
    pdf.close()