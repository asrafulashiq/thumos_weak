# This code is originally from the official ActivityNet repo
# https://github.com/activitynet/ActivityNet
# Small modification from ActivityNet Code

import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
import sys
import scipy.io as sio

from utils_eval import get_blocked_videos
from utils_eval import interpolated_prec_rec
from utils_eval import segment_iou


def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i]][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]



def sigmoid(x, eps=1e-10):
    return 1/(1+np.exp(-x) + eps)

def softmax(x, axis=-1, wt=1):
    xm = x.max()


def smooth(v, order=2):
    return v
    # l = min(5, len(v))
    # l = l - (1 - l % 2)
    # if len(v) <= order:
    #     return v
    # return savgol_filter(v, l, order)

def filter_segments(segment_predict, videonames, ambilist):
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        vn = videonames[int(segment_predict[i, 0])]
        for a in ambilist:
            if a[0] == vn:
                gt = range(
                    int(round(float(a[2]) * 25 / 16)), int(round(float(a[3]) * 25 / 16))
                )
                pd = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(
                    len(set(gt).union(set(pd)))
                )
                if IoU > 0:
                    ind[i] = 1
    s = [
        segment_predict[i, :]
        for i in range(np.shape(segment_predict)[0])
        if ind[i] == 0
    ]
    return np.array(s)


class ANETdetection(object):

    def __init__(
        self,
        annotation_path='./Thumos14reduced-Annotations',
        tiou_thresholds=np.array([0.1, 0.3, 0.5]),
        args=None,
        subset="test",
        verbose=False
    ):
        self.subset = subset
        self.args = args
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.ap = None
        self.annotation_path = annotation_path

        self._import_ground_truth(self.annotation_path)

        #####

    def _import_ground_truth(self, annotation_path):
        gtsegments = np.load(annotation_path + "/segments.npy", allow_pickle=True)
        gtlabels = np.load(annotation_path + "/labels.npy", allow_pickle=True)
        videoname = np.load(annotation_path + "/videoname.npy", allow_pickle=True)
        videoname = np.array([i.decode("utf8") for i in videoname])
        subset = np.load(annotation_path + "/subset.npy", allow_pickle=True)
        subset = np.array([s.decode("utf-8") for s in subset])
        classlist = np.load(annotation_path + "/classlist.npy", allow_pickle=True)
        classlist = np.array([c.decode("utf-8") for c in classlist])
        duration = np.load(annotation_path + "/duration.npy", allow_pickle=True)
        ambilist = annotation_path + "/Ambiguous_test.txt"

        try:
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ") for a in ambilist]
        except:
            ambilist = []

        self.ambilist = ambilist
        self.classlist = classlist

        subset_ind = (subset == self.subset)
        gtsegments = gtsegments[subset_ind]
        gtlabels = gtlabels[subset_ind]
        videoname = videoname[subset_ind]
        duration = duration[subset_ind]

        self.idx_to_take = [i for i, s in enumerate(gtsegments)
                            if len(s) > 0 ]

        gtsegments = gtsegments[self.idx_to_take]
        gtlabels = gtlabels[self.idx_to_take]
        videoname = videoname[self.idx_to_take]
        duration = duration[self.idx_to_take].squeeze()

        self.videoname = np.array(videoname)

        self.video_info = pd.DataFrame(
            {
                "video-id": videoname,
                "duration": duration*25/16,
                "gt-labels": gtlabels,
                "gt-segments": gtsegments,
                "_id": list(range(len(videoname)))
            }
        )

        # which categories have temporal labels ?
        templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

        # # the number index for those categories.
        templabelidx = []
        for t in templabelcategories:
            templabelidx.append(str2ind(t, classlist))

        self.templabelidx = templabelidx

        video_lst, t_start_lst, t_end_lst, label_lst, _id = [], [], [], [], []

        for i in range(len(gtsegments)):
            # if self.ind_to_keep is not None and i not in self.ind_to_keep:
            #     continue
            for j in range(len(gtsegments[i])):
                _id.append(i)
                video_lst.append(str(videoname[i]))
                t_start_lst.append(round(gtsegments[i][j][0]*25/16))
                t_end_lst.append(round(gtsegments[i][j][1]*25/16))
                label_lst.append(str2ind(gtlabels[i][j], self.classlist))
        ground_truth = pd.DataFrame(
            {
                "id": _id,
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
            }
        )
        self.ground_truth = ground_truth
        self.activity_index = {i:templabelidx[i] for i in range(len(templabelidx))}

    def __getitem__(self, idx):
        rows = self.ground_truth[self.ground_truth["id"] == idx]
        return rows

    def get_topk_mean(self, x, k, axis=0):
        return np.mean(np.sort(x, axis=axis)[-int(k):, :], axis=0)

    def _get_vid_score(self, pred):
        # pred : (n, class)
        if self.args is None:
            k = 8
            topk_mean = self.get_topk_mean(pred, k)
            # ind = topk_mean > -50
            return pred, topk_mean

        win_size = int(self.args.topk)
        split_list = [i*win_size for i in range(1, int(pred.shape[0]//win_size))]
        splits = np.split(pred, split_list, axis=0)

        tops = []
        for each_split in splits:
            top_mean = self.get_topk_mean(each_split, self.args.topk2)
            tops.append(top_mean)
        tops = np.array(tops)
        c_s = np.max(tops, axis=0)
        # prob = 1 - np.prod(1-tops, axis=0)
        # inv_prob = np.log(prob/(1-prob))
        return pred, c_s


    def _import_prediction_bmn(self, segment_predict):
        # pred = []
        # _len = []
        # for i, p in enumerate(predictions):
        #     if i in self.idx_to_take:
        #         pred.append(p)
        #         _len.append(len_stack[i])
        # predictions = pred
        # len_stack = _len

        # args = self.args
        # segment_predict = []

        # for c in self.templabelidx:
        #     for i in range(len(predictions)):
        #         tmp = predictions[i][c, ...]

        #         if tmp.shape[0] < len_stack[i]:
        #             mul = len_stack[i] / tmp.shape[0]
        #         else:
        #             mul = 1.

        #         if args is None:
        #             thres = 0.5
        #         else:
        #             thres = args.thres

        #         tmp_max = np.max(tmp, -1)
        #         end_max = np.argmax(tmp, -1)

        #         ind_start = np.where(tmp_max > thres)[0]
        #         ind_end = end_max[ind_start]
        #         conf_filtered = tmp_max[ind_start]

        #         if len(conf_filtered) > 0:
        #             _strt = []
        #             _end = []
        #             _conf = []
        #             for each in np.unique(ind_end):
        #                 tmp_end = ind_end[ind_end==each]
        #                 tmp_strt = ind_start[ind_end==each]
        #                 tmp_conf = conf_filtered[ind_end==each]
        #                 _ii = np.argmax(tmp_conf)
        #                 _strt.append(tmp_strt[_ii])
        #                 _end.append(tmp_end[_ii])
        #                 _conf.append(tmp_conf[_ii])
        #             ind_start = _strt
        #             ind_end = _end
        #             conf_filtered = _conf

        #         for kk in range(len(conf_filtered)):
        #             s = ind_start[kk] * mul
        #             e = ind_end[kk] * mul
        #             # if e - s >= 2:
        #             segment_predict.append(
        #                 [i, round(s), round(e), conf_filtered[kk], int(c)]
        #             )

        segment_predict = np.array(segment_predict)
        # segment_predict = filter_segments(segment_predict, self.videoname, self.ambilist)

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst, dur_lst = [], [], []

        # for i in range(np.shape(segment_predict)[0]):
        #     video_lst.append(self.videoname[int(segment_predict[i, 0])])
        #     t_start_lst.append(segment_predict[i, 1])
        #     t_end_lst.append(segment_predict[i, 2])
        #     score_lst.append(segment_predict[i, 3])
        #     label_lst.append(segment_predict[i, 4])
        #     dur_lst.append(segment_predict[i, 5])
        
        # prediction = pd.DataFrame(
        #     {
        #         "video-id": video_lst,
        #         "t-start": t_start_lst,
        #         "t-end": t_end_lst,
        #         "label": label_lst,
        #         "score": score_lst,
        #         "duration": dur_lst
        #     }
        # )
        prediction = pd.DataFrame(
            {
                "video-id": self.videoname[segment_predict[:, 0].astype(int)],
                "t-start": segment_predict[:, 1],
                "t-end": segment_predict[:, 2],
                "score": segment_predict[:, 3],
                "label": segment_predict[:, 4],
                "duration": segment_predict[:, 5]
            }
        )
        self.prediction = prediction

    def _import_prediction(self, predictions):
        pred = []
        for i, p in enumerate(predictions):
            if i in self.idx_to_take:
                pred.append(p)
        predictions = pred

        # process the predictions such that classes having greater than a certain threshold are detected only
        # predictions_mod = []
        # c_score = []
        # for p in predictions:
        #     # pp = -p
        #     # [pp[:, i].sort() for i in range(np.shape(pp)[1])]
        #     # pp = -pp
        #     # c_s = np.mean(pp[: int(np.ceil(np.shape(pp)[0] / 8)), :], axis=0)
        #     # ind = c_s > -50
        #     # pind = p * ind
        #     pind, c_s = self._get_vid_score(p)
        #     c_score.append(c_s)
        #     predictions_mod.append(pind)

        # predictions = predictions_mod
        args = self.args
        segment_predict = []

        for c in self.templabelidx:
            for i in range(len(predictions)):
                
                tmp = smooth(predictions[i][:, c])
                if args is None:
                    thres = 0.5
                else:
                    thres = 1 - args.thres
                # tmp = np.clip(tmp, a_min=-5, a_max=5)
                threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp)) * thres
                vid_pred = np.concatenate(
                    [np.zeros(1), (tmp > threshold).astype("float32"), np.zeros(1)], axis=0
                )
                vid_pred_diff = [
                    vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
                ]
                s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
                e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]

                duration = self.video_info[self.video_info["_id"]==i]["duration"].values[0]
                cur_len = len(tmp)
                for j in range(len(s)):
                    if e[j] - s[j] >= 2:
                        segment_predict.append(
                            [i, s[j]/cur_len*duration, e[j]/cur_len*duration, np.max(tmp[s[j] : e[j]]),
                            c]
                        )
        segment_predict = np.array(segment_predict)
        # segment_predict = filter_segments(segment_predict, self.videoname, self.ambilist)

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []

        for i in range(np.shape(segment_predict)[0]):
            video_lst.append(self.videoname[int(segment_predict[i, 0])])
            t_start_lst.append(segment_predict[i, 1])
            t_end_lst.append(segment_predict[i, 2])
            score_lst.append(segment_predict[i, 3])
            label_lst.append(segment_predict[i, 4])

        prediction = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
                "score": score_lst,
            }
        )
        self.prediction = prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            print("Warning: No predictions of label '%s' were provdied." % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """

        activity_index = self.ground_truth.label.unique()
        ap = np.zeros((len(self.tiou_thresholds), len(activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby("label")
        prediction_by_label = self.prediction.groupby("label")

        results = Parallel(n_jobs=3)(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(
                    drop=True
                ),
                prediction=self._get_predictions_with_label(
                    prediction_by_label, cidx, cidx
                ),
                tiou_thresholds=self.tiou_thresholds,
            )
            for cidx in activity_index
        )

        # results = []
        # for cidx in activity_index:
        #     res = compute_average_precision_detection(
        #         ground_truth=ground_truth_by_label.get_group(cidx).reset_index(
        #             drop=True
        #         ),
        #         prediction=self._get_predictions_with_label(
        #             prediction_by_label, cidx, cidx
        #         ),
        #         tiou_thresholds=self.tiou_thresholds,
        #     )
        #     results.append(res)

        for i, cidx in enumerate(activity_index):
            ap[:, i] = results[i]

        return ap

    def evaluate(self, ind_to_keep=None):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        if ind_to_keep is not None and len(ind_to_keep) > 0:
            self.ground_truth = self.ground_truth[self.ground_truth['id'].isin(ind_to_keep)]
        if self.verbose:
            print("[INIT] Loaded annotations from {} subset.".format(self.subset))
            nr_gt = len(self.ground_truth)
            print("\tNumber of ground truth instances: {}".format(nr_gt))
            nr_pred = len(self.prediction)
            print("\tNumber of predictions: {}".format(nr_pred))
            print("\tFixed threshold for tiou score: {}".format(self.tiou_thresholds))

        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            # print ('[RESULTS] Performance on ActivityNet detection task.')
            for k in range(len(self.tiou_thresholds)):
                print("Detection map @ %f = %f" % (self.tiou_thresholds[k], self.mAP[k]))
            print("Average-mAP: {}\n".format(self.mAP))
        return self.mAP


    def save_info(self, fname):
        import pickle
        Dat = {
            "prediction": self.prediction,
            "gt": self.ground_truth
        }
        with open(fname, 'wb') as fp:
            pickle.dump(Dat, fp)



def compute_average_precision_detection(
    ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)
    ):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(
            this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values
        )
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(
            precision_cumsum[tidx, :], recall_cumsum[tidx, :]
        )

    return ap
