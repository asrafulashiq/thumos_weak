import numpy as np
import utils
import random

np.random.seed(0)


class Dataset:
    def __init__(self, args, mode='both'):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.feature_size = args.feature_size
        self.path_to_features = self.dataset_name + "-I3D-JOINTFeatures.npy"
        self.path_to_annotations = self.dataset_name + "-Annotations/"
        self.features = np.load(self.path_to_features, encoding="bytes")
        self.segments = np.load(self.path_to_annotations + "segments.npy")
        self.labels = np.load(self.path_to_annotations + "labels_all.npy")
        # Specific to Thumos14

        self._labels = np.load(self.path_to_annotations + "labels.npy")
        self.classlist = np.load(self.path_to_annotations + "classlist.npy")
        self.subset = np.load(self.path_to_annotations + "subset.npy")
        self.videonames = np.load(self.path_to_annotations + "videoname.npy")
        self.batch_size = args.batch_size
        # self.batch_size = args.num_similar * args.similar_size
        self.t_max = args.max_seqlen
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]

        ambilist = self.path_to_annotations + "/Ambiguous_test.txt"
        ambilist = list(open(ambilist, "r"))
        ambilist = [a.strip("\n").split(" ")[0] for a in ambilist]

        self.num_gt = 5
        self.gt_loc_ind = np.zeros(
            (len(self.classlist), self.num_gt,
             self.feature_size), dtype=np.float32
        )
        self.train_test_idx()
        self.classwise_feature_mapping()

        self.get_gt_for_sup()

        self.normalize = False
        self.mode = mode
        if mode == 'rgb' or mode == 'flow':
            self.feature_size = 1024

    def get_gt_for_sup(self):
        for category in self.classlist:
            cnt = 0
            label = category.decode("utf-8")
            cls_pos = utils.str2ind(label, self.classlist)
            for idx in self.testidx:
                if label in self._labels[idx]:
                    lab_indices = [
                        i for i, _l in enumerate(self._labels[idx])
                        if _l == label
                    ]
                    ind = random.choice(lab_indices)
                    s, e = self.segments[idx][ind]
                    s, e = round(s * 25 / 16), round(e * 25 / 16)

                    tmp = self.features[idx][s: e + 1]
                    if tmp.size == 0:
                        continue

                    self.gt_loc_ind[cls_pos][cnt] = np.mean(tmp, 0)
                    if np.isnan(self.gt_loc_ind[cls_pos]).any():
                        import pdb
                        pdb.set_trace()
                    cnt += 1
                    if cnt >= self.num_gt:
                        break
            if cnt < self.num_gt:
                import pdb
                pdb.set_trace()
            self.gt_loc_ind[cls_pos] = self.gt_loc_ind[cls_pos] / cnt
        self.feat_loc = np.mean(self.gt_loc_ind, axis=1)

    def load_partial(self, is_random=False):
        if is_random:
            ind = np.random.choice(
                range(self.num_gt), size=len(self.classlist), replace=True
            )
            feat = self.gt_loc_ind[
                list(range(len(self.classlist))),
                ind
            ]
            return feat
        else:
            return self.feat_loc

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode("utf-8") == "validation":  # Specific to Thumos14
                self.trainidx.append(i)
            elif s.decode("utf-8") == "test":
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode("utf-8"):
                        idx.append(i)
                        break

            self.classwiseidx.append(idx)

    def load_data(self, n_similar=0, is_training=True, similar_size=2):
        if is_training:
            labels = []
            idx = []

            # Load similar pairs
            if n_similar != 0:
                rand_classid = np.random.choice(
                    len(self.classwiseidx), size=n_similar)
                for rid in rand_classid:
                    rand_sampleid = np.random.choice(
                        len(self.classwiseidx[rid]),
                        size=similar_size, replace=False)

                    for k in rand_sampleid:
                        idx.append(self.classwiseidx[rid][k])

            # Load rest pairs
            if self.batch_size-similar_size*n_similar < 0:
                self.batch_size = similar_size*n_similar

            rand_sampleid = np.random.choice(
                len(self.trainidx),
                size=self.batch_size-similar_size*n_similar)

            for r in rand_sampleid:
                idx.append(self.trainidx[r])

            feat = np.array(
                [utils.process_feat(
                    self.features[i], self.t_max, self.normalize) for i in idx]
            )
            labels = np.array([self.labels_multihot[i] for i in idx])

            if self.mode == 'rgb':
                feat = feat[..., :self.feature_size]
            elif self.mode == 'flow':
                feat = feat[..., self.feature_size:]
            return feat, labels

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            feat = utils.process_feat(feat, normalize=self.normalize)

            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1

            feat = np.array(feat)
            if self.mode == 'rgb':
                feat = feat[..., :self.feature_size]
            elif self.mode == 'flow':
                feat = feat[..., self.feature_size:]
            return feat, np.array(labs), done

    def load_valid(self):
        indices = np.random.choice(self.testidx, size=self.batch_size)
        data = np.array(
            [utils.process_feat(self.features[i], self.t_max,
                                self.normalize) for i in indices]
        )
        labels = np.array([self.labels_multihot[i] for i in indices])
        return data, labels

    def load_one_test_with_segment(self):
        for idx in self.testidx:
            feat = self.features[idx]
            labs = self._labels[idx]
            seg = self.segments[idx]
            vname = self.videonames[idx]
            yield np.array(feat), np.array(labs), np.array(seg), vname
