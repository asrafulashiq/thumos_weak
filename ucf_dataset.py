import random
import numpy as np
import os
from pathlib import Path
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import random
import utils2 as utils


class Dataset():
    def __init__(self, args):
        self.test_path = '/media/ash/New Volume/dataset/UCF_crime'+\
            '/custom_split_C3D/Custom_test_split_mini.txt'
        self.train_path = '/media/ash/New Volume/dataset/UCF_crime/'+\
            'custom_split_C3D/Custom_train_split_mini_abnormal.txt'
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.feature_size = args.feature_size
        # self.path_to_features = self.dataset_name + '-I3D-JOINTFeatures.npy'
        # self.path_to_annotations = self.dataset_name + '-Annotations/'
        self.features = [] #np.load(self.path_to_features, encoding='bytes')
        # self.segments = np.load(self.path_to_annotations + 'segments.npy')
        self.labels =  [] #np.load(self.path_to_annotations + 'labels_all.npy')     # Specific to Thumos14
        self.classlist = set()  # np.load(self.path_to_annotations + 'classlist.npy')
        self.subset = []  # np.load(self.path_to_annotations + 'subset.npy')
        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0

        self.get_data_label()

        self.labels_multihot = [utils.strlist2multihot(labs, self.classlist)
                                for labs in self.labels]

        self.train_test_idx()
        self.classwise_feature_mapping()

    def get_data_label(self):
        prog = re.compile('[^a-zA-Z]')
        with open(self.train_path, 'r') as fp:
            for line in fp:
                line = line.rstrip()
                self.features.append(line)
                name = line.split(os.path.sep)[-1]
                _category = prog.split(name)[0]
                self.classlist.add(_category)
                self.subset.append('train')
                self.labels.append([_category])

        with open(self.test_path, 'r') as fp:
            for line in fp:
                line = line.rstrip()
                self.features.append(line)
                name = line.split(os.path.sep)[-1]
                _category = prog.split(name)[0]
                self.classlist.add(_category)
                self.subset.append('test')
                self.labels.append([_category])
        self.classlist = sorted(list(self.classlist))

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s == 'train':   # Specific to Thumos14
                self.trainidx.append(i)
            else:
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category:
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def _load(self, path, normalize=False):
        x = np.load(path)
        if normalize:
            x = x/np.linalg.norm(x)
        return x

    def load_data(self, n_similar=3, is_training=True):
        if is_training:
            features = []
            labels = []
            idx = []

            # Load similar pairs
            rand_classid = np.random.choice(len(self.classwiseidx), size=n_similar)
            for rid in rand_classid:
                rand_sampleid = np.random.choice(len(self.classwiseidx[rid]), size=2)
                idx.append(self.classwiseidx[rid][rand_sampleid[0]])
                idx.append(self.classwiseidx[rid][rand_sampleid[1]])

            # Load rest pairs
            rand_sampleid = np.random.choice(len(self.trainidx), size=self.batch_size-2*n_similar)
            for r in rand_sampleid:
                idx.append(self.trainidx[r])

            data = np.array([utils.process_feat(self._load(self.features[i]), self.t_max)
                             for i in idx])
            labels = np.array([self.labels_multihot[i] for i in idx])
            return data, labels

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self._load(self.features[self.testidx[self.currenttestidx]])

            if self.currenttestidx == len(self.testidx)-1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1

            return np.array(feat), np.array(labs), done

    def load_valid(self):
        indices = np.random.choice(self.testidx, size=self.batch_size)
        data = np.array([utils.process_feat(self._load(self.features[i]),
                        self.t_max) for i in indices])
        labels = np.array([self.labels_multihot[i] for i in indices])
        return data, labels

    def load_one_test(self):
        for idx in self.testidx:
            feat = self._load(self.features[idx])
            labs = self.labels_multihot[idx]
            yield np.array(feat), np.array(labs), self.features[idx]
