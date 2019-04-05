import numpy as np
from pathlib import Path


root = './ActivityNet1.2-Annotations'
data = './ActivityNet1.2-I3D-JOINTFeatures.npy'


classlist = np.load('./ActivityNet1.2-Annotations/classlist.npy')
duration = np.load('./ActivityNet1.2-Annotations/duration.npy')
labels = np.load('./ActivityNet1.2-Annotations/labels.npy')
labels_all = np.load('./ActivityNet1.2-Annotations/labels_all.npy')
segments = np.load('./ActivityNet1.2-Annotations/segments.npy')
subset = np.load('./ActivityNet1.2-Annotations/subset.npy')
vname = np.load('./ActivityNet1.2-Annotations/videoname.npy')

mini_classlist_utf = np.random.choice(classlist, size=20, replace=False)
mini_classlist = np.array([i.decode('utf8') for i in mini_classlist_utf])

mini_idx = []
mini_labels = []
mini_labels_all = []
mini_segments = []

for i, each_lab in enumerate(labels):
    flag = False
    idx_lab = []
    idx_seg = []
    for j, lab in enumerate(each_lab):
        if lab in mini_classlist:
            flag = True
            idx_lab.append(lab)
            idx_seg.append(segments[i][j])

    if flag:
        mini_idx.append(i)
        mini_labels.append(idx_lab)
        mini_labels_all.append(np.unique(idx_lab))
        mini_segments.append(idx_seg)

mini_duration = duration[mini_idx]
mini_subset = subset[mini_idx]
mini_vname = vname[mini_idx]

mini_labels = np.array(mini_labels)
mini_labels_all = np.array(mini_labels_all)
mini_segments = np.array(mini_segments)

data = np.load('./ActivityNet1.2-I3D-JOINTFeatures.npy', encoding="bytes")
print("data loaded")

mini_data = data[mini_idx]


np.save('./ActivityNet1.2-mini-Annotations/classlist.npy', mini_classlist_utf)
np.save('./ActivityNet1.2-mini-Annotations/duration.npy', mini_duration)
np.save('./ActivityNet1.2-mini-Annotations/labels.npy', mini_labels)
np.save('./ActivityNet1.2-mini-Annotations/labels_all.npy', mini_labels_all)
np.save('./ActivityNet1.2-mini-Annotations/segments.npy', mini_segments)
np.save('./ActivityNet1.2-mini-Annotations/subset.npy', mini_subset)
np.save('./ActivityNet1.2-mini-Annotations/videoname.npy', mini_vname)

np.save('./ActivityNet1.2-mini-I3D-JOINTFeatures.npy', mini_data)