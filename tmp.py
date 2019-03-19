import numpy as np

segments = np.load('./Thumos14reduced-Annotations/segments.npy')
labels = np.load('./Thumos14reduced-Annotations/labels.npy')
classes = np.load('./Thumos14reduced-Annotations/classlist.npy')


def get_duration_for_class(class_idx):
    class_name = classes[class_idx].decode('utf-8')
    list_duration = []
    for i in range(len(segments)):
        cur_segs = segments[i]
        cur_lbls = labels[i]

        for idx, lbl in enumerate(cur_lbls):
            if lbl == class_name:
                seg = [round(k*25/16) for k in cur_segs[idx]]
                list_duration.append(
                    abs(seg[0]-seg[1])
                )
    return np.array(list_duration)


def info(l):
    print("mean : {:.2f}".format(np.mean(l)))
    print("median : {:.2f}".format(np.median(l)))
    print("std : {:.2f}".format(np.std(l)))
    print("({:.2f}, {:.2f})".format(np.min(l), np.max(l)))