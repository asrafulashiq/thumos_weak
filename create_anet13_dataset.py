import utils
import os
import json
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from collections import defaultdict
from scipy.interpolate import interp1d
from skimage import measure
from skimage.morphology import dilation
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import medfilt
import random

import pdb


new_cls_names = {
    "ActivityNet12": np.load("misc/new_cls_names_anet12.npy").item(),
    "ActivityNet13": np.load("misc/new_cls_names_anet13.npy").item(),
}

new_cls_indices = {
    "ActivityNet12": np.load("misc/new_cls_indices_anet12.npy").item(),
    "ActivityNet13": np.load("misc/new_cls_indices_anet13.npy").item(),
}


def load_config_file(config_file):
    """
    -- Doc for parameters in the json file --

    feature_oversample:   Whether data augmentation is used (five crop and filp).
    sample_rate:          How many frames between adjacent feature snippet.

    with_bg:              Whether hard negative mining is used.
    diversity_reg:        Whether diversity loss and norm regularization are used.
    diversity_weight:     The weight of both diversity loss and norm regularization.

    train_run_num:        How many times the experiment is repeated.
    training_max_len:     Crop the feature sequence when training if it exceeds this length.

    learning_rate_decay:  Whether to reduce the learning rate at half of training steps.
    max_step_num:         Number of training steps.
    check_points:         Check points to test and save models.
    log_freq:             How many training steps the log is added to tensorboard.

    model_params:
    cls_branch_num:       Branch number in the multibranch network.
    base_layer_params:    Filter number and size in each layer of the embedding module.
    cls_layer_params:     Filter number and size in each layer of the classification module.
    att_layer_params:     Filter number and size in each layer of the attention module.

    detect_params:        Parameters for action localization on the CAS. 
                          See detect.py for details.

    base_sample_rate:     'sample_rate' when feature extraction.
    base_snippet_size:    The size of each feature snippet.

    bg_mask_dir:          The folder of masks of static clips.

    < Others are easy to guess >

    """

    all_params = json.load(open(config_file))

    dataset_name = all_params["dataset_name"]
    feature_type = all_params["feature_type"]

    all_params["file_paths"] = all_params["file_paths"][dataset_name]
    all_params["action_class_num"] = all_params["action_class_num"][
        dataset_name
    ]
    all_params["base_sample_rate"] = all_params["base_sample_rate"][
        dataset_name
    ][feature_type]
    all_params["base_snippet_size"] = all_params["base_snippet_size"][
        feature_type
    ]

    assert all_params["sample_rate"] % all_params["base_sample_rate"] == 0

    all_params["model_class_num"] = all_params["action_class_num"]
    if all_params["with_bg"]:
        all_params["model_class_num"] += 1

    all_params["model_params"]["class_num"] = all_params["model_class_num"]

    # Convert second to frames
    all_params["detect_params"]["proc_value"] = int(
        all_params["detect_params"]["proc_value"] * all_params["sample_rate"]
    )

    print(all_params)
    return all_params


def load_features(
    dataset_dict,  # dataset_dict will be modified
    key,
    feature_type,
    sample_rate,
    base_sample_rate,
    temporal_aug,
    rgb_feature_dir,
    flow_feature_dir,
    dataset_name="ActivityNet13",
):

    assert feature_type in ["i3d", "untri"]

    assert sample_rate % base_sample_rate == 0
    f_sample_rate = int(sample_rate / base_sample_rate)

    # sample_rate of feature sequences, not original video

    ###############
    def __process_feature_file(filename):
        """ Load features from a single file. """

        feature_data = np.load(filename)

        frame_cnt = feature_data["frame_cnt"].item()

        if feature_type == "untri":
            feature = np.swapaxes(feature_data["feature"][:, :, :, 0, 0], 0, 1)
        elif feature_type == "i3d":
            feature = feature_data["feature"]

        # Feature: (B, T, F)
        # Example: (1, 249, 1024) or (10, 249, 1024) (Oversample)

        if temporal_aug:  # Data augmentation with temporal offsets
            feature = [
                feature[:, offset::f_sample_rate, :]
                for offset in range(f_sample_rate)
            ]
            # Cut to same length, OK when training
            min_len = int(min([i.shape[1] for i in feature]))
            feature = [i[:, :min_len, :] for i in feature]

            assert len(set([i.shape[1] for i in feature])) == 1
            feature = np.concatenate(feature, axis=0)

        else:
            feature = feature[:, ::f_sample_rate, :]

        return feature, frame_cnt

        # Feature: (B x f_sample_rate, T, F)

    ###############

    # Load all features

    k = key
    print("Loading: {}".format(k))

    # Init empty
    dataset_dict[k]["frame_cnt"] = -1
    dataset_dict[k]["rgb_feature"] = -1
    dataset_dict[k]["flow_feature"] = -1

    if rgb_feature_dir:

        if dataset_name == "thumos14":
            rgb_feature_file = os.path.join(rgb_feature_dir, k + "-rgb.npz")
        else:
            rgb_feature_file = os.path.join(
                rgb_feature_dir, "v_" + k + "-rgb.npz"
            )

        rgb_feature, rgb_frame_cnt = __process_feature_file(rgb_feature_file)

        dataset_dict[k]["frame_cnt"] = rgb_frame_cnt
        dataset_dict[k]["rgb_feature"] = rgb_feature

    if flow_feature_dir:

        if dataset_name == "thumos14":
            flow_feature_file = os.path.join(flow_feature_dir, k + "-flow.npz")
        else:
            flow_feature_file = os.path.join(
                flow_feature_dir, "v_" + k + "-flow.npz"
            )

        flow_feature, flow_frame_cnt = __process_feature_file(
            flow_feature_file
        )

        dataset_dict[k]["frame_cnt"] = flow_frame_cnt
        dataset_dict[k]["flow_feature"] = flow_feature

    if rgb_feature_dir and flow_feature_dir:
        assert rgb_frame_cnt == flow_frame_cnt
        assert (
            dataset_dict[k]["rgb_feature"].shape[1]
            == dataset_dict[k]["flow_feature"].shape[1]
        )
        assert (
            dataset_dict[k]["rgb_feature"].mean()
            != dataset_dict[k]["flow_feature"].mean()
        )

    return rgb_feature, flow_feature


def __load_background(
    dataset_dict,  # dataset_dict will be modified
    dataset_name,
    bg_mask_dir,
    sample_rate,
    action_class_num,
):

    bg_mask_files = os.listdir(bg_mask_dir)
    bg_mask_files.sort()

    for bg_mask_file in bg_mask_files:

        if dataset_name == "thumos14":
            video_name = bg_mask_file[:-4]
        else:
            video_name = bg_mask_file[2:-4]

        new_key = video_name + "_bg"

        if video_name not in dataset_dict.keys():
            continue

        bg_mask = np.load(os.path.join(bg_mask_dir, bg_mask_file))
        bg_mask = bg_mask["mask"]

        assert dataset_dict[video_name]["frame_cnt"] == bg_mask.shape[0]

        # Remove if static clips are too long or too short
        bg_ratio = bg_mask.sum() / bg_mask.shape[0]
        if bg_ratio < 0.05 or bg_ratio > 0.30:
            print("Bad bg {}: {}".format(bg_ratio, video_name))
            continue

        bg_mask = bg_mask[::sample_rate]  # sample rate of original videos

        dataset_dict[new_key] = {}

        if type(dataset_dict[video_name]["rgb_feature"]) != int:

            rgb = np.array(dataset_dict[video_name]["rgb_feature"])
            bg_mask = bg_mask[: rgb.shape[1]]  # same length
            bg_rgb = rgb[:, bg_mask.astype(bool), :]
            dataset_dict[new_key]["rgb_feature"] = bg_rgb

            frame_cnt = bg_rgb.shape[
                1
            ]  # Psuedo frame count of a virtual bg video

        if type(dataset_dict[video_name]["flow_feature"]) != int:

            flow = np.array(dataset_dict[video_name]["flow_feature"])
            bg_mask = bg_mask[: flow.shape[1]]
            bg_flow = flow[:, bg_mask.astype(bool), :]
            dataset_dict[new_key]["flow_feature"] = bg_flow

            frame_cnt = bg_flow.shape[
                1
            ]  # Psuedo frame count of a virtual bg video

        dataset_dict[new_key]["annotations"] = {action_class_num: []}
        dataset_dict[new_key]["labels"] = [
            action_class_num
        ]  # background class

        fps = dataset_dict[video_name]["frame_rate"]
        dataset_dict[new_key]["frame_rate"] = fps
        dataset_dict[new_key]["frame_cnt"] = frame_cnt  # Psuedo
        dataset_dict[new_key]["duration"] = frame_cnt / fps  # Psuedo

    return dataset_dict


def get_dataset(
    subset,
    file_paths,
    sample_rate,
    base_sample_rate,
    action_class_num,
    modality="both",
    feature_type="i3d",
    feature_oversample=True,
    temporal_aug=False,
    load_background=False,
):
    dataset_name = "ActivityNet13"
    assert subset in ["train", "val", "train_and_val", "test"]
    dataset_dict = __get_anet_meta(
        file_paths[subset]["anno_json_file"], dataset_name, subset
    )

    _temp_f_type = (
        feature_type + "-oversample"
        if feature_oversample
        else feature_type + "-resize"
    )

    if modality == "both":
        rgb_dir = file_paths[subset]["feature_dir"][_temp_f_type]["rgb"]
        flow_dir = file_paths[subset]["feature_dir"][_temp_f_type]["flow"]
    elif modality == "rgb":
        rgb_dir = file_paths[subset]["feature_dir"][_temp_f_type]["rgb"]
        flow_dir = None
    elif modality == "flow":
        rgb_dir = None
        flow_dir = file_paths[subset]["feature_dir"][_temp_f_type]["flow"]
    else:
        rgb_dir = None
        flow_dir = None

    return dataset_dict, rgb_dir, flow_dir

    # dataset_dict = __load_features(dataset_dict, dataset_name, feature_type,
    #                                sample_rate, base_sample_rate, temporal_aug,
    #                                rgb_dir, flow_dir)

    # if load_background:
    #     dataset_dict = __load_background(dataset_dict, dataset_name,
    #                                      file_paths[subset]['bg_mask_dir'],
    #                                      sample_rate, action_class_num)

    # return dataset_dict


def __get_anet_meta(anno_json_file, dataset_name, subset):

    data = json.load(open(anno_json_file, "r"))
    taxonomy_data = data["taxonomy"]
    database_data = data["database"]
    missing_videos = np.load("misc/anet_missing_videos.npy")

    if subset == "train":
        subset_data = {
            k: v for k, v in database_data.items() if v["subset"] == "training"
        }
    elif subset == "val":
        subset_data = {
            k: v
            for k, v in database_data.items()
            if v["subset"] == "validation"
        }
    elif subset == "train_and_val":
        subset_data = {
            k: v
            for k, v in database_data.items()
            if v["subset"] in ["training", "validation"]
        }
    elif subset == "test":
        subset_data = {
            k: v for k, v in database_data.items() if v["subset"] == "testing"
        }

    dataset_dict = {}

    for video_name, v in subset_data.items():

        if video_name in missing_videos:
            print("Missing video: {}".format(video_name))
            continue

        dataset_dict[video_name] = {
            "duration": v["duration"],
            "frame_rate": 25,  # ActivityNet should be formatted to 25 fps first
            "labels": [],
            "annotations": {},
        }

        for entry in v["annotations"]:

            action_label = entry["label"]
            action_label = new_cls_indices[dataset_name][action_label]

            if action_label not in dataset_dict[video_name]["labels"]:
                dataset_dict[video_name]["labels"].append(action_label)
                dataset_dict[video_name]["annotations"][action_label] = []

            dataset_dict[video_name]["annotations"][action_label].append(
                entry["segment"]
            )

    return dataset_dict


def _random_select(rgb=-1, flow=-1):
    """ Randomly select one augmented feature sequence. """

    if type(rgb) != int and type(flow) != int:

        assert rgb.shape[0] == flow.shape[0]
        random_idx = random.randint(0, rgb.shape[0] - 1)
        rgb = np.array(rgb[random_idx, :, :])
        flow = np.array(flow[random_idx, :, :])

    elif type(rgb) != int:
        random_idx = random.randint(0, rgb.shape[0] - 1)
        rgb = np.array(rgb[random_idx, :, :])

    elif type(flow) != int:
        random_idx = random.randint(0, flow.shape[0] - 1)
        flow = np.array(flow[random_idx, :, :])
    else:
        pass

    return rgb, flow


def _check_length(rgb, flow, max_len):

    if type(rgb) != int and type(flow) != int:

        assert rgb.shape[1] == flow.shape[1]
        if rgb.shape[1] > max_len:
            print("Crop Both!")
            start = random.randint(0, rgb.shape[1] - max_len)
            rgb = np.array(rgb[:, start : start + max_len, :])
            flow = np.array(flow[:, start : start + max_len, :])

    elif type(rgb) != int:

        if rgb.shape[1] > max_len:
            print("Crop RGB!")
            start = random.randint(0, rgb.shape[1] - max_len)
            rgb = np.array(rgb[:, start : start + max_len, :])

    elif type(flow) != int:

        if flow.shape[1] > max_len:
            print("Crop FLOW!")
            start = random.randint(0, flow.shape[1] - max_len)
            flow = np.array(flow[:, start : start + max_len, :])
    else:
        pass

    return rgb, flow


class Dataset:
    def __init__(self, args, mode="both"):

        self.args = args
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.feature_size = args.feature_size

        params = load_config_file("./configs/anet13-local-I3D.json")

        self.params = params
        self.dataset_dict, self.rgb_path, self.flow_path = get_dataset(
            "val",
            params["file_paths"],
            params["sample_rate"],
            params["base_sample_rate"],
            params["action_class_num"],
            modality="both",
            feature_type="i3d",
            feature_oversample=True,
            temporal_aug=False,
            load_background=False,
        )

        self.single_label = False
        self.random_select = True
        self.max_len = args.max_seqlen

        self.video_list = list(self.dataset_dict.keys())
        self.video_list.sort()

    def save_dat(self):
        from pathlib import Path

        DIR = Path("./ActivityNet1.3-Annotations")
        params = load_config_file("./configs/anet13-local-I3D.json")
        dataset_dict, rgb_path_train, flow_path_train = get_dataset(
            "train",
            params["file_paths"],
            params["sample_rate"],
            params["base_sample_rate"],
            params["action_class_num"],
            modality="both",
            feature_type="i3d",
            feature_oversample=True,
            temporal_aug=False,
            load_background=False,
        )

        dataset_dict_val, rgb_path_val, flow_path_val = get_dataset(
            "val",
            params["file_paths"],
            params["sample_rate"],
            params["base_sample_rate"],
            params["action_class_num"],
            modality="both",
            feature_type="i3d",
            feature_oversample=True,
            temporal_aug=False,
            load_background=False,
        )

        dataset_dict.update(dataset_dict_val)

        videonames = []
        cls_list = new_cls_indices["ActivityNet13"]
        fn_cls = dict(zip(list(cls_list.values()), list(cls_list.keys())))
        subset = []
        duration = []
        resolution = []
        segments = []
        labels = []
        labels_unique = []
        features = []

        for cnt, k in enumerate(dataset_dict):
            if k in dataset_dict_val:
                subset.append(b"validation")
                paths = [rgb_path_val, flow_path_val]
            else:
                subset.append(b"training")
                paths = [rgb_path_train, flow_path_train]

            v = dataset_dict[k]

            _lab = []
            _segm = []
            ann = v["annotations"]

            for _k, val in ann.items():
                _lab.extend([fn_cls[_k]] * len(val))
                _segm.extend(val)

            rgb_feature_file = os.path.join(paths[0], "v_" + k + "-rgb.npz")
            flow_feature_file = os.path.join(paths[1], "v_" + k + "-flow.npz")

            if (not os.path.exists(rgb_feature_file)) or (not os.path.exists(flow_feature_file)):
                print(f"{k} does not exist")
                continue

            features.append((rgb_feature_file, flow_feature_file))
            duration.append(v["duration"])
            labels.append(_lab)
            labels_unique.append(np.unique(_lab))
            segments.append(_segm)
            videonames.append(k)

        np.save(str(DIR / "videoname.npy"), 
            np.array([i.encode('utf8') for i in videonames]))
        np.save(
            str(DIR / "classlist.npy"),
            np.array([i.encode("utf8") for i in list(cls_list.keys())]),
        )
        np.save(str(DIR / "segments.npy"), np.array(segments))
        np.save(str(DIR / "duration.npy"), np.array(duration))
        np.save(str(DIR / "labels.npy"), np.array(labels))
        np.save(str(DIR / "labels_all.npy"), np.array(labels_unique))
        np.save("ActivityNet1.3-I3D-path.npy", features)

    def save_dat_mini(self):
        from pathlib import Path

        DIR = Path("./ActivityNet1.3-mini-Annotations")
        params = load_config_file("./configs/anet13-local-I3D.json")
        dataset_dict, rgb_path_train, flow_path_train = get_dataset(
            "train",
            params["file_paths"],
            params["sample_rate"],
            params["base_sample_rate"],
            params["action_class_num"],
            modality="both",
            feature_type="i3d",
            feature_oversample=True,
            temporal_aug=False,
            load_background=False,
        )

        dataset_dict_val, rgb_path_val, flow_path_val = get_dataset(
            "val",
            params["file_paths"],
            params["sample_rate"],
            params["base_sample_rate"],
            params["action_class_num"],
            modality="both",
            feature_type="i3d",
            feature_oversample=True,
            temporal_aug=False,
            load_background=False,
        )

        dataset_dict.update(dataset_dict_val)

        videonames = []

        cls_list = new_cls_indices["ActivityNet13"]


        all_classes = list(cls_list.keys())

        selected_classes = np.random.choice(all_classes, size=20, replace=False)        



        fn_cls = dict(zip(list(cls_list.values()), list(cls_list.keys())))
        subset = []
        duration = []
        resolution = []
        segments = []
        labels = []
        labels_unique = []
        features = []

        for cnt, k in enumerate(dataset_dict):
            if k in dataset_dict_val:
                is_val = True
                paths = [rgb_path_val, flow_path_val]
            else:
                is_val = False
                paths = [rgb_path_train, flow_path_train]

            v = dataset_dict[k]

            _lab = []
            _segm = []
            ann = v["annotations"]

            for _k, val in ann.items():
                if fn_cls[_k] in selected_classes:
                    _lab.extend([fn_cls[_k]] * len(val))
                    _segm.extend(val)
            if not _lab:
                continue
            rgb_feature_file = os.path.join(paths[0], "v_" + k + "-rgb.npz")
            flow_feature_file = os.path.join(paths[1], "v_" + k + "-flow.npz")

            if (not os.path.exists(rgb_feature_file)) or (not os.path.exists(flow_feature_file)):
                print(f"{k} does not exist")
                continue

            if is_val:
                subset.append(b"validation")
            else:
                subset.append(b"training")
            features.append((rgb_feature_file, flow_feature_file))
            duration.append(v["duration"])
            labels.append(_lab)
            labels_unique.append(np.unique(_lab))
            segments.append(_segm)
            videonames.append(k)

        np.save(str(DIR / "videoname.npy"), 
            np.array([i.encode('utf8') for i in videonames]))
        np.save(
            str(DIR / "classlist.npy"),
            np.array([i.encode("utf8") for i in list(selected_classes)]),
        )
        np.save(str(DIR / "segments.npy"), np.array(segments))
        np.save(str(DIR / "subset.npy"), np.array(subset))
        np.save(str(DIR / "duration.npy"), np.array(duration))
        np.save(str(DIR / "labels.npy"), np.array(labels))
        np.save(str(DIR / "labels_all.npy"), np.array(labels_unique))
        np.save("ActivityNet1.3-mini-I3D-path.npy", features)


    def __len__(self):
        return len(self.video_list)

    def __load_feat(self, key):
        params = self.params
        ret = load_features(
            self.dataset_dict,
            key,
            "i3d",
            params["sample_rate"],
            params["base_sample_rate"],
            temporal_aug=False,
            rgb_feature_dir=self.rgb_path,
            flow_feature_dir=self.flow_path,
        )
        return ret

    def __getitem__(self, idx):
        video = self.video_list[idx]

        # rgb, flow = self.__load_feat(video)

        try:
            rgb, flow = self.__load_feat(video)
        except FileNotFoundError:
            return None

        if self.max_len:
            rgb, flow = _check_length(rgb, flow, self.max_len)

        if self.random_select:
            rgb, flow = _random_select(rgb, flow)

        return_dict = {
            "video_name": video,
            "rgb": rgb,
            "flow": flow,
            "frame_rate": self.dataset_dict[video][
                "frame_rate"
            ],  # frame_rate == fps
            "frame_cnt": self.dataset_dict[video]["frame_cnt"],
            "anno": self.dataset_dict[video]["annotations"],
        }

        if self.single_label:
            return_dict["label"] = self.dataset_dict[video]["label_single"]
            return_dict["weight"] = self.dataset_dict[video]["weight"]

        return return_dict


if __name__ == "__main__":
    import options_expand as options

    args = options.parser.parse_args()
    np.random.seed(args.seed)

    dat = Dataset(args)
    dat.save_dat_mini()
    # dat.save_dat()