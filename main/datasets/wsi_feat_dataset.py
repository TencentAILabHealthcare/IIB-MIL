import pickle

import numpy as np
import torch
import torch.utils.data as data_utils


class WSIFeatDataset(data_utils.Dataset):
    def __init__(
        self,
        ann_file=None,
        transforms=None,
        num_class=2,
        is_training=False,
        feature_len=500,
        name2patch2score=None,
        patch_level=True,
    ):
        super().__init__()

        self.patch_level = patch_level
        self.is_training = is_training
        with open(ann_file, "rt") as infile:
            data = infile.readlines()

        self.wsi_loc = []
        for each in data:
            pid, feat_fp, label = each.strip().split(",")
            label = int(label)
            self.wsi_loc.append((pid, feat_fp, label))
        # modify by sunkia
        self.name2patch2score = name2patch2score
        # 按照WSI ID排序
        sorted(self.wsi_loc, key=lambda x: x[0])

        self.num_class = num_class

        self.pid_2_img_id = {}
        for idx, v in enumerate(self.wsi_loc):
            self.pid_2_img_id[v[0]] = idx

        self._transforms = transforms
        self.feature_len = feature_len
        self.is_training = is_training

    def __len__(self):
        return len(self.wsi_loc)

    def __getitem__(self, index):
        feat_len = self.feature_len
        image_id, patch_feat_fp, label = self.wsi_loc[index]
        label = torch.tensor(label).long()

        # load a wsi
        patch_feat = []
        with open(patch_feat_fp, "rb") as infile:
            patch_feat = pickle.load(infile)

        # train or val
        ret_feat = []
        ret_feat_name = []
        for each_ob in patch_feat:
            if "tr" in each_ob.keys():
                c_feat = each_ob["tr"]
                select_idx = np.random.choice(c_feat.shape[0], 1).item()
                select_feat = c_feat[select_idx]
                select_feat_name = each_ob["feat_name"]
            else:
                select_feat = each_ob["val"]
                select_feat_name = each_ob["feat_name"]

            select_feat = select_feat.reshape(-1)
            ret_feat.append(select_feat)
            ret_feat_name.append(select_feat_name)

        ret_feat = np.stack(ret_feat)
        feat_dim = ret_feat.shape[-1]
        if self.is_training:
            pp = np.random.random_sample()
            if pp > 0.5:  ####0
                xx = len(ret_feat)
                xx = list(np.random.permutation(xx))
                ret_feat = ret_feat[xx, :]
                ret_feat_name = [ret_feat_name[xxx] for xxx in xx]

        # useful mask
        mask = np.ones(len(ret_feat))

        # cut long tail
        if len(ret_feat) > feat_len:
            ret_feat = ret_feat[:feat_len]
            mask = mask[:feat_len]
            ret_feat_name = ret_feat_name[:feat_len]

        # impute # less than feat_len
        cur_len = len(ret_feat)
        if cur_len < feat_len:
            residual = feat_len - len(ret_feat)
            if 0:
                pass
            else:
                append_data = np.zeros((residual, ret_feat.shape[1]))
                append_data_name = [ret_feat_name[0]] * residual
                #  append_mask = np.ones(residual)
                append_mask = np.zeros(residual)

                ret_feat = np.concatenate([ret_feat, append_data])
                mask = np.concatenate([mask, append_mask])
                ret_feat_name = ret_feat_name + append_data_name

        # type setting
        ret_feat = torch.from_numpy(ret_feat).float()
        mask = torch.from_numpy(mask)
        mask = mask.bool()

        if self.patch_level:
            label = label.reshape(
                -1,
            ).repeat(len(mask))

        if self.num_class == 2:
            if int(list(label.numpy())[0]) == 1:
                batchY = [[1, 1] for _ in range(feat_len)]
            elif int(list(label.numpy())[0]) == 0:
                batchY = [[1, 1] for _ in range(feat_len)]
        elif self.num_class == 3:
            batchY = [[1, 1, 1] for _ in range(feat_len)]

        if self.is_training:
            target = {
                "label": label,
                "pid": image_id,
                "patch_name": ret_feat_name,
                "index": [self.confidence_dir[image_id + x] for x in ret_feat_name],
                "batchY": batchY,
            }
        else:
            target = {
                "label": label,
                "pid": image_id,
                "patch_name": ret_feat_name,
            }

        return ret_feat, mask, target


def build(input_data_file, image_set, args):
    scale = args.DATASET.DATASET_SCALE
    num_class = args.MODEL.NUM_CLASSES
    feature_len = args.DATASET.FEATURE_LEN
    score_path = args.DATASET.PATCH_SCORE_PATH
    patch_level = args.DATASET.PATCH_LEVEL

    print(f"Build dataset : {scale} with num class {num_class}")
    dataset = WSIFeatDataset(
        ann_file=input_data_file,
        transforms=None,
        num_class=num_class,
        is_training=args.is_training,
        feature_len=feature_len,
        name2patch2score=score_path,
        patch_level=patch_level,
    )

    if args.is_training:
        num_class = dataset.num_class
        wsi_loc = dataset.wsi_loc

        idx = 0
        confidence = []
        confidence_dir = {}
        for index in range(len(wsi_loc)):
            image_id, patch_feat_fp, label = wsi_loc[index]
            patch_feat = []
            with open(patch_feat_fp, "rb") as infile:
                patch_feat = pickle.load(infile)

            patch_name = []
            for each_ob in patch_feat:
                patch_name = each_ob["feat_name"]
                if num_class == 2:
                    if label == 0:
                        confidence.append([1.0 - 1e-6, 0.0 + 1e-6])
                        confidence_dir[image_id + patch_name] = idx
                        idx += 1
                    elif label == 1:
                        confidence.append([0.0 + 1e-6, 1.0 - 1e-6])
                        confidence_dir[image_id + patch_name] = idx
                        idx += 1
                    else:
                        print(patch_name)
                elif num_class == 3:
                    if label == 0:
                        confidence.append([1.0 - 1e-6, 0.0 + 5e-7, 0.0 + 5e-7])
                        confidence_dir[image_id + patch_name] = idx
                        idx += 1
                    elif label == 1:
                        confidence.append([0.0 + 5e-7, 1.0 - 1e-6, 0.0 + 5e-7])
                        confidence_dir[image_id + patch_name] = idx
                        idx += 1
                    elif label == 2:
                        confidence.append([0.0 + 5e-7, 0.0 + 5e-7, 1.0 - 1e-6])
                        confidence_dir[image_id + patch_name] = idx
                        idx += 1
                    else:
                        print(patch_name)
        dataset.confidence = torch.tensor(confidence)
        dataset.confidence_dir = confidence_dir

    return dataset
