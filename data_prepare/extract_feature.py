"""
Extract feature offline
"""
import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import argparse
import os.path as osp
import pickle
from concurrent.futures import ThreadPoolExecutor

import albumentations as alb
import cv2
import numpy as np
import pretrainedmodels
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from PIL import Image
from rich import progress
from torch import nn

from configs import get_cfg_defaults, update_default_cfg

"""
extract path features
- WSI id1
    - patch1 feat.pkl --> dict({'val': [1280] , 'tr': })
    - patch2 feat.pkl
- WSI id2
    - patch1 feat.pkl
    - patch2 feat.pkl

"""
tr_trans = alb.Compose(
    [
        alb.Resize(224, 224),
        alb.RandomRotate90(),
        alb.RandomBrightnessContrast(),
        alb.HueSaturationValue(),
        # alb.GaussNoise(),
        # alb.GaussianBlur(),
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
        # alb.CLAHE(),
        alb.CoarseDropout(max_holes=4),
        alb.Normalize(),
        ToTensorV2(),
    ]
)

val_trans = alb.Compose(
    [
        alb.Resize(224, 224),
        alb.Normalize(),
        ToTensorV2(),
    ]
)


class PatchDataset(data_utils.Dataset):
    def __init__(self, img_fp_list, trans):
        print("dataset for imagenet")
        self.img_fp_list = img_fp_list
        self.trans = trans

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, idx):
        try:
            img_fp = self.img_fp_list[idx]
            img = cv2.imread(img_fp)
            img = img[:, :, ::-1]
        except:
            # if img is None:
            img = np.zeros((224, 224, 3)).astype("uint8")

        pid = osp.basename(osp.dirname(img_fp))
        img_bname = osp.basename(img_fp).rsplit(".", 1)[0]
        aug_img = self.trans(image=img)["image"]
        return pid, img_bname, aug_img


class PatchDatasetCLR(data_utils.Dataset):
    def __init__(self, img_fp_list):
        print("dataset for CLR")
        self.img_fp_list = img_fp_list

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, idx):
        try:
            img_fp = self.img_fp_list[idx]
            img_pil = Image.open(img_fp)
            img = cv2.resize(np.array(img_pil), (224, 224))
        except:
            # if img is None:
            img = np.zeros((224, 224, 3)).astype("uint8")
        pid = osp.basename(osp.dirname(img_fp))
        img_bname = osp.basename(img_fp).rsplit(".", 1)[0]
        aug_img = transforms.functional.to_tensor(img)
        return pid, img_bname, aug_img


def save_feat_in_thread(batch_pid, batch_img_bname, batch_tr_feat, save_dir):
    for b_idx, (pid, img_bname, tr_feat) in enumerate(
        zip(batch_pid, batch_img_bname, batch_tr_feat)
    ):
        feat_save_dir = osp.join(save_dir, pid)
        os.makedirs(feat_save_dir, exist_ok=True)
        feat_save_name = osp.join(feat_save_dir, f"{img_bname}.pkl")

        if osp.exists(feat_save_name):
            try:
                with open(feat_save_name, "rb") as infile:
                    save_dict = pickle.load(infile)
            except:
                save_dict = {}
        else:
            save_dict = {}

        tr_aug_feat = np.stack([tr_feat])
        if "tr" in save_dict.keys():
            save_dict["tr"] = np.concatenate([tr_aug_feat, save_dict["tr"]])
        else:
            save_dict["tr"] = tr_aug_feat

        # print(f'Save to : {feat_save_name}')
        with open(feat_save_name, "wb") as outfile:
            pickle.dump(save_dict, outfile)


def save_val_feat_in_thread(batch_pid, batch_img_bname, batch_val_feat, save_dir):
    for b_idx, (pid, img_bname, val_feat) in enumerate(
        zip(batch_pid, batch_img_bname, batch_val_feat)
    ):
        feat_save_dir = osp.join(save_dir, pid)
        os.makedirs(feat_save_dir, exist_ok=True)
        feat_save_name = osp.join(feat_save_dir, f"{img_bname}.pkl")

        if osp.exists(feat_save_name):
            with open(feat_save_name, "rb") as infile:
                save_dict = pickle.load(infile)
        else:
            save_dict = {}

        save_dict["val"] = val_feat

        with open(feat_save_name, "wb") as outfile:
            pickle.dump(save_dict, outfile)


def pred_and_save_with_dataloader(
    model, img_fp_list, save_dir, batch_size=1, dataset_class="imagenet"
):
    print("start dataloader..")
    model.eval()

    import multiprocessing

    num_processes = multiprocessing.cpu_count()
    num_processes = min(48, num_processes)

    executor = ThreadPoolExecutor(max_workers=num_processes)

    val_dl = torch.utils.data.DataLoader(
        PatchDataset(img_fp_list, trans=val_trans)
        if dataset_class == "imagenet"
        else PatchDatasetCLR(img_fp_list),
        batch_size=batch_size,
        num_workers=num_processes,
        shuffle=False,
        drop_last=False,
    )

    for batch in progress.track(val_dl):
        batch_pid, batch_img_bname, batch_tr_img = batch
        batch_tr_img = batch_tr_img.cuda()

        with torch.no_grad():
            val_feat = model(batch_tr_img)
            val_feat = val_feat.cpu().numpy()
            executor.submit(
                save_val_feat_in_thread, batch_pid, batch_img_bname, val_feat, save_dir
            )
    print("task finished..")


class EffNet(nn.Module):
    def __init__(self, efname="efficientnet-b0", model_path=""):
        super(EffNet, self).__init__()
        self.model = EfficientNet.from_name(efname)
        print(f"Load pretrain model from {model_path}")
        self.model.load_state_dict(torch.load(model_path))

    def forward(self, data):
        bs = data.shape[0]
        feat = self.model.extract_features(data)
        feat = nn.functional.adaptive_avg_pool2d(feat, output_size=(1))
        feat = feat.view(bs, -1)
        return feat


class PretrainedNet(nn.Module):
    def __init__(self, model_name="se_resnet50", model_path="", num_classes=2):
        super(PretrainedNet, self).__init__()
        self.model = pretrainedmodels.__dict__[model_name](
            num_classes=1000, pretrained=None
        )
        self.model.last_linear = torch.nn.Linear(
            self.model.last_linear.in_features, num_classes
        )
        print(f"Load pretrain model from {model_path}")
        self.model.load_state_dict(torch.load(model_path))

    def forward(self, data):
        bs = data.shape[0]
        feat = self.model.features(data)
        feat = nn.functional.adaptive_avg_pool2d(feat, output_size=(1))
        feat = feat.view(bs, -1)
        return feat  # 2048


def get_model(model_name="", model_path="", num_classes=1000):
    if model_name == "efficientnet-b0":
        model = EffNet(efname="efficientnet-b0", model_path=model_path)
    else:
        model = pretrainedmodels.__dict__[model_name](
            num_classes=num_classes, pretrained="imagenet"
        )
        print(f"Load pretrain model from {model_path}")
    return model


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI patch features extraction")
    parser.add_argument(
        "--cfg",
        default=None,
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    update_default_cfg(cfg)

    wsi_patch_info = cfg.PATCH.IDX2IMG
    model_name = cfg.PRETRAIN.MODEL_NAME
    model_path = cfg.PRETRAIN.MODEL_PATH
    save_dir = cfg.PRETRAIN.SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    # creat model
    model = get_model(model_name=model_name, model_path=model_path)  # efficientnet
    model = nn.DataParallel(model).cuda()

    # load path of original images
    print("load img path..")
    img_fp_list = []
    with open(wsi_patch_info, "rb") as fp:
        dt = pickle.load(fp)
        for k, v in dt.items():
            img_fp_list.extend(v)
    print(f"Len of img {len(img_fp_list)}")

    img_fp_list = sorted(img_fp_list)

    # extract feature to *.pkl
    pred_and_save_with_dataloader(model, img_fp_list, save_dir=save_dir, batch_size=1)
