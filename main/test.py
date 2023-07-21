import argparse
import copy
import datetime
import os
import os.path as osp
import pprint
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import util.misc as utils
from datasets import build_dataset
from models import build_iibmil_model
from torch.utils.data import DataLoader
from util.logger import getLog

sys.path.insert(0, osp.abspath("./"))
from configs import get_cfg_defaults, update_default_cfg
from main.engine import evaluate


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        if "data_archive" in root or "vis" in root or "eps":
            # print(f'Ignore {root}')
            continue
        for file in files:
            write_fp = os.path.join(root, file)
            if "core." in write_fp:
                continue
            ziph.write(write_fp)


def main(cfg):
    time_now = datetime.datetime.now()

    unique_comment = f"{cfg.MODEL.MODEL_NAME}{time_now.month}{time_now.day}{time_now.hour}{time_now.minute}"
    cfg.TRAIN.OUTPUT_DIR = osp.join(cfg.TRAIN.OUTPUT_DIR, unique_comment)
    os.makedirs(cfg.TRAIN.OUTPUT_DIR, exist_ok=True)

    logger = getLog(cfg.TRAIN.OUTPUT_DIR + "/log.txt", screen=True)

    # TODO
    utils.init_distributed_mode(cfg)
    logger.info("git:\n  {}\n".format(utils.get_sha()))

    logger.info(cfg)

    device = torch.device("cuda")

    seed = cfg.TRAIN.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_name = cfg.MODEL.MODEL_NAME
    logger.info(f"Build Model: {model_name}")
    if model_name == "iibmil":
        model, criterion = build_iibmil_model(cfg)
        model.to(device)
    else:
        logger.info(f"Model name not found, {model_name}")
        raise ValueError(f"Model name not found")

    if cfg.TRAIN.LOSS_NAME == "be":
        criterion = nn.CrossEntropyLoss()

    if cfg.MULTI_LABEL:
        from util.loss import BCEWithLogitsLossWrapper

        criterion = BCEWithLogitsLossWrapper()

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params:  {n_parameters}")
    dataset_test = build_dataset(cfg.input_data_file, image_set="test", args=cfg)

    print("dataset...")
    print(len(dataset_test))

    if cfg.distributed:
        logger.info("ddp is not implemented")
        raise ModuleNotFoundError(f"ddp is not implemented")
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    collate_func = (
        utils.collate_fn_multi_modal
        if cfg.DATASET.DATASET_NAME in ("vilt", "vilt_surv", "unit")
        else utils.collate_fn
    )

    data_loader_test = DataLoader(
        dataset_test,
        cfg.TRAIN.BATCH_SIZE,
        sampler=sampler_test,
        drop_last=False,
        collate_fn=collate_func,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
    )

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, cfg.MODEL.LR_BACKBONE_NAME)
                and not match_name_keywords(n, cfg.MODEL.LR_LINEAR_PROJ_NAME)
                and p.requires_grad
            ],
            "lr": cfg.TRAIN.LR,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, cfg.MODEL.LR_BACKBONE_NAME)
                and p.requires_grad
            ],
            "lr": cfg.TRAIN.LR_BACKBONE,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, cfg.MODEL.LR_LINEAR_PROJ_NAME)
                and p.requires_grad
            ],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.LR_LINEAR_PROJ_MULT,
        },
    ]
    if cfg.TRAIN.OPTIM_NAME == "sgd":
        optimizer = torch.optim.SGD(
            param_dicts,
            lr=cfg.TRAIN.LR,
            momentum=0.9,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )
    elif cfg.TRAIN.OPTIM_NAME == "adamw":
        optimizer = torch.optim.AdamW(
            model_without_ddp.parameters(),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )
    else:
        optimizer = torch.optim.Adam(
            model_without_ddp.parameters(),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP)

    output_dir = Path(cfg.TRAIN.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    args_dict = vars(args)
    with open(osp.join(output_dir, "params.txt"), "wt") as outfile:
        pprint.pprint(args_dict, indent=4, stream=outfile)

    # backup file
    backup_fname = osp.join(cfg.TRAIN.OUTPUT_DIR, "code_backup.zip")
    c_dir_path = os.path.dirname(os.path.realpath(__file__))
    logger.info(f"Backup file from {c_dir_path} to {backup_fname}")
    zipf = zipfile.ZipFile(backup_fname, "w", zipfile.ZIP_DEFLATED)
    zipdir(c_dir_path, zipf)
    zipf.close()
    logger.info(f"Backup finish")

    # load if checkpoint
    if cfg.TRAIN.RESUME_PATH:
        print("loading model........")
        logger.info(f"resume from {cfg.TRAIN.RESUME_PATH}")
        if cfg.TRAIN.RESUME_PATH.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.TRAIN.RESUME_PATH, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(cfg.TRAIN.RESUME_PATH, map_location="cpu")
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint["model"], strict=False
        )
        unexpected_keys = [
            k
            for k in unexpected_keys
            if not (k.endswith("total_params") or k.endswith("total_ops"))
        ]
        if len(missing_keys) > 0:
            logger.info("Missing Keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Unexpected Keys: {}".format(unexpected_keys))
        if (
            not cfg.TRAIN.EVAL
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            # import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint["optimizer"])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg["lr"] = pg_old["lr"]
                pg["initial_lr"] = pg_old["initial_lr"]
            logger.info(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            cfg.override_resumed_lr_drop = True
            if cfg.override_resumed_lr_drop:
                logger.info(
                    "Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler."
                )
                lr_scheduler.step_size = cfg.TRAIN.LR_DROP
                lr_scheduler.base_lrs = list(
                    map(lambda group: group["initial_lr"], optimizer.param_groups)
                )
            lr_scheduler.step(lr_scheduler.last_epoch)
            cfg.TRAIN.START_EPOCH = checkpoint["epoch"] + 1

    test_stats, test_result = evaluate(
        logger,
        model,
        criterion,
        data_loader_test,
        device,
        output_dir,
        cfg.distributed,
        display_header="Test",
        kappa_flag=cfg.TRAIN.KAPPA,
    )
    print(test_stats)
    test_record = {
        "ID": test_result["img_id"],
        "negative": test_result["pred"][:, 0],
        "positive": test_result["pred"][:, 1],
        "label": test_result["label"],
    }
    df = pd.DataFrame(test_record)
    df.to_csv(os.path.join(cfg.TRAIN.OUTPUT_DIR, "test_record.csv"))
    print(cfg.TRAIN.OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI training and evaluation script")
    parser.add_argument(
        "--cfg",
        default="./configs/TCGA-RCC.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--input_data_file",
        default="./data/TCGA-RCC.txt",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--is_training",
        default=0,
        help="is_training",
        type=int,
    )
    parser.add_argument(
        "--output_file",
        default="default.csv",
        help="",
        type=str,
    )
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    update_default_cfg(cfg)

    cfg.input_data_file = args.input_data_file
    cfg.is_training = args.is_training

    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(cfg)
