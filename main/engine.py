import numpy as np
import torch
import util.custom_metrics as custom_metrics
import util.misc as utils
from sklearn import metrics
from util.gpu_gather import GpuGather


@torch.no_grad()
def evaluate(
    logger,
    model,
    criterion,
    data_loader,
    device,
    output_dir,
    is_distributed,
    display_header="Valid",
    kappa_flag=False,
):
    model.eval()

    try:
        criterion.eval()
    except:
        pass

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"{display_header}:"

    gpu_gather = GpuGather(is_distributed=is_distributed)
    print_freq = len(data_loader) // 4
    print_freq = max(1, print_freq)

    IMG_id = []

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        pid = []
        for x in targets:
            pid_now = x["pid"]
            pid.append(pid_now)
        IMG_id += pid

        samples = samples.to(device)
        mask = samples.mask

        bs, N, num_cls = (
            samples.tensors.shape[0],
            samples.tensors.shape[1],
            model.prototypes.shape[0],
        )

        try:
            label = torch.tensor([x["label"].cpu().numpy() for x in targets])
        except:
            label = torch.tensor([x["label"] for x in targets])

        assert not np.any(np.isnan(label.cpu().numpy())), f"Label is nan"

        label = label[:, 0].detach()

        try:
            outputs_tuple = model(samples)
        except Exception as e:
            logger.info(samples.target)
            raise e

        wsi_pre = torch.softmax(outputs_tuple[3].detach(), dim=-1)

        loss = criterion(wsi_pre, label.cuda())

        gpu_gather.update(pred=wsi_pre.detach().cpu().numpy())
        gpu_gather.update(label=label.cpu().numpy().reshape(-1))

        loss_dict = {"loss": loss}
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_value = loss_dict_reduced["loss"].item()
        metric_logger.update(loss=loss_dict_reduced["loss"].item())

    # gather the stats from all processes
    gpu_gather.synchronize_between_processes()

    pred = gpu_gather.pred
    pred = np.concatenate(pred)
    num_of_class = pred.shape[1]

    label = gpu_gather.label
    label = np.concatenate(label)

    # multi-label
    if len(label) > len(pred):
        label = label.reshape(len(pred), -1)

        auc = []
        for i in range(num_of_class):
            pred_i = pred[:, i]
            # pred_i = torch.nn.sigmoid(pred_i)
            labek_i = label[:, i]
            if np.sum(labek_i) > 0:
                auc_i = metrics.roc_auc_score(labek_i, pred_i)
                auc.append(auc_i)

        logger.info(f"{display_header} AUC: {auc}")
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # logger.info("Averaged stats:", metric_logger)
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats["auc"] = np.mean(auc)
        stats["acc"] = 0
        stats["f1"] = 0
        stats["recall"] = 0
        stats["precision"] = 0

        eval_result = {}
        eval_result["label"] = label
        eval_result["pred"] = pred
        eval_result["imd_id"] = IMG_id

        return stats, eval_result
    else:
        try:
            if num_of_class == 1:
                auc = custom_metrics.c_index(label, pred)
            elif num_of_class == 2:
                auc = metrics.roc_auc_score(label, pred[:, 1])
            else:
                auc = []
                for i in range(num_of_class):
                    pred_i = pred[:, i]
                    labek_i = np.eye(num_of_class)[label][:, i]
                    if np.sum(labek_i) > 0:
                        auc_i = metrics.roc_auc_score(labek_i, pred_i)
                        auc.append(auc_i)
                auc = np.mean(auc)
        except Exception as e:
            print(e)
            auc = 0

    if num_of_class > 1:
        cls_report = metrics.classification_report(label, np.argmax(pred, axis=1))
        cfm = metrics.confusion_matrix(label, np.argmax(pred, axis=1))

        logger.info(f"report: \n {cls_report}")
        logger.info(f"confusion matrix: \n {cfm}")

        pred_label = np.argmax(pred, axis=1)
        acc_score = metrics.accuracy_score(label, pred_label)
        if num_of_class == 2:
            f1_score = metrics.f1_score(label, pred_label)
            recall_score = metrics.recall_score(label, pred_label)
            precision_score = metrics.precision_score(label, pred_label)
        else:
            f1_score = metrics.f1_score(
                label, pred_label, average="weighted", zero_division=1
            )
            recall_score = metrics.recall_score(
                label, pred_label, average="weighted", zero_division=1
            )
            precision_score = metrics.precision_score(
                label, pred_label, average="weighted", zero_division=1
            )

        logger.info(
            f"{display_header} AUC: {auc:.4f} Acc: {acc_score:.4f} F1: {f1_score:.4f} Recall: {recall_score:.4f} Precision: {precision_score:.4f}"
        )

        if kappa_flag:
            logger.info(f"ground truth: {label}, prediction: {pred_label}")
            kappa_res = custom_metrics.quadratic_kappa(
                label, pred_label, N=num_of_class
            )
            logger.info(f"quadratic kappa: \n {kappa_res}")

        metric_logger.synchronize_between_processes()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats["auc"] = auc
        stats["acc"] = acc_score
        stats["f1"] = f1_score
        stats["recall"] = recall_score
        stats["precision"] = precision_score
        if kappa_flag:
            stats["kappa"] = kappa_res

        eval_result = {}
        eval_result["label"] = label
        eval_result["pred"] = pred
        eval_result["img_id"] = IMG_id

    else:
        logger.info(f"{display_header} C-index: {auc:.4f}")
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        stats["auc"] = auc
        stats["acc"] = 0
        stats["f1"] = 0
        stats["recall"] = 0
        stats["precision"] = 0

        eval_result = {}
        eval_result["label"] = label
        eval_result["pred"] = pred
        eval_result["imd_id"] = IMG_id
    return stats, eval_result
