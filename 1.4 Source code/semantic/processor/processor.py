import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


def extract_global_feature(outputs):
    if isinstance(outputs, dict):
        if outputs.get("global_bn_feat") is not None:
            return outputs["global_bn_feat"]
        return outputs["global_feat"]

    _, feat = outputs
    if isinstance(feat, list):
        return feat[0]
    return feat


def apply_synthetic_occlusion(images, cfg):
    occluded_images = images.clone()
    batch_size, _, height, width = images.shape
    if batch_size == 0:
        return occluded_images

    fill_values = images.mean(dim=(2, 3), keepdim=True)
    source_indices = torch.randperm(batch_size, device=images.device) if batch_size > 1 else None

    for batch_index in range(batch_size):
        if torch.rand(1, device=images.device).item() > cfg.MODEL.OCC_AUG.PROB:
            continue

        target_area = torch.empty(1, device=images.device).uniform_(
            cfg.MODEL.OCC_AUG.MIN_AREA,
            cfg.MODEL.OCC_AUG.MAX_AREA
        ).item() * height * width
        aspect_ratio = torch.empty(1, device=images.device).uniform_(
            cfg.MODEL.OCC_AUG.MIN_ASPECT,
            cfg.MODEL.OCC_AUG.MAX_ASPECT
        ).item()

        occ_height = max(1, min(height, int(round((target_area * aspect_ratio) ** 0.5))))
        occ_width = max(1, min(width, int(round((target_area / max(aspect_ratio, 1e-6)) ** 0.5))))
        top = torch.randint(0, height - occ_height + 1, (1,), device=images.device).item()
        left = torch.randint(0, width - occ_width + 1, (1,), device=images.device).item()

        if cfg.MODEL.OCC_AUG.INTER_PERSON and batch_size > 1:
            source_index = source_indices[batch_index].item()
            if source_index == batch_index:
                source_index = (source_index + 1) % batch_size
            source_top = torch.randint(0, height - occ_height + 1, (1,), device=images.device).item()
            source_left = torch.randint(0, width - occ_width + 1, (1,), device=images.device).item()
            occluded_images[batch_index, :, top:top + occ_height, left:left + occ_width] = (
                images[source_index, :, source_top:source_top + occ_height, source_left:source_left + occ_width]
            )
        else:
            occluded_images[batch_index, :, top:top + occ_height, left:left + occ_width] = fill_values[batch_index]

    return occluded_images


def build_reliability_occlusion(images, cfg):
    occluded_images = images.clone()
    batch_size, _, height, width = images.shape
    occ_mask = torch.zeros(batch_size, 1, height, width, device=images.device, dtype=images.dtype)
    if batch_size == 0:
        return occluded_images, occ_mask

    min_area = cfg.INPUT.REL_OCC_MIN_AREA
    max_area = cfg.INPUT.REL_OCC_MAX_AREA
    min_aspect = cfg.INPUT.REL_OCC_MIN_ASPECT
    fill_value = cfg.INPUT.REL_OCC_FILL

    for batch_index in range(batch_size):
        if torch.rand(1, device=images.device).item() > cfg.INPUT.REL_OCC_PROB:
            continue

        area = height * width
        for _ in range(10):
            target_area = torch.empty(1, device=images.device).uniform_(min_area, max_area).item() * area
            aspect_ratio = torch.empty(1, device=images.device).uniform_(
                min_aspect,
                1.0 / max(min_aspect, 1e-6)
            ).item()
            occ_height = int(round((target_area * aspect_ratio) ** 0.5))
            occ_width = int(round((target_area / max(aspect_ratio, 1e-6)) ** 0.5))
            if 0 < occ_height < height and 0 < occ_width < width:
                top = torch.randint(0, height - occ_height + 1, (1,), device=images.device).item()
                left = torch.randint(0, width - occ_width + 1, (1,), device=images.device).item()
                occluded_images[batch_index, :, top:top + occ_height, left:left + occ_width] = fill_value
                occ_mask[batch_index, :, top:top + occ_height, left:left + occ_width] = 1.0
                break

    return occluded_images, occ_mask


def apply_occlusion_to_semantic_masks(semantic_masks, occ_mask):
    if semantic_masks is None:
        return None
    occluded_masks = semantic_masks.clone()
    occluded_masks[occ_mask.squeeze(1) > 0] = 0
    return occluded_masks


def run_validation(model, val_loader, evaluator, device):
    evaluator.reset()
    model.eval()

    for _, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid))

    return evaluator.compute()

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    epochs = cfg.SOLVER.MAX_EPOCHS
    start_epoch = cfg.SOLVER.START_EPOCH

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    model.to(device)
    if device.type == "cuda":
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(enabled=device.type == "cuda")
    best_mAP = -1.0

    def maybe_save_best(current_mAP):
        nonlocal best_mAP
        if not cfg.TEST.SAVE_BEST:
            return
        if current_mAP <= best_mAP:
            return
        best_mAP = current_mAP
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(model_state, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
    # train
    for epoch in range(start_epoch + 1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, batch in enumerate(train_loader):
            if len(batch) == 5:
                img, vid, target_cam, target_view, semantic_masks = batch
                semantic_masks = semantic_masks.to(device)
            else:
                img, vid, target_cam, target_view = batch
                semantic_masks = None
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=device.type == "cuda"):
                if cfg.MODEL.RELIABILITY_PIPELINE:
                    occ_img, occ_mask = build_reliability_occlusion(img, cfg)
                    occ_semantic_masks = apply_occlusion_to_semantic_masks(semantic_masks, occ_mask)
                    outputs = model(
                        img,
                        target,
                        cam_label=target_cam,
                        view_label=target_view,
                        epoch=epoch,
                        semantic_masks=semantic_masks
                    )
                    occ_outputs = model(
                        occ_img,
                        target,
                        cam_label=target_cam,
                        view_label=target_view,
                        epoch=epoch,
                        semantic_masks=occ_semantic_masks
                    )
                    loss = loss_fn(
                        {
                            "clean_outputs": outputs,
                            "occ_outputs": occ_outputs,
                            "occ_mask": occ_mask,
                        },
                        target,
                        target_cam
                    )
                else:
                    outputs = model(
                        img,
                        target,
                        cam_label=target_cam,
                        view_label=target_view,
                        epoch=epoch,
                        semantic_masks=semantic_masks
                    )
                    loss = loss_fn(outputs, target, target_cam)

                if not cfg.MODEL.RELIABILITY_PIPELINE and cfg.MODEL.OCC_AUG.ENABLED and epoch >= cfg.MODEL.OCC_AUG.START_EPOCH:
                    occ_img = apply_synthetic_occlusion(img, cfg)
                    occ_outputs = model(
                        occ_img,
                        target,
                        cam_label=target_cam,
                        view_label=target_view,
                        epoch=epoch,
                        semantic_masks=semantic_masks
                    )
                    occ_loss = loss_fn(occ_outputs, target, target_cam)
                    consistency_loss = 1.0 - F.cosine_similarity(
                        extract_global_feature(outputs),
                        extract_global_feature(occ_outputs),
                        dim=1
                    ).mean()
                    loss = loss + cfg.MODEL.OCC_AUG.OCC_LOSS_WEIGHT * occ_loss + \
                           cfg.MODEL.OCC_AUG.CONSISTENCY_WEIGHT * consistency_loss

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(outputs, dict):
                acc = (outputs["global_logits"].max(1)[1] == target).float().mean()
            elif isinstance(outputs[0], list):
                acc = (outputs[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (outputs[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if checkpoint_period > 0 and (epoch % checkpoint_period == 0 or epoch == epochs):
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if eval_period > 0 and (epoch % eval_period == 0 or epoch == epochs):
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    cmc, mAP, _, _, _, _, _ = run_validation(model, val_loader, evaluator, device)
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    maybe_save_best(mAP)
                    torch.cuda.empty_cache()
            else:
                cmc, mAP, _, _, _, _, _ = run_validation(model, val_loader, evaluator, device)
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                maybe_save_best(mAP)
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device.type == "cuda":
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


