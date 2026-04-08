import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _save_model_weights(model, output_path):
    torch.save(_unwrap_model(model).state_dict(), output_path)


def _save_training_checkpoint(cfg, model, optimizer, optimizer_center, scheduler, scaler, epoch, best_mAP, output_path):
    checkpoint = {
        'epoch': epoch,
        'model': _unwrap_model(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_center': optimizer_center.state_dict(),
        'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'scaler': scaler.state_dict(),
        'best_mAP': best_mAP,
        'config': str(cfg),
    }
    torch.save(checkpoint, output_path)


def _run_validation(model, val_loader, evaluator, device):
    model.eval()
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
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
             num_query, local_rank,
             start_epoch=1,
             best_mAP=0.0,
             scaler_state=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        optimizer_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        _optimizer_to_device(optimizer, optimizer_device)
        _optimizer_to_device(optimizer_center, optimizer_device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    # train
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                model_outputs = model(img, target, cam_label=target_cam, view_label=target_view)
                if isinstance(model_outputs, tuple) and len(model_outputs) == 3:
                    score, feat, aux_outputs = model_outputs
                else:
                    score, feat = model_outputs
                    aux_outputs = None
                loss = loss_fn(score, feat, target, target_cam, aux_outputs)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

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

        save_periodic = checkpoint_period > 0 and epoch % checkpoint_period == 0
        if save_periodic:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    _save_model_weights(model, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                _save_model_weights(model, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if cfg.SOLVER.SAVE_LATEST:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    _save_training_checkpoint(
                        cfg, model, optimizer, optimizer_center, scheduler, scaler, epoch, best_mAP,
                        os.path.join(cfg.OUTPUT_DIR, 'latest_checkpoint.pth')
                    )
                    _save_model_weights(model, os.path.join(cfg.OUTPUT_DIR, 'latest_model.pth'))
            else:
                _save_training_checkpoint(
                    cfg, model, optimizer, optimizer_center, scheduler, scaler, epoch, best_mAP,
                    os.path.join(cfg.OUTPUT_DIR, 'latest_checkpoint.pth')
                )
                _save_model_weights(model, os.path.join(cfg.OUTPUT_DIR, 'latest_model.pth'))
        elif cfg.SOLVER.SAVE_EVERY_EPOCH:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    _save_training_checkpoint(
                        cfg, model, optimizer, optimizer_center, scheduler, scaler, epoch, best_mAP,
                        os.path.join(cfg.OUTPUT_DIR, 'checkpoint_epoch_{}.pth'.format(epoch))
                    )
            else:
                _save_training_checkpoint(
                    cfg, model, optimizer, optimizer_center, scheduler, scaler, epoch, best_mAP,
                    os.path.join(cfg.OUTPUT_DIR, 'checkpoint_epoch_{}.pth'.format(epoch))
                )

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    cmc, mAP, _, _, _, _, _ = _run_validation(model, val_loader, evaluator, device)
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    if cfg.SOLVER.SAVE_BEST and mAP >= best_mAP:
                        best_mAP = mAP
                        _save_training_checkpoint(
                            cfg, model, optimizer, optimizer_center, scheduler, scaler, epoch, best_mAP,
                            os.path.join(cfg.OUTPUT_DIR, 'best_checkpoint.pth')
                        )
                        _save_model_weights(model, os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
                        logger.info("Saved new best checkpoint with mAP: {:.1%}".format(best_mAP))
                    if cfg.SOLVER.SAVE_LATEST:
                        _save_training_checkpoint(
                            cfg, model, optimizer, optimizer_center, scheduler, scaler, epoch, best_mAP,
                            os.path.join(cfg.OUTPUT_DIR, 'latest_checkpoint.pth')
                        )
                    torch.cuda.empty_cache()
            else:
                cmc, mAP, _, _, _, _, _ = _run_validation(model, val_loader, evaluator, device)
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                if cfg.SOLVER.SAVE_BEST and mAP >= best_mAP:
                    best_mAP = mAP
                    _save_training_checkpoint(
                        cfg, model, optimizer, optimizer_center, scheduler, scaler, epoch, best_mAP,
                        os.path.join(cfg.OUTPUT_DIR, 'best_checkpoint.pth')
                    )
                    _save_model_weights(model, os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
                    logger.info("Saved new best checkpoint with mAP: {:.1%}".format(best_mAP))
                if cfg.SOLVER.SAVE_LATEST:
                    _save_training_checkpoint(
                        cfg, model, optimizer, optimizer_center, scheduler, scaler, epoch, best_mAP,
                        os.path.join(cfg.OUTPUT_DIR, 'latest_checkpoint.pth')
                    )
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
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


