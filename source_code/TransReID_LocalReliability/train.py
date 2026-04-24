import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import argparse
# from timm.scheduler import create_scheduler
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def resume_training_if_needed(cfg, model, optimizer, optimizer_center, scheduler, logger):
    start_epoch = 1
    best_mAP = 0.0
    scaler_state = None
    if not cfg.SOLVER.RESUME:
        return start_epoch, best_mAP, scaler_state

    resume_path = cfg.SOLVER.RESUME_PATH
    if not resume_path:
        raise ValueError("SOLVER.RESUME is True but SOLVER.RESUME_PATH is empty")

    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer_center.load_state_dict(checkpoint['optimizer_center'])
    if checkpoint.get('scheduler') is not None and hasattr(scheduler, 'load_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler'])

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_mAP = checkpoint.get('best_mAP', 0.0)
    scaler_state = checkpoint.get('scaler')
    logger.info("Resumed training from {} at epoch {} with best mAP {:.1%}".format(
        resume_path, start_epoch, best_mAP
    ))
    return start_epoch, best_mAP, scaler_state

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    start_epoch, best_mAP, scaler_state = resume_training_if_needed(
        cfg, model, optimizer, optimizer_center, scheduler, logger
    )

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank,
        start_epoch=start_epoch,
        best_mAP=best_mAP,
        scaler_state=scaler_state
    )
