# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import datetime
import logging
import time
import argparse
import os
import pandas as pd

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

import numpy as np
import ipdb
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


from maskrcnn_benchmark.utils.metric_logger import MetricLogger
#import maskrcnn_benchmark.modeling.detector.generalized_rcnn as grcnn
import maskrcnn_benchmark.modeling.detector.pertrain_cnn as pcnn

CACHE_PATH = "/media/zhen/Research/deepsz_pytorch/"

import tools.pretrail_utils as putils


def do_train(
        args,
        model,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    start_iter = arguments["iteration"]
    curr_iter, curr_base = start_iter, 0
    start_training_time = time.time()
    end = time.time()
    val_hist = {"loss":{}, "acc":{}}
    train_hist = {'loss':{}, "acc":{}}
    for epoch in range(args.num_epochs):
        curr_ratio = (epoch // args.change_ratio_after) + 1
        curr_ratio = min(curr_ratio, args.ratio_up_to)
        data_loader = putils.DEEPSZ(which='train', ratio=curr_ratio, oversample_pos=True)
        max_iter_curr_epoch = len(data_loader)
        if max_iter_curr_epoch < curr_iter - curr_base:
            curr_base += max_iter_curr_epoch
            continue

        for curr_i in putils.ProgressBar(range(curr_iter - curr_base, max_iter_curr_epoch)):
            images, targets, _ = data_loader[curr_i]
            model.train()

            if any(len(target) < 1 for target in targets):
                logger.error(
                    f"Iteration={curr_iter + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
                continue
            data_time = time.time() - end

            curr_iter = curr_iter + 1
            arguments["iteration"] = curr_iter

            scheduler.step()

            images = images.to(device)
            targets = targets.to(device)

            loss_dict, softmax = model(images, targets)
            train_hist['loss'][curr_iter] = float(loss_dict['loss_classifier'].cpu().detach())
            _y = targets[:,1].cpu().detach().numpy().astype(int)
            _yhat = (softmax[:,1].cpu().detach().numpy() > 0.5).astype(int)
            train_hist['acc'][curr_iter] = (_yhat == _y).astype(float).mean()

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            meters.update(loss=losses, **loss_dict)

            optimizer.zero_grad()
            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter_curr_epoch - curr_iter)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if curr_iter % 20 == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "ratio: {ratio}",
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        ratio=curr_ratio,
                        eta=eta_string,
                        iter=curr_iter,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                if curr_iter % 100 == 0:
                    val_ret = run_test(model, cfg, args, which='valid', ratio=curr_ratio)
                    val_hist['loss'][curr_iter] = val_ret['loss']
                    val_hist['acc'][curr_iter] = val_ret['acc']
                    print(val_ret)
            if curr_iter % checkpoint_period == 0:
                checkpointer.save("model_{:d}-{:07d}".format(epoch, curr_iter), **arguments)
        ret = run_test(model, cfg, args, which='test', ratio=None)
        ret = {"test": ret, "val":val_hist, "train":train_hist}
        curr_base += max_iter_curr_epoch

        to_pickle(ret, os.path.join(cfg.OUTPUT_DIR, "results", "epoch%d.pkl"%epoch))

    checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter_curr_epoch)
        )
    )

def to_pickle(d, path):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return pd.to_pickle(d,path)

def train(cfg, args):
    #model = grcnn.GeneralizedRCNN(cfg)#TODO: change this
    model = pcnn.PretrainCNN(cfg)


    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
    #ipdb.set_trace()

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        args,
        model,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def run_test(model, cfg, args, **kwargs):

    device = torch.device(cfg.MODEL.DEVICE)

    torch.cuda.empty_cache()  # TODO check if it helps
    data_loader = putils.DEEPSZ(**kwargs)
    model.eval()
    output_folder = os.path.join(CACHE_PATH, "")
    if not os.path.isdir(output_folder): os.makedirs(output_folder)

    y_preds, ys = [], []
    losses = []
    for epoch in range(args.num_epochs):
        for iteration, (images, targets, _) in enumerate(data_loader, 0):
            images = images.to(device)
            targets = targets.to(device)
            loss_dict, softmax = model(images, targets)
            losses.append(float(loss_dict['loss_classifier'].cpu().detach()))
            y_preds.append(softmax[:,1].cpu().detach().numpy())
            ys.append(targets[:, 1].cpu().detach().numpy())

    ret = {"y_pred":np.concatenate(y_preds),
           "y":np.concatenate(ys).astype(int),
           "loss": np.asarray(losses).mean()}

    ret['acc'] = ((ret['y_pred'] > 0.5).astype(int) == ret['y']).astype(float).mean()
    return ret




def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    parser.add_argument(
        "--num_epochs",
        help="",
        default=20,
        type=int,
    )


    parser.add_argument(
        "--ratio_up_to",
        help="",
        default=20,
        type=int,
    )

    parser.add_argument(
        "--change_ratio_after",
        help="",
        default=1,
        type=int,
    )


    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )



    args = parser.parse_args()


    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = False

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_path = os.path.join(CACHE_PATH, "ratio{}-{}_convbody={}_lr={}_wd={}".format(args.change_ratio_after,
                                                                                       args.ratio_up_to,
                                                                                       cfg['MODEL']['BACKBONE']['CONV_BODY'],
                                                                                       cfg['SOLVER']['BASE_LR'],
                                                                                       cfg['SOLVER']['WEIGHT_DECAY']))
    cfg.merge_from_list(["OUTPUT_DIR", output_path])
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args)

    if not args.skip_test:
        run_test(model, cfg, args)


if __name__ == "__main__":
    main()
