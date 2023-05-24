# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Some parts of this code is Copy-paste from DINO library:
https://github.com/facebookresearch/dino
"""
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import wandb
from torchvision.transforms import InterpolationMode
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from scoreloss import SCORELoss
import utils
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('ProMin', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
                        help="""Name of architecture to train. For quick experiments with ViTs,
           we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
           of input square patches - default 16 (for 16x16 patches).""")
    parser.add_argument('--out_dim', default=12000, type=int, help="""Dimensionality of
           the output.""")
    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the projection head.
           we don't use normalization in our experiments""")
    parser.add_argument('--momentum_target', default=0.9, type=float, help="""Base EMA
           parameter for target update. The value is increased to 1 during training with cosine schedule.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
           to use half precision for training. Improves training time and memory requirements,
           but can provoke instability and slight decay of performance. We recommend disabling
           mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.065, help="""Initial value of the
           weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.65, help="""Final value of the
           weight decay. We use a cosine schedule for WD and using a larger decay by
           the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
           gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
           help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=800, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=0, type=int, help="""Number of epochs
           during which we keep the output layer fixed. Typically doing so during
           the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=1e-4, type=float, help="""Learning rate at the end of
           linear warmup (highest LR used during training). The learning rate is linearly scaled
           with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
           end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--lamda', type=float, default=2.4,
                        help='hyper parameter for second part of the ProSMin loss')
    parser.add_argument('--center_momentum', type=float, default=0.9,
                        help='momentum for centering of target results')
    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
           Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
           recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=16, help="""Number of small
           local views to generate. Set this parameter to 0 to disable multi-crop training.
           When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
           Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='',
                        type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir',
                        default="", type=str,
                        help="")
    parser.add_argument('--saveckp_freq', default=50, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
           distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    return parser


def train_ProSMin(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building online and target networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        online = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        target = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = online.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        online = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                pretrained=False, drop_path_rate=args.drop_path_rate)
        target = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = online.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        online = torchvision_models.__dict__[args.arch]()
        target = torchvision_models.__dict__[args.arch]()
        embed_dim = online.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")
    # multi-crop wrapper handles forward with inputs of different resolutions
    online_proSMin = utils.MultiCropWrapper(online, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    target = utils.MultiCropWrapper(
        online,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
        ),
    )
    online = utils.wrapper(online_proSMin, args.out_dim, args.out_dim)
    # move networks to gpu
    online, target = online.cuda(), target.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(online):
        online = nn.SyncBatchNorm.convert_sync_batchnorm(online)
        target = nn.SyncBatchNorm.convert_sync_batchnorm(target)

        # we need DDP wrapper to have synchro batch norms working...
        target = nn.parallel.DistributedDataParallel(target, device_ids=[args.gpu], find_unused_parameters=False)
        target_without_ddp = target.module
    else:
        # target_without_ddp and target are the same thing
        target_without_ddp = target
    online = nn.parallel.DistributedDataParallel(online, device_ids=[args.gpu], find_unused_parameters=False)
    # target and online start with the same weights
    target_without_ddp.load_state_dict(online.module.model.state_dict())
    # there is no backpropagation through the target, so no need for gradients
    for p in target.parameters():
        p.requires_grad = False
    print(f"online and target are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    scoreLoss = SCORELoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.center_momentum,
        args.lamda

    ).cuda()
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(online)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_target, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        online=online,
        target=target,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        score_loss=scoreLoss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting ProSMin training !")
    wandb.init(project="score-project", mode="disabled", tags=["exp300"])
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of ProSMin ... ============
        train_stats = train_one_epoch(online, target, target_without_ddp, scoreLoss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'online': online.state_dict(),
            'target': target.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'score_loss': scoreLoss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(online, target, target_without_ddp, scoreLoss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is
                # regularized
                param_group["weight_decay"] = wd_schedule[it]
        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        # target and online forward passes + compute ProSMin loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            target_output = target(images[:2])  # only the 2 global views pass through the target
            online_mean, online_var = online(images)
            loss, loss1, loss2 = scoreLoss(online_mean, online_var,
                                           target_output)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # online update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(online, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, online,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(online, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, online,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the target
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(online.module.model.parameters(), target_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(loss1=loss1.item())
        metric_logger.update(loss2=loss2.item())
        wandb.log({'totalloss': loss.item(), 'firstloss': loss1.item(), 'secondloss': loss2.item()})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ProSMin', parents=[get_args_parser()])
    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ProSMin(args)
