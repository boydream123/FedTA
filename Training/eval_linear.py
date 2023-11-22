# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.utils import *
from src.logger import PD_Stats

# (
#     bool_flag,
#     initialize_exp,
#     restart_from_checkpoint,
#     fix_random_seeds,
#     AverageMeter,
#     init_distributed_mode,
#     accuracy,
# )
import src.resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")
#########################
#### 我的 parameters ####
#########################
parser.add_argument("--gpu", type=str, default="0,1,2,3",
                    help="指定gpu，gpu3可选0,1,2,3")
parser.add_argument("--num_clients", type=int, default=16,
                    help="参与训练的客户数量")
parser.add_argument("--cifar_mean", type=bool_flag, default=True,
                    help="是否开启cifar的平均值，否则使用imagenet平均值")
parser.add_argument("--multi_crop", type=bool_flag, default=True,
                    help="是否开启multiCrop")
parser.add_argument("--checkpoint_freq", type=int, default=100,
                    help="Save the model periodically")

#########################
#### main parameters ####
#########################
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet18", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
parser.add_argument("--global_pooling", default=True, type=bool_flag,
                    help="if True, we use the resnet50 global average pooling")
parser.add_argument("--use_bn", default=False, type=bool_flag,
                    help="optionally add a batchnorm layer before the linear classifier")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.3, type=float, help="initial learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--nesterov", default=False, type=bool_flag, help="nesterov momentum")
parser.add_argument("--scheduler_type", default="cosine", type=str, choices=["step", "cosine"])
# for multi-step learning rate decay
parser.add_argument("--decay_epochs", type=int, nargs="+", default=[60, 80],
                    help="Epochs at which to decay learning rate.")
parser.add_argument("--gamma", type=float, default=0.1, help="decay factor")
# for cosine learning rate schedule
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")


def main():
    global args, best_acc
    args = parser.parse_args()
    # init_distributed_mode(args)
    fix_random_seeds(args.seed)
    outer_logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
    )

    # 指定GPU https://www.cnblogs.com/ying-chease/p/9473938.html
    # gpu3可以指定0，1，2，3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # build data
    train_dataset = datasets.CIFAR10(args.data_path, train=True)
    val_dataset = datasets.CIFAR10(args.data_path, train=False)
    tr_normalize = transforms.Normalize(
        # mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
        # The CIFAR-10 mean below might lead to better results
        mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.262]
    )
    train_dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_dataset.transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        tr_normalize,
    ])
    # sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    # build model
    model = resnet_models.__dict__[args.arch](output_dim=0, eval_mode=True)
    linear_classifier = RegLog(1000, args.arch, args.global_pooling, args.use_bn)

    # convert batch norm layers (if any)
    # linear_classifier = nn.SyncBatchNorm.convert_sync_batchnorm(linear_classifier)

    # model to gpu
    model = model.cuda()
    linear_classifier = linear_classifier.cuda()
    # linear_classifier = nn.parallel.DistributedDataParallel(
    #     linear_classifier,
    #     device_ids=[args.gpu_to_work_on],
    #     find_unused_parameters=True,
    # )
    model.eval()

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr=args.lr,
        nesterov=args.nesterov,
        momentum=0.9,
        weight_decay=args.wd,
    )

    # set scheduler
    if args.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.decay_epochs, gamma=args.gamma
        )
    elif args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.final_lr
        )

    # 获取所有检查点
    checkpoints = []
    if os.path.isdir(args.pretrained):
        files = os.listdir(args.pretrained)
        for it in files:
            checkpoint = os.path.join(args.pretrained, it)
            if os.path.isfile(checkpoint):
                checkpoints.append(checkpoint)
    elif os.path.isfile(args.pretrained):
        checkpoints.append(args.pretrained)

    origin_dump_path = args.dump_path
    for checkpoint in checkpoints:
        # 为每个检查点创建存放文件夹
        dump_path = os.path.join(origin_dump_path, os.path.splitext(os.path.basename(checkpoint))[0])
        if os.path.isdir(dump_path):
            outer_logger.info('checkpoint dir:' + str(dump_path) + " already exist")
            continue
        else:
            os.mkdir(dump_path)
        args.dump_path = dump_path
        logger, training_stats = initialize_exp(
            args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
        )
        # load weights
        if os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint, map_location="cuda:0")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # remove prefixe "module."
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            for k, v in model.state_dict().items():
                if k not in list(state_dict):
                    logger.info('key "{}" could not be found in provided state dict'.format(k))
                elif state_dict[k].shape != v.shape:
                    logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                    state_dict[k] = v
            msg = model.load_state_dict(state_dict, strict=False)
            logger.info("Load pretrained model {0} with msg: {1}".format(checkpoint, msg))
            logger.info("")
        else:
            logger.info("No pretrained weights found => training with random weights")
            continue
        start_epoch = 0
        best_acc = 0
        cudnn.benchmark = True
        logger.info("Start training")
        for epoch in range(start_epoch, args.epochs):

            # train the network for one epoch

            # set samplers
            # train_loader.sampler.set_epoch(epoch)

            scores = train(model, linear_classifier, optimizer, train_loader, epoch)
            scores_val = validate_network(val_loader, model, linear_classifier)
            training_stats.update(scores + scores_val)

            scheduler.step()

            # save checkpoint
            if (epoch + 1) % args.checkpoint_freq == 0 and args.rank == 0:
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": best_acc,
                }
                torch.save(save_dict, os.path.join(args.dump_path, "eval_" + os.path.basename(checkpoint)))


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels, arch="resnet18", global_avg=False, use_bn=True):
        super(RegLog, self).__init__()
        self.bn = None
        if global_avg:
            if arch == "resnet50":
                s = 2048
            elif arch == "resnet50w2":
                s = 4096
            elif arch == "resnet50w4":
                s = 8192
            elif arch == "resnet18":
                s = 512
            self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            assert arch == "resnet50"
            s = 8192
            self.av_pool = nn.AvgPool2d(6, stride=1)
            if use_bn:
                self.bn = nn.BatchNorm2d(2048)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # average pool the final feature map
        x = self.av_pool(x)

        # optional BN
        if self.bn is not None:
            x = self.bn(x)

        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def train(model, reglog, optimizer, loader, epoch):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()

    model.eval()
    reglog.train()
    criterion = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(inp)
        output = reglog(output)

        # compute cross entropy loss
        loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # update stats
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1[0], inp.size(0))
        top5.update(acc5[0], inp.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

    # verbose
    if args.rank == 0 and epoch % 50 == 0:
        logger.info(
            "Epoch[{0}]\t"
            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
            "LR {lr}".format(
                epoch,
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                lr=optimizer.param_groups[0]["lr"],
            )
        )

    return epoch, losses.avg, top1.avg.item(), top5.avg.item()


def validate_network(val_loader, model, linear_classifier):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global best_acc

    # switch to evaluate mode
    model.eval()
    linear_classifier.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(val_loader):
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = linear_classifier(model(inp))
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    if top1.avg.item() > best_acc:
        best_acc = top1.avg.item()

    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, acc=best_acc))

    return losses.avg, top1.avg.item(), top5.avg.item()


if __name__ == "__main__":
    main()
