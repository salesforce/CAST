'''
python main_cast.py   -a resnet50 --cos  --lr 0.5   --batch-size 256   --dist-url 'tcp://localhost:10005' <ImageFolder> --mask-dir <MaskFolder> --crit-gcam cosine --alpha-masked 3 --second-constraint "ref" --output-mask-region "ref" --num-gpus-per-machine 8  --print-freq 10 --workers 8'''

import argparse
import math
import os
import os.path as osp
import random
import shutil
import time
import warnings
import matplotlib.cm as cm
import copy
import sys
import subprocess
import psutil

import albumentations as alb
import scipy
import cv2
from loguru import logger
import numpy as np
from PIL import Image, ImageDraw


import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler#, autocast
from apex.parallel import DistributedDataParallel as ApexDDP
from grad_cam import GradCAM


import moco.builder

from albumentations.pytorch.transforms import ToTensorV2
from moco.datasets import SaliencyConstrainedRandomCropping
from moco.utils.checkpointing import CheckpointManager
import moco.utils.distributed as dist

import pdb
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

# fmt: off
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR",
                    help="path to serialized LMDB file")
parser.add_argument('--mask-dir', default='', type=str, metavar='PATH',
                    help='path where masks are available')
parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet50",
                    choices=model_names,
                    help="model architecture: " +
                        " | ".join(model_names) +
                        " (default: resnet50)")
parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                    help="number of data loading workers per GPU (default: 4)")
parser.add_argument("--epochs", default=200, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=256, type=int,
                    metavar="N",
                    help="mini-batch size (default: 256), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--lr", "--learning-rate", default=0.03, type=float,
                    metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--schedule", default=[120, 160], nargs="*", type=int,
                    help="learning rate schedule (when to drop lr by 10x)")
parser.add_argument("--lr-cont-schedule-start", default=120, type=int,
                    help="continual learning rate decay schedule (when to start dropping lr by 0.94267)")
parser.add_argument("--lr-cont-decay", action="store_true",
                    help="True if you want to continuously decay learning rate")

parser.add_argument("--momentum", default=0.9, type=float, metavar="M",
                    help="momentum of SGD solver")
parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float,
                    metavar="W", help="weight decay (default: 1e-4)",
                    dest="weight_decay")
parser.add_argument("-p", "--print-freq", default=10, type=int,
                    metavar="N", help="print frequency (default: 10)")

parser.add_argument("--resume", default="", type=str, metavar="PATH",
                    help="path to latest checkpoint (default: none)")

parser.add_argument("--min-areacover", default=0.2, type=float, help="min area cover")

parser.add_argument("--second-constraint", default="ref", type=str,
                    help="Second constraint possible values ['all', 'ref']")
parser.add_argument("--output-mask-region", default="ref", type=str,
                    help="output mask region possible values ['all', 'ref']")

parser.add_argument("--layer_name", default="layer4", type=str,
                    help="Which layer to compute gradcam")
parser.add_argument("--output-mask-size", default=7, type=int,
                    help="size of output_mask")

parser.add_argument("-e", "--same-encoder", dest="same_encoder", action="store_true",
                    help="compute gradcam on train set")

parser.add_argument("--alpha-masked", default=1, type=float,
                    help="gcam loss multiplier",
                    dest="alpha_masked")
parser.add_argument("--clip", default=2, type=float,
                    help="clip grad norm",
                    dest="clip")
parser.add_argument("--beta", default=1, type=float,
                    help="ssl loss multiplier",
                    dest="beta")
parser.add_argument("--crit-gcam", default="cosine", type=str,
                    help="criterion for gcam supervision [cosine]")         
  

# Distributed training arguments.
parser.add_argument(
    "--num-machines", type=int, default=1,
    help="Number of machines used in distributed training."
)
parser.add_argument(
    "--num-gpus-per-machine", type=int, default=8,
    help="""Number of GPUs per machine with IDs as (0, 1, 2 ...). Set as
    zero for single-process CPU training.""",
)
parser.add_argument(
    "--machine-rank", type=int, default=0,
    help="""Rank of the machine, integer in [0, num_machines). Default 0
    for training with a single machine.""",
)
parser.add_argument("--dist-url", default="tcp://localhost:10001", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--dist-backend", default="nccl", type=str,
                    help="distributed backend")
parser.add_argument("--seed", default=None, type=int,
                    help="seed for initializing training. ")

# moco specific configs:
parser.add_argument("--moco-dim", default=128, type=int,
                    help="feature dimension (default: 128)")
parser.add_argument("--moco-k", default=65536, type=int,
                    help="queue size; number of negative keys (default: 65536)")
parser.add_argument("--moco-m", default=0.999, type=float,
                    help="moco momentum of updating key encoder (default: 0.999)")
parser.add_argument("--moco-t", default=0.07, type=float,
                    help="softmax temperature (default: 0.07)")

# options for moco v2
parser.add_argument("--mlp", action="store_true",
                    help="use mlp head")

parser.add_argument("--cos", action="store_true",
                    help="use cosine lr schedule")

parser.add_argument(
    "--serialization-dir", default="save/test_exp",
    help="Path to a directory to serialize checkpoints and save job logs."
)
# fmt: on



def main(args: argparse.Namespace):
    # This method will only work for GPU training (single or multi).
    # Get the current device as set for current distributed process.
    # Check `launch` function in `moco.utils.distributed` module.
    device = torch.cuda.current_device()
    # pdb.set_trace()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    # Remove default logger, create a logger for each process which writes to a
    # separate log-file. This makes changes in global scope.
    logger.remove(0)
    if dist.get_world_size() > 1:
        logger.add(
            os.path.join(args.serialization_dir, f"log-rank{dist.get_rank()}.txt"),
            format="{time} {level} {message}",
        )

    # Add a logger for stdout only for the master process.
    if dist.is_master_process():
        logger.add(
            sys.stdout, format="<g>{time}</g>: <lvl>{message}</lvl>", colorize=True
        )

    logger.info(
        f"Current process: Rank {dist.get_rank()}, World size {dist.get_world_size()}"
    )
    # create model
    logger.info(f"=> creating model {args.arch}")
    logger.info(f"args.mlp:{args.mlp}")
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        args.mlp,
    ).to(device)
    args.batch_size = int(args.batch_size / dist.get_world_size())

    # define loss function (criterion) 
    criterion = nn.CrossEntropyLoss().to(device)

    # define loss function for the loss on gradcam
    if args.crit_gcam == "cosine":
        criterion_gcam = nn.CosineSimilarity(dim=1).to(device)
    else:
        raise NotImplementedError("Only cosine loss implemented.")        
        # criterion_gcam = nn.BCEWithLogitsLoss(reduction="mean").to(device)

    # define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


    # Create checkpoint manager and tensorboard writer.
    checkpoint_manager = CheckpointManager(
        serialization_dir=args.serialization_dir,
        filename_prefix=f"checkpoint_{args.arch}",
        state_dict=model,
        optimizer=optimizer,
    )
    tensorboard_writer = SummaryWriter(log_dir=args.serialization_dir)

    if dist.is_master_process():
        tensorboard_writer.add_text("args", f"```\n{vars(args)}\n```")

    # optionally resume from a checkpoint
    if args.resume:
        args.start_epoch = CheckpointManager(state_dict=model).load(args.resume)
        if args.same_encoder:
            # if you want to use the same weights for query encoder and key encoder
            model.encoder_k.load_state_dict(model.encoder_q.state_dict())

    cudnn.benchmark = True

    # Wrap model in ApexDDP if using more than one processes.
    if dist.get_world_size() > 1:
        dist.synchronize()    
        model = ApexDDP(model, delay_allreduce=True)
        
    DatasetClass = SaliencyConstrainedRandomCropping  
    train_dataset = DatasetClass(args.data,  args.mask_dir, 'train2017', args.second_constraint,  args.output_mask_region,  args.output_mask_size, args.min_areacover)
    val_dataset = DatasetClass(args.data,  args.mask_dir, 'val2017', args.second_constraint,  args.output_mask_region,  args.output_mask_size, args.min_areacover)
    
    if dist.get_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.sampler.SequentialSampler(train_dataset)
        val_sampler = torch.utils.data.sampler.SequentialSampler(val_dataset)
    
    # create train and val dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=(val_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True,
    )


    # fmt: off
    if dist.get_world_size() > 1:
        candidate_layers_q = ["module.encoder_q." + args.layer_name]
    else:
        candidate_layers_q = ["encoder_q." + args.layer_name]

    model.train()

    # define instance of gradcam applied on query
    gcam_q = GradCAM(model=model, candidate_layers=candidate_layers_q)
    
    # define instance of ChecckpointManager to save checkpoints after every epoch
    checkpoint_manager = CheckpointManager(
        serialization_dir=args.serialization_dir,
        filename_prefix=f"checkpoint_{args.arch}",
        state_dict=gcam_q.model,
        optimizer=optimizer,
        )

    # start training
    for epoch in range(args.start_epoch, args.epochs):
        if dist.get_world_size() > 1:
            train_sampler.set_epoch(epoch)
        
        # at the start of every epoch, adjust the learning rate
        lr = adjust_learning_rate(optimizer, epoch, args)
        
        logger.info("Current learning rate is {}".format(lr))
        
        # train for one epoch
        
        CAST(
            train_loader, gcam_q, model, criterion, criterion_gcam,
            optimizer, tensorboard_writer, epoch, device, args,
        )

        if dist.is_master_process():
            checkpoint_manager.step(epoch=epoch + 1)


# fmt: off
def train(
    train_loader, model, criterion, optimizer,
    tensorboard_writer, epoch, device, args
):
# fmt: on
    batch_time_meter = AverageMeter("Time", ":6.3f")
    data_time_meter = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time_meter, data_time_meter, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    start_time = time.perf_counter()
    for i, (images, _) in enumerate(train_loader):
        data_time = time.perf_counter() - start_time

        images[0] = images[0].to(device, non_blocking=True)
        images[1] = images[1].to(device, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
 
        # compute gradient and do SGD step
        optimizer.zero_grad()

        # Perform dynamic scaling of loss to adjust for mixed precision.
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time = time.perf_counter() - start_time

        # update all progress meters
        data_time_meter.update(data_time)
        batch_time_meter.update(batch_time)
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        if dist.is_master_process():
            tensorboard_writer.add_scalars(
                "pretrain",
                {
                    "data_time": data_time,
                    "batch_time": batch_time,
                    "loss": loss,
                    "acc1": acc1,
                    "acc5": acc5,
                },
                epoch * len(train_loader) + i,
            )

        if i % args.print_freq == 0:
            progress.display(i)

        start_time = time.perf_counter()

# fmt: off
def CAST(
    train_loader, gcam_q, model, criterion, criterion_gcam, optimizer,
    tensorboard_writer, epoch, device, args
):
# fmt: on
    # define progress meters for measuring time, losses and accuracies
    batch_time_meter = AverageMeter("Time", ":6.3f")
    data_time_meter = AverageMeter("Data", ":6.3f")
    losses_total = AverageMeter("Loss_total", ":.4e")
    losses_ssl = AverageMeter("Loss_ssl", ":.4e")
    losses_gcam_masked = AverageMeter("Loss_gcam_masked", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time_meter, data_time_meter, losses_total, losses_ssl, losses_gcam_masked, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    if dist.get_world_size() > 1:
        target_layer_q = "module.encoder_q." + args.layer_name
    else:
        # single gpu
        target_layer_q = "encoder_q." + args.layer_name

    start_time = time.perf_counter()
    for i, (images, paths, query_masks, masked_keys) in enumerate(train_loader):

        data_time = time.perf_counter() - start_time
        images[0] = images[0].to(device, non_blocking=True)
        images[1] = images[1].to(device, non_blocking=True)
        key_masked = masked_keys.to(device, non_blocking=True)
        query_masks = query_masks.to(device=device, dtype=images[0].dtype, non_blocking=True)
        

        # forward query and key to obtain logits
        output, target = gcam_q.forward( images[0], images[1], add_to_queue=True)

        # forward query and masked key to obtain masked logits
        output_masked, target_masked = gcam_q.forward(images[0], key_masked, add_to_queue=False)

        # compute gradients of the dot-product wrt query convolutional activations
        grad_wrt_act_masked = gcam_q.backward(
            torch.zeros(0, dtype=images[0].dtype, device=device),
            target_layer=target_layer_q
        )
        
        # combine gradients with forward query activation maps to get gradcam
        gcam_masked = gcam_q.generate(
            target_layer=target_layer_q, grads=grad_wrt_act_masked
        )
        
        # Regular Contrastive loss 
        loss_ssl = criterion(output, target)

        # Attention loss on Grad-CAM with gt saliency masks wrt query
        if args.crit_gcam == "cosine":
            loss_gcam_masked = 1-criterion_gcam(gcam_masked.view(1,-1), query_masks.view(1,-1))
        else:
            raise NotImplementedError("Only cosine loss supported")
        
        # total loss        
        loss = (args.beta/(args.beta+args.alpha_masked)) * loss_ssl + (args.alpha_masked/(args.beta+args.alpha_masked)) * loss_gcam_masked

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        # zero grad before gradient computation on nodes
        optimizer.zero_grad()
        gcam_q.model.zero_grad()

        # compute gradient and do SGD step
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(gcam_q.model.parameters(), args.clip)

        # Update model parameters
        optimizer.step()

        # measure elapsed time
        batch_time = time.perf_counter() - start_time

        # update all progress meters
        data_time_meter.update(data_time)
        batch_time_meter.update(batch_time)
        losses_total.update(loss.item(), images[0].size(0))
        losses_ssl.update(loss_ssl.item(), images[0].size(0))
        losses_gcam_masked.update(loss_gcam_masked.item(), images[0].size(0))

        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # log to tensorboard
        if dist.is_master_process():
            tensorboard_writer.add_scalar("data_time", data_time, epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar("batch_time", batch_time, epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar("loss_total", loss.detach(), epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar("loss_ssl", loss_ssl.detach(), epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar("loss_gcam_masked", loss_gcam_masked.detach(), epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar("acc1", acc1, epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar("acc5", acc5, epoch * len(train_loader) + i)

            
        if i % args.print_freq == 0:
            progress.display(i)

        start_time = time.perf_counter()
        

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":

    _A = parser.parse_args()
    os.makedirs(_A.serialization_dir, exist_ok=True)    

    _A.serialization_dir = os.path.abspath(_A.serialization_dir)

    if _A.num_gpus_per_machine == 0:
        raise NotImplementedError("Training on CPU is not supported.")        
    else:
        # This will launch `main` and set appropriate CUDA device (GPU ID) as
        # per process (accessed in the beginning of `main`).
        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus_per_machine,
            machine_rank=_A.machine_rank,
            dist_url=_A.dist_url,
            dist_backend=_A.dist_backend,
            args=(_A, ),
        )
