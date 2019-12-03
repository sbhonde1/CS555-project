#!/usr/bin/env python
# coding: utf-8
# https://github.com/pytorch/examples/blob/91f230a13b95c6259e5cb22b6cef355de998cede/imagenet/main.py
"""
python train.py \
        --batch-size 6 \
        --n-epochs 10 \
        --jaccard-weight 0.3 \
"""

from torch.utils.data import DataLoader
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from engine_utils import utils as t_util
from engine_utils.engine import train_one_epoch, evaluate, train_one_epoch_distributed
from engine_utils.coco_utils import *
from dataset import *
from utils import *

MODE_TRAIN = 'train'
MODE_TEST = 'test'
MODE_VALID = 'valid'
cudnn.benchmark = True


def distributed():
    args = parse()
    num_workers = 1
    # 2 classes: mug and background
    num_classes = 2
    args.gpu = 0
    args.world_size = 1
    args.distributed = False
    # pytorch creates 'WORLD_SIZE' environment when you launch a script with torch.distributed.launch module
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.distributed:
        # Multiple/Single GPUs and Machines
        args.gpu = args.local_rank
        setup(args.gpu)
        args.world_size = torch.distributed.get_world_size()

    # setup devices for this process, rank 1 uses GPUs [0] and rank 2 uses GPUs [1].
    n = torch.cuda.device_count() // args.world_size
    device_ids = list(range(args.gpu * n, (args.gpu + 1) * n))
    # Create a model
    model = get_model(num_classes)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # Initialize model save path only once
    # dt = datetime.now().isoformat() # dt doesn't work with distributed training as we spawn n processes
    common_name = 'model_{}_{}'.format(model.__class__.__name__, args.save_as)
    model_path = Path(args.data).joinpath('models', common_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Initialize Tensorboard log
    tb_writer = None
    if args.gpu == 0:
        tb_writer = initialize_tensorboard(args.log_dir, common_name)

    if args.distributed:
        # delay_allreduce delays all communication to the end of the backward pass.
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids, output_device=args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
    else:
        # One machine & one GPU
        model = model.cuda()
    # Create dataset
    if torch.distributed.get_rank() == 0:
        print("Preparing data...")
    dataset_train = MugDataset(args.data, MODE_TRAIN,
                               transform=get_transform(train=True))
    dataset_test = MugDataset(args.data, MODE_TEST, transform=get_transform(train=False))
    dataset_valid = MugDataset(args.data, MODE_VALID, transform=get_transform(train=False))
    train_sampler = None
    test_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_valid)

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=t_util.collate_fn, sampler=train_sampler)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                             collate_fn=t_util.collate_fn, sampler=test_sampler)
    loader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=t_util.collate_fn, sampler=val_sampler)
    if torch.distributed.get_rank() == 0:
        print("Data preparation complete")
    # let's train it for 10 epochs
    num_epochs = 20

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # train_one_epoch_distributed(model, optimizer, data_loader, sampler, epoch, args)
        train_loss = train_one_epoch_distributed(model, optimizer, loader_train, train_sampler,  epoch, args)
        # update the learning rate
        # lr_scheduler.step()
        if torch.distributed.get_rank() == 0:
            update_train_loss(tb_writer, train_loss, epoch)
            # evaluate on the test dataset
            evaluate(model, loader_valid, torch.device('cuda:0'), epoch, tb_writer)




def single_gpu():
    args = parse()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 2 classes: mug and background
    num_classes = 2
    # Create dataset
    dataset_train = MugDataset(args.data, MODE_TRAIN,
                               transform=get_transform(train=True))
    dataset_test = MugDataset(args.data, MODE_TEST, transform=get_transform(train=False))
    dataset_valid = MugDataset(args.data, MODE_VALID, transform=get_transform(train=False))

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=t_util.collate_fn)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=4,
                             collate_fn=t_util.collate_fn)
    loader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=t_util.collate_fn)

    model = get_model(num_classes)

    # move model to the right device
    model.to(device)
    # Initialize model save path only once
    # dt = datetime.now().isoformat() # dt doesn't work with distributed training as we spawn n processes
    common_name = 'model_{}_{}'.format(model.__class__.__name__, args.save_as)
    model_path = Path(args.data).joinpath('models', common_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Initialize Tensorboard log
    tb_writer = initialize_tensorboard(args.log_dir, common_name)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 20

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_loss = train_one_epoch(model, optimizer, loader_train, device, epoch, print_freq=10)
        update_train_loss(tb_writer, train_loss, epoch)
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, loader_valid, device, epoch, tb_writer)


if __name__ == "__main__":
    single_gpu()

    # # get the model using our helper function

