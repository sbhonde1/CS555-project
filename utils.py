import argparse
import torchvision.transforms as T
import torchvision
import torch.distributed as dist
import torch
from pathlib import Path
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def get_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = num_classes  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def setup(rank):
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, init_method='env://', world_size=4)
    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)
    torch.cuda.set_device(rank)


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def initialize_tensorboard(log_dir, common_name):
    """
    In distributed training, tensorboard doesn't work with multiple writers
    reference: https://stackoverflow.com/a/37411400/4569025
    """
    tb_log_path = Path(log_dir).joinpath(common_name)
    if not os.path.exists(tb_log_path):
        os.mkdir(tb_log_path)
    tb_writer = SummaryWriter(log_dir=tb_log_path)
    return tb_writer


def update_train_loss(tb_writer, train_loss, epoch):
    tb_writer.add_scalar(tag='train loss', scalar_value=train_loss, global_step=epoch)


def update_prediction_image(tb_writer, box, image, score, epoch, i):
    tb_writer.add_image_with_boxes("Prediction {}".format(i), img_tensor=np.array(image), box_tensor=np.array(box),
                                   global_step=epoch,
                                   walltime=None, rescale=1, dataformats='CHW')
    tb_writer.add_scalar(tag='IoU score', scalar_value=score, global_step=epoch)


def parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data', metavar='DIR', help='path to dataset')
    arg('--n-epochs', type=int, default=100)
    arg('--batch-size', type=int, default=1)
    arg('--lr', type=float, default=0.0001)
    arg('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    arg('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    arg('--log-dir', metavar='DIR', help='path to save tensorboard logs', required=True)
    arg('--save-as', metavar='NAME', help='save model as', required=True)
    arg('--load-saved', metavar='DIR', help='path to load the previous saved model')

    # args taken for NVIDIA-APEX
    arg("--local_rank", default=0, type=int)
    arg('--opt-level', type=str)
    # arg('--keep-batchnorm-fp32', type=str, default=None)
    # arg('--sync_bn', action='store_true', help='enabling apex sync BN.')
    # arg('--loss-scale', type=str, default=None)
    args = parser.parse_args()
    return args
