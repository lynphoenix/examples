import argparse
import os
import random
import shutil
import time
import warnings
import sys

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def argparse():
    parser = argparse.ArgumentParser(description='PyTorch Jit trace test')
    parser.add_argument('--pytorch-model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--onnx-model', default='model.onnx', type=str,
                        help='path to save onnx')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    args = parser.parse_args()
    return args


def create_model(args):
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    return model


def load_model(args, model):
    if os.path.isfile(args.pytorch_model):
        print("=> loading checkpoint '{}'".format(args.pytorch_model))
        checkpoint = torch.load(args.pytorch_model)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.pytorch_model, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.pytorch_model))


def convert_onnx(model, args):
    example = torch.rand(1, 3, 224, 224).cuda()
    traced_script_module = torch.jit.trace(model, example)
    # traced_script_module.save(save_path)
    torch_out = torch.onnx._export(model,  # model being run
                               example,  # model input (or a tuple for multiple inputs)
                               args.onnx_model,  # where to save the model
                               export_params=True)
    

if __name__ == '__main__':
    args = argparse()
    model = create_model(args)
    model.cuda()
    model.eval()
    convert_onnx(args, model)
