import os
import sys
import time
import re
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from basedataset import MyDataset
from model import *
from utils import save_checkpoint, AverageMeter, calc_topk_accuracy

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms


parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet50', type=str)
parser.add_argument('--seq_len', default=16, type=int, help='number of frames in each video block')
parser.add_argument('--num_dis', default=7, type=int, help='number of distortion types')
parser.add_argument('--pred_step', default=1, type=int)
parser.add_argument('--temperature', default=0.2, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--seed', default=233, type=int)
parser.add_argument('--crop_size', default=224, type=int)
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--SOURCE_TXT_DIR', type=str)
parser.add_argument('--SOURCE_FRAME_DIR', type=str)
parser.add_argument('--VALIDATE_TXT_DIR', type=str)
parser.add_argument('--VALIDATE_FRAME_DIR', type=str)
parser.add_argument('--model_path', type=str)


def main():
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = Model(num_seq=args.num_seq,
                  seq_len=args.seq_len,
                  pred_step=args.pred_step,
                  T=args.temperature,
                  network=args.net
                  )

    model = model.to(cuda)
    global criterion; criterion = nn.CrossEntropyLoss()

    if args.train_what == 'last':
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False
    else: pass # train all layers

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    args.old_lr = None

    best_acc = 0
    global iteration; iteration = 0

    train_data = MyDataset(txt_dir=args.SOURCE_TXT_DIR,
                           root=args.SOURCE_FRAME_DIR,
                           num_types=args.num_dis,
                           num_frames=args.seq_len,
                           crop_size=args.crop_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True)

    val_data = MyDataset(txt_dir=args.VALIDATE_TXT_DIR,
                         root=args.VALIDATE_FRAME_DIR,
                         num_types=args.num_dis,
                         num_frames=args.seq_len,
                         crop_size=args.crop_size)

    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=True)

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc, train_accuracy_list = train(train_loader, model, optimizer, epoch)
        val_loss, val_acc, val_accuracy_list = validate(val_loader, model, epoch)

        is_best = val_acc > best_acc; best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch+1,
                         'net': args.net,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'iteration': iteration},
                         is_best, filename=os.path.join(args.model_path, 'epoch%s.pth.tar' % str(epoch+1)), keep_all=False)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    (B1, N1, PR, B2, N2, SL) = mask.size() # [B, N, pred, B, N, SL]
    target = mask == 1
    target.requires_grad = False
    return target, (B1, N1, PR, B2, N2, SL)


def train(data_loader, model, optimizer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.train()
    global iteration

    for idx, input_seq in enumerate(data_loader):
        tic = time.time()
        input_seq = input_seq.to(cuda)
        # B = input_seq.size(0)
        [score_, mask_] = model(input_seq)

        del input_seq

        if idx == 0: target_, (B1, N1, PR, B2, N2, SL) = process_output(mask_)

        score_flattened = score_.view(B1 * N1 * PR, B2 * N2 * SL)
        target_flattened = target_.view(B1 * N1 * PR, B2 * N2 * SL).to(cuda)
        target_flattened = target_flattened.to(int).argmax(dim=1)

        loss = criterion(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

        accuracy_list[0].update(top1.item(), B1)
        accuracy_list[1].update(top3.item(), B1)
        accuracy_list[2].update(top5.item(), B1)

        losses.update(loss.item(), B1)
        accuracy.update(top1.item(), B1)

        del score_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                  'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f}\t'.format(
                epoch, idx, len(data_loader), top1, top3, top5, time.time() - tic, loss=losses))

            iteration += 1

    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def validate(data_loader, model, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():
        for idx, input_seq in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            # B = input_seq.size(0)
            [score_, mask_] = model(input_seq)
            del input_seq

            if idx == 0: target_, (B1, N1, PR, B2, N2, SL) = process_output(mask_)

            score_flattened = score_.view(B1 * N1 * PR, B2 * N2 * SL)
            target_flattened = target_.view(B1 * N1 * PR, B2 * N2 * SL).to(cuda)
            target_flattened = target_flattened.to(int).argmax(dim=1)

            loss = criterion(score_flattened, target_flattened)
            top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

            losses.update(loss.item(), B1)
            accuracy.update(top1.item(), B1)

            accuracy_list[0].update(top1.item(), B1)
            accuracy_list[1].update(top3.item(), B1)
            accuracy_list[2].update(top5.item(), B1)

    print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
          'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} \t'.format(
        epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses))
    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


if __name__ == '__main__':
    main()




