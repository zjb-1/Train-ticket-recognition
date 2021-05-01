import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from model_MobileNetV2 import EAST_MobileV2
from model_SENet import EAST_SENet
from loss import Loss
import os
import time
import argparse
import numpy as np
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(
    description='EAST Detector Training With Pytorch')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('-s', '--start_epoch', default=0, type=int,
                    help='Resume training at this epoch')
parser.add_argument('--min_len', default=800, type=int,
                    help='resize the smallest edge of the image to min_len')
parser.add_argument('--crop_size', default=512, type=int,
                    help='random crop image to cnn')
args = parser.parse_args()


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, writer):
    file_num = len(os.listdir(train_img_path))
    trainset = custom_dataset(train_img_path, train_gt_path, args.min_len, args.crop_size)

    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers= num_workers, drop_last=True)
    criterion = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # if args.resume is None:                 ### 从头开始训练
    #     model = EAST_MobileV2(args.crop_size, True)
    # else:                                   ### 从checkpoint 处恢复训练
    #     model = EAST_MobileV2(args.crop_size, False)
    #     print('Resuming training, loading {}...'.format(args.resume))
    #     model.load_state_dict(torch.load(args.resume))
    model = EAST_SENet()

    data_parallel = False
    #if torch.cuda.device_count() > 10:
        #model = nn.DataParallel(model, device_ids=[0, 1])
        #data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)

    for epoch in range(args.start_epoch, epoch_iter):
        model.train()
        scheduler.step()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

            epoch_loss += float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps = epoch * len(train_loader) + i
            writer.add_scalar('loss', loss, steps)

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
                epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))

        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('='*50)
        if (epoch + 1) % interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(pths_path, 'model_SE_epoch_{}.pth'.format(epoch+1)))

    writer.close()


if __name__ == '__main__':
    train_img_path = "/home/data/fapiao_data_new/train_images/"
    train_gt_path  = "/home/data/fapiao_data_new/train_gts/"
    pths_path      = './pths'
    batch_size     = 4
    lr             = 1e-3
    num_workers    = 4
    epoch_iter     = 300
    save_interval  = 5
    writer = SummaryWriter()
    train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval, writer)





