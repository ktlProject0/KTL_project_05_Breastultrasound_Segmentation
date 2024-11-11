from __future__ import print_function
import os
import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from dataset import CustomDataset
from util import EarlyStopping
from loss import DiceChannelLoss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Breast cancer Segmentation')
    parser.add_argument('--data_direc', type=str,default='./data', help="data directory")
    parser.add_argument('--n_classes', type=int,default=1, help="num of classes")
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--total_epoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.001')
    parser.add_argument('--lr_schedule_patience', type=float, default=10, help='Learning Rate schedule patience. Default=10')
    parser.add_argument('--earlystop_patience', type=float, default=20, help='Earlystop_patience. Default=20')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=123')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='Path for save best model')
    opt = parser.parse_args()
    
    os.makedirs(opt.model_save_path,exist_ok = True)
    
    print(opt)
    
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    torch.manual_seed(opt.seed)
    
    if opt.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device ='cuda'
    print('===> Loading datasets')
    
    train_set = CustomDataset(f"{opt.data_direc}/train",mode='train')
    test_set = CustomDataset(f"{opt.data_direc}/val",mode='eval')
    train_dataloader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    val_dataloader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
    
    print('===> Building model')
    model = Net(n_classes=opt.n_classes).to(device)
    criterion = nn.BCELoss()
    criterion_dice =DiceChannelLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=opt.lr_schedule_patience)
    monitor = EarlyStopping(patience=opt.earlystop_patience, verbose=True, path=os.path.join(opt.model_save_path,'model_statedict.pth'))
    
    metric_logger = {k:[] for k in ['train_ce','val_ce',
                                'train_dice','val_dice',
                                'train_loss','val_loss',
                                'train_dice_per_channel', 'val_dice_per_channel',
                                'lr']}
    total_train_num = len(train_dataloader.sampler)
    total_val_num = len(val_dataloader.sampler)
    
    
    for epoch in range(opt.total_epoch):
        
        for param in optimizer.param_groups:
            lr_stauts = param['lr']
        metric_logger['lr'].append(lr_stauts)
    
        epoch_loss = {k:0 for k in metric_logger if k not in ['lr']}
        epoch_loss['train_dice_per_channel'] = torch.zeros(opt.n_classes)
        epoch_loss['val_dice_per_channel'] = torch.zeros(opt.n_classes)
        
        print(f"Epoch {epoch+1:03d}/{opt.total_epoch:03d}\tLR: {lr_stauts:.0e}")
        
        model.train()
        for data in tqdm(train_dataloader,total=len(train_dataloader),position=0,desc='Train',colour='blue'):
            batch_num = len(data['input'])
            
            image = data['input'].to(device)
            target = data['target'].to(device).unsqueeze(1)
            
            pred = model(image.float())
            ce_loss = criterion(pred,target.float())
            dice_loss_per_channel, dice_loss = criterion_dice(pred,target)
            
            loss = ce_loss + dice_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss['train_ce'] += ce_loss.item()*batch_num
            epoch_loss['train_dice'] += dice_loss.item()*batch_num
            epoch_loss['train_loss'] += loss.item()*batch_num
            epoch_loss['train_dice_per_channel'] += dice_loss_per_channel.cpu()*batch_num
            
            
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_dataloader,total=len(val_dataloader),position=0,desc='Val',colour='green'):
                batch_num = len(data['input'])
            
                image = data['input'].to(device)
                target = data['target'].to(device).unsqueeze(1)
    
                pred = model(image.float())
                ce_loss = criterion(pred,target.float())
                dice_loss_per_channel, dice_loss = criterion_dice(pred,target)
                loss = ce_loss + dice_loss
            
                epoch_loss['val_ce'] += ce_loss.item()*batch_num
                epoch_loss['val_dice'] += dice_loss.item()*batch_num
                epoch_loss['val_loss'] += loss.item()*batch_num
                epoch_loss['val_dice_per_channel'] += dice_loss_per_channel.cpu()*batch_num
    
    
        
        epoch_loss = {k:(v/total_train_num if 'train' in k else v/total_val_num) for k,v in epoch_loss.items()}
        
        for k,v in epoch_loss.items():
            if '_per_channel' in k:
                v=v.detach().numpy().tolist()
            metric_logger[k].append(v)
    
        monitor(epoch_loss['val_loss'],model)
        if monitor.early_stop:
            print(f"Train early stopped, Minimum validation loss: {monitor.val_loss_min}")
            break
        
        scheduler.step(epoch_loss['val_loss'])        
        
        print(f"Train loss: {epoch_loss['train_loss']:.7f}\tTrain ce: {epoch_loss['train_ce']:.7f}\tTrain dice: {epoch_loss['train_dice']:.7f}\n\
    Val loss: {epoch_loss['val_loss']:.7f}\tVal ce: {epoch_loss['val_ce']:.7f}\tVal dice: {epoch_loss['val_dice']:.7f}")
    
        formatted_list = ['{:4d}'.format(num) for num in [0,1]]
        print("Class\n",', '.join(formatted_list))
    
        formatted_list = ['{:.2f}'.format(num) for num in epoch_loss['train_dice_per_channel']]
        print("Train dice loss per channel\n",', '.join(formatted_list))
    
        formatted_list = ['{:.2f}'.format(num) for num in epoch_loss['val_dice_per_channel']]
        print("Val dice loss per channel\n",', '.join(formatted_list))
    
    
        with open(os.path.join(opt.model_save_path,'metric_logger.json'),'w') as f:
            json.dump(metric_logger, f)