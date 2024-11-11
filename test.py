from __future__ import print_function
import os
import argparse
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Net
from dataset import CustomDataset
from loss import DiceChannelLoss


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Breast cancer Segmentation')
    parser.add_argument('--data_direc', type=str,default='./data', help="data directory")
    parser.add_argument('--n_classes', type=int,default=1, help="num of classes")
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=123')
    parser.add_argument('--testBatchSize', type=int, default=4, help='test batch size')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='Path for save best model')
    opt = parser.parse_args()
    
    if not os.path.isdir(opt.model_save_path):
        raise Exception("checkpoints not found, please run train.py first")

    os.makedirs("test_results",exist_ok=True)
    
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
    
    test_set = CustomDataset(f"{opt.data_direc}/test",mode='eval')
    test_dataloader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
    
    print('===> Building model')
    model = Net(n_classes=opt.n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.model_save_path,'model_statedict.pth'),map_location=device))
    model.eval()

    with open(os.path.join(opt.model_save_path,'metric_logger.json'), 'r') as f:
        metric_logger = json.load(f)
    
    criterion = nn.BCELoss()
    criterion_dice =DiceChannelLoss()
    
    
    total_test_num = len(test_dataloader.sampler)
    test_dice,test_ce =torch.zeros(opt.n_classes),0
    
    with torch.no_grad():
        for data in tqdm(test_dataloader,total=len(test_dataloader),position=0,desc='Test',colour='green'):
            batch_num = len(data['input'])
        
            image = data['input'].to(device)
            target = data['target'].to(device).unsqueeze(1)

            pred_logit = model(image.float())
            pred = (pred_logit > 0.5).float()
            ce_loss = criterion(pred,target.float())
            dice_channel_loss, dice_loss = criterion_dice(pred,target)
            
            test_dice+=dice_channel_loss.cpu()*batch_num
            test_ce+=ce_loss.item()*batch_num
    
    test_dice/=total_test_num
    test_ce/=total_test_num


    eval_df = pd.DataFrame({"Test Dice Coefficient Score":[test_dice.numpy()]})

    eval_df.to_csv(f"test_results/metric_df.csv",index=None)

    plt.figure()
    for k in ['train_dice','val_dice']:
        plt.plot(np.arange(len(metric_logger[k])),metric_logger[k],label=k)
    plt.title("Dice Coefficient Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"test_results/learning_graph_dice_coefficient.png",dpi=200)
    
    
    plt.figure()
    for k in ['train_ce','val_ce']:
        plt.plot(np.arange(len(metric_logger[k])),metric_logger[k],label=k)
    plt.title("Binary Cross Entropy Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"test_results/learning_graph_binary_cross_enropy.png",dpi=200)
