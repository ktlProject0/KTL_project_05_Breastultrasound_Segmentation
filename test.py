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
from sklearn.metrics import precision_score, recall_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Breast Cancer Segmentation')
    parser.add_argument('--data_direc', type=str, default='./data', help="Data directory")
    parser.add_argument('--n_classes', type=int, default=1, help="Number of classes")
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--testBatchSize', type=int, default=4, help='Test batch size')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='Path for saved model')
    opt = parser.parse_args()

    if not os.path.isdir(opt.model_save_path):
        raise Exception("Checkpoints not found, please run train.py first")

    os.makedirs("test_results", exist_ok=True)

    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

    print('===> Loading datasets')
    test_set = CustomDataset(f"{opt.data_direc}/test", mode='eval')
    test_dataloader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    print('===> Building model')
    model = Net(n_classes=opt.n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.model_save_path, 'model_statedict.pth'), map_location=device))
    model.eval()

    with open(os.path.join(opt.model_save_path, 'metric_logger.json'), 'r') as f:
        metric_logger = json.load(f)

    criterion = nn.BCELoss()
    criterion_dice = DiceChannelLoss()

    all_dice_scores = []
    all_precisions = []
    all_recalls = []

    with torch.no_grad():
        for data in tqdm(test_dataloader, total=len(test_dataloader), position=0, desc='Test', colour='green'):
            image = data['input'].to(device)
            target = data['target'].to(device).unsqueeze(1)

            pred_logit = model(image.float())
            pred = (pred_logit > 0.5).float()

            dice_channel_loss, _ = criterion_dice(pred, target)
            dice_score = 1 - dice_channel_loss.cpu().numpy()  # Dice score = 1 - Dice loss
            all_dice_scores.append(dice_score)

            pred_flat = pred.cpu().numpy().flatten()
            target_flat = target.cpu().numpy().flatten()
            precision = precision_score(target_flat, pred_flat, average='binary', zero_division=1)
            recall = recall_score(target_flat, pred_flat, average='binary', zero_division=1)
            all_precisions.append(precision)
            all_recalls.append(recall)

    # Compute metrics
    avg_dice = np.mean(all_dice_scores)
    std_dice = np.std(all_dice_scores)
    avg_precision = np.mean(all_precisions)
    std_precision = np.std(all_precisions)
    avg_recall = np.mean(all_recalls)
    std_recall = np.std(all_recalls)

    # Save metrics
    eval_df = pd.DataFrame({
        "Test Dice Coefficient Score": [avg_dice],
        "Dice Std": [std_dice],
        "Test Precision": [avg_precision],
        "Precision Std": [std_precision],
        "Test Recall": [avg_recall],
        "Recall Std": [std_recall],
    })
    eval_df.to_csv("test_results/metric_df.csv", index=None)

    plt.figure()
    for k in ['train_dice', 'val_dice']:
        plt.plot(np.arange(len(metric_logger[k])), metric_logger[k], label=k)
    plt.title("Dice Coefficient Loss")
    plt.legend()
    plt.grid()
    plt.savefig("test_results/learning_graph_dice_coefficient.png", dpi=200)
