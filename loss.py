import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceChannelLoss(nn.Module):
    def __init__(self):
        super(DiceChannelLoss, self).__init__()

    def forward(self, pred, target, smooth=1e-9, weights_apply=False):
        pred = torch.sigmoid(pred)

        if target.dim() == 3:
            target = target.unsqueeze(1)

        num_channels = pred.shape[1]
        dice = torch.zeros(num_channels, device=pred.device)
        
        for i in range(num_channels):
            pred_channel = pred[:, i]
            target_channel = target[:, i]
            
            intersection = (pred_channel * target_channel).sum()
            dice_coeff = (2. * intersection + smooth) / (pred_channel.sum() + target_channel.sum() + smooth)

            dice[i] = 1 - dice_coeff.item()

        if weights_apply:
            weights = (dice / torch.sum(dice))
            dice = dice * weights.to(pred.device)

        dice_loss = dice.sum()
        
        del pred, pred_channel, target_channel, intersection, dice_coeff
        torch.cuda.empty_cache()
        
        return dice, dice_loss