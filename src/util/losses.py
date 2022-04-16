import torch
import torch.nn as nn
import torch.nn.functional as F

#Mean Correct Forecast Direction
class MCFD(nn.Module):
    def __init__(self):
        super(MCFD, self).__init__()

    def forward(self, predict, original):
        # Estimate direction loss
        original_diff = torch.sign(torch.subtract(original[:,1:,:],original[:,:-1,:]))
        predict_diff  = torch.sign(torch.subtract(predict[:,1:,:],predict[:,:-1,:]))
        direction_loss = 1 - torch.mean(torch.eq(original_diff, predict_diff).float()).item()
        
        # MSE + Direction Loss
        alpha = 1000
        loss = torch.multiply(F.mse_loss(predict, original), alpha * direction_loss)

        return loss