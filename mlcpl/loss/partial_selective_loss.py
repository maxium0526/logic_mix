import torch
from torch import nn as nn

def PartialAsymmetricLoss(
    clip = 0,
    gamma_pos = 0,
    gamma_neg = 1,
    alpha_pos = 1,
    alpha_neg = 1,
    discard_focal_grad = True,
    reduction = 'mean',
):
    return PartialLoss(
        lossfn_pos = FocalLossTerm(alpha_pos, gamma_pos, discard_focal_grad=discard_focal_grad),
        lossfn_neg = FocalLossTerm(alpha_neg, gamma_neg, clip, discard_focal_grad=discard_focal_grad),

        reduction = reduction,
    )
    
class FocalLossTerm(nn.Module):
    def __init__(self, alpha=1, gamma=1, shift=0, discard_focal_grad=True) -> None:
        super(FocalLossTerm, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.shift = shift # negative term of asymmetric loss
        self.discard_focal_grad = discard_focal_grad
    
    def forward(self, p):
        p = torch.clamp(p + self.shift, max=1)
        p_focal = p.detach() if self.discard_focal_grad else p

        return - self.alpha * torch.pow(1 - p_focal, self.gamma) * torch.log(p)

# from https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/src/loss_functions/partial_asymmetric_loss.py 
class PartialLoss(nn.Module):
    def __init__(
            self,
            lossfn_pos = FocalLossTerm(),
            lossfn_neg = FocalLossTerm(),
            reduction = 'mean',
            ):
        super(PartialLoss, self).__init__()

        self.lossfn_pos = lossfn_pos
        self.lossfn_neg = lossfn_neg

        self.reduction = reduction

    def forward(self, logits, targets):
        targets = torch.where(torch.isnan(targets), -1, targets) # adopt to my code

        targets_pos = (targets==1).float()
        targets_neg = (targets==0).float()

        # Activation
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Loss calculation
        loss_pos = targets_pos * self.lossfn_pos(torch.clamp(xs_pos, min=1e-8))
        loss_neg = targets_neg * self.lossfn_neg(torch.clamp(xs_neg, min=1e-8))

        total_loss = loss_pos + loss_neg

        if self.reduction == 'mean':
            return total_loss.mean()
        if self.reduction == 'sum':
            return total_loss.sum()
        return total_loss