import torch
import torch.nn as nn
import torch.nn.functional as F

# adversarial loss
adv_loss_fn = nn.CrossEntropyLoss()

# content loss
class ContentLoss(nn.Module):
    def __init__(self, target_feature):
        super(ContentLoss, self).__init__()
        self.target = target_feature.detach()
        self.loss_fn = nn.MSELoss()

    def forward(self, input_feature):
        self.loss = self.loss_fn(input_feature, self.target)
        return self.loss

def gram_matrix(feature):
    B, C, H, W = feature.size()
    F_flat = feature.view(B, C, H * W)
    G = torch.bmm(F_flat, F_flat.transpose(1, 2))
    return G

# style loss
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target_gram = gram_matrix(target_feature).detach()
        self.loss_fn = nn.MSELoss()

    def forward(self, input_feature):
        G = gram_matrix(input_feature)
        self.loss = self.loss_fn(G, self.target_gram)
        return self.loss

def tv_loss(x):
    return (x[..., :, 1:] - x[..., :, :-1]).abs().mean() + (x[..., 1:, :] - x[..., :-1, :]).abs().mean()


