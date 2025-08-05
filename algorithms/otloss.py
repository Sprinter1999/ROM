import torch
import torch.nn.functional as F
# import ot
import numpy as np


#TODO: We use the center classification loss implementation to replace the OT loss which can be computationally expensive
def ot_loss(proto_t, feat_tu_w, feat_tu_s, args):
    # bs = feat_tu_s.shape[0]
    # with torch.no_grad():
    #     M_st_weak = 1 - pairwise_cosine_sim(proto_t.mo_pro, feat_tu_w)
    # gamma_st_weak = ot_mapping(M_st_weak.data.cpu().numpy().astype(np.float64))
    # score_ot, pred_ot = gamma_st_weak.t().max(dim=1)
    # pseudo_label_ot = pred_ot.clone().detach()
    # Lm = center_loss_cls(proto_t.mo_pro, feat_tu_s, pred_ot, num_classes=args.num_classes)
    # return Lm, pseudo_label_ot, score_ot
    pass

def pairwise_cosine_sim(a, b):
    pass


def ot_mapping(M):
    pass

def center_loss_cls(centers, x, labels=None, num_classes=15):
    batch_size = x.size(0)
    centers_norm2 = F.normalize(centers)
    x = F.normalize(x)
    # Only use non-NaN centers for distance calculation
    valid_mask = ~torch.isnan(centers_norm2).any(dim=1)
    valid_centers = centers_norm2[valid_mask]
    valid_classes = torch.arange(num_classes, device=x.device)[valid_mask]
    if valid_centers.size(0) == 0:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
    distmat = -1. * x @ valid_centers.t() + 1  # [batch, valid_centers]
    if labels is None:
        loss = distmat.min(dim=1)[0].mean()
    else:
        # Only apply mask for valid classes
        batch_size = x.size(0)
        labels = labels.unsqueeze(1)
        mask = labels == valid_classes.unsqueeze(0)  # [batch, valid_centers]
        dist = distmat * mask.float()
        # If a sample's label is in valid classes, mask is True, otherwise False (0)
        denom = mask.float().sum(dim=1).clamp(min=1)
        loss = dist.sum(dim=1) / denom
        loss = loss.mean()
    return loss