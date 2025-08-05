import torch
import torch.nn.functional as F

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10, reduction='mean'):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)  # shape: [batch] if reduction='none', else scalar

        # RCE
        pred_softmax = F.softmax(pred, dim=1)
        pred_softmax = torch.clamp(pred_softmax, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred_softmax * torch.log(label_one_hot), dim=1))  # shape: [batch]

        # Loss
        if self.reduction == 'none':
            loss = self.alpha * ce + self.beta * rce
        else:
            loss = self.alpha * ce + self.beta * rce.mean()
        return loss