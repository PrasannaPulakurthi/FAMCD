import torch
import torch.nn.functional as F

def source_classification_loss(preds1, preds2, labels):
    return F.cross_entropy(preds1, labels) + F.cross_entropy(preds2, labels)

def feature_alignment_loss(f_src, f_tgt):
    return torch.mean(torch.abs(f_src - f_tgt))

def max_entropy_loss(preds1, preds2):
    def entropy(p):
        p = F.softmax(p, dim=1)
        return -torch.sum(p * torch.log(p + 1e-6), dim=1).mean()

    return -0.5 * (entropy(preds1) + entropy(preds2))

def classifier_discrepancy(p1, p2):
    return torch.mean(torch.abs(F.softmax(p1, dim=1) - F.softmax(p2, dim=1)))