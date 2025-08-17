import torch
import torch.nn.functional as F

def source_classification_loss(preds1, preds2, labels):
    return F.cross_entropy(preds1, labels) + F.cross_entropy(preds2, labels)

def feature_alignment_loss(source, target):
    return torch.mean(torch.abs(source - target))

def max_entropy_loss(p1, p2):
    C = p1.size(1)  # number of classes
    LOG2C = torch.log2(torch.tensor(C, dtype=torch.float32, device=p1.device))
    def entropy(p):
        return -torch.sum(p * torch.log(p + 1e-6))/LOG2C
    # Conver to probablities
    p1, p2 = F.softmax(p1, dim=1), F.softmax(p2, dim=1)
    # Expectation over all samples
    Ep1, Ep2 = torch.mean(p1,dim=0), torch.mean(p2,dim=0)
    return -0.5 * (entropy(Ep1) + entropy(Ep2))

def classifier_discrepancy(p1, p2):
    return torch.mean(torch.abs(F.softmax(p1, dim=1) - F.softmax(p2, dim=1)))