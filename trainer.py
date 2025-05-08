import torch
from losses import source_classification_loss, feature_alignment_loss, max_entropy_loss, classifier_discrepancy
import torch.nn.functional as F

def evaluate(model_G, model_F1, model_F2, loader, device):
    model_G.eval()
    model_F1.eval()
    model_F2.eval()
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feats = model_G(x)
            preds1 = model_F1(feats)
            preds2 = model_F2(feats)
            avg_preds = (F.softmax(preds1, dim=1) + F.softmax(preds2, dim=1)) / 2
            pred_labels = avg_preds.argmax(dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)

    acc = 100. * correct / total
    return acc


def train_epoch_step1(model_G, model_F1, model_F2, optimizer_G, optimizer_F, loader_src, loader_tgt, device, lambda_fa=0.25, lambda_h=0.5):
    model_G.train()
    model_F1.train()
    model_F2.train()

    for (x_s, y_s), (x_t, _) in zip(loader_src, loader_tgt):
        x_s, y_s = x_s.to(device), y_s.to(device)
        x_t = x_t.to(device)

        f_s = model_G(x_s)
        f_t = model_G(x_t)
        out_s1 = model_F1(f_s)
        out_s2 = model_F2(f_s)
        out_t1 = model_F1(f_t)
        out_t2 = model_F2(f_t)

        loss = source_classification_loss(out_s1, out_s2, y_s)
        # loss += lambda_fa * feature_alignment_loss(f_s, f_t)
        # loss += lambda_h * max_entropy_loss(out_t1, out_t2)

        optimizer_G.zero_grad()
        optimizer_F.zero_grad()
        loss.backward()
        optimizer_G.step()
        optimizer_F.step()

def train_epoch_step2(model_G, model_F1, model_F2, optimizer_F, loader_src, loader_tgt, device):
    model_G.eval()
    model_F1.train()
    model_F2.train()

    for (x_s, y_s), (x_t, _) in zip(loader_src, loader_tgt):
        x_s, y_s = x_s.to(device), y_s.to(device)
        x_t = x_t.to(device)

        with torch.no_grad():
            f_s = model_G(x_s)
            f_t = model_G(x_t)

        out_s1 = model_F1(f_s)
        out_s2 = model_F2(f_s)
        out_t1 = model_F1(f_t)
        out_t2 = model_F2(f_t)

        loss = source_classification_loss(out_s1, out_s2, y_s)
        loss -= classifier_discrepancy(out_t1, out_t2)

        optimizer_F.zero_grad()
        loss.backward()
        optimizer_F.step()


def train_epoch_step3(model_G, model_F1, model_F2, optimizer_G, loader_tgt, device, repeat=4):
    model_G.train()
    model_F1.eval()
    model_F2.eval()

    for _ in range(repeat):
        for (x_t, _) in loader_tgt:
            x_t = x_t.to(device)
            f_t = model_G(x_t)
            out_t1 = model_F1(f_t)
            out_t2 = model_F2(f_t)

            loss = classifier_discrepancy(out_t1, out_t2)
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
