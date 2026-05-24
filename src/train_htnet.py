#!/usr/bin/env python3
"""
train_htnet_exclude_other.py
Fine-tune HTNet on CASME-II (excluding the "others" class) with LOSO CV,
transfer-learning from a 3-class pretrained checkpoint, grid-search LR,
batch-size, dropout, logging to Weights & Biases, saving only the best model
per grid-run based on average fold UF1 (with UAR tiebreaker), recording all
per-fold metrics & confusion matrices to CSV, tracking per-fold training time,
and excluding the "others" label (orig label 1) at load time.
"""
import os
import itertools
import copy
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import wandb
from Model import HTNet

# === Hardcoded paths ===
DATA_ROOT       = "/home/tsd-pc-06/azhar/bachelor_thesis/datasets/data after/Casme_Augmented-LOSO-split"
PRETRAINED_PATH = "/home/tsd-pc-06/azhar/HTNet-master/ourmodel_threedatasets_weights/sub13.pth"
WANDB_PROJECT   = "casme3_finetune_AUGMENTED"
OUTPUT_CSV      = "exclude_other_grid_results.csv"
SAVE_DIR = "trained_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Updated label mapping (excluding original label 1: 'others')
LABELS = {
    0: 'happiness',
    1: 'disgust',
    2: 'repression',
    3: 'surprise',
    4: 'fear',
    5: 'sadness'
}

class Casme2FlowDataset(Dataset):
    def __init__(self, root, subjects, split, transform=None):
        self.samples = []
        self.transform = transform
        for subj in subjects:
            split_dir = os.path.join(root, subj, split)
            if not os.path.isdir(split_dir):
                continue
            for lbl_str in sorted(os.listdir(split_dir), key=int):
                # skip the "others" class directory
                if lbl_str == '1':
                    continue
                orig_lbl = int(lbl_str)
                # remap labels above 1 down by one (2->1, 3->2, ..., 6->5)
                lbl = orig_lbl if orig_lbl < 1 else orig_lbl - 1
                lbl_dir = os.path.join(split_dir, lbl_str)
                if not os.path.isdir(lbl_dir):
                    continue
                for fn in os.listdir(lbl_dir):
                    if fn.lower().endswith('.png'):
                        self.samples.append((os.path.join(lbl_dir, fn), lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, lbl = self.samples[idx]
        img = __import__('PIL').Image.open(fp).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, lbl


def compute_macro_metrics(gt, pred):
    classes = np.unique(gt)
    f1s, recs = [], []
    for cls in classes:
        y_true = (gt == cls).astype(int)
        y_pred = (pred == cls).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        f1s.append((2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0)
        recs.append(tp/(tp+fn) if (tp+fn)>0 else 0)
    return np.mean(f1s), np.mean(recs)


def main():
    IMG_SIZE   = 28
    N_CLASS    = 6  # six classes after excluding 'others'
    MAX_EPOCHS = 800
    PATIENCE   = 10
    BLOCKS     = (2,2,6)
    DIM        = 256
    HEADS      = 3
    SEED       = 123
    GRID = list(itertools.product(
        [1e-5,1e-4,5e-5],  # learning rates
        [128, 256],         # batch sizes
        [0.0,0.2,0.5]       # dropout rates
    ))

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    subjects = sorted([d for d in os.listdir(DATA_ROOT)
                       if os.path.isdir(os.path.join(DATA_ROOT, d))])

    results = []
    transform = T.Compose([
        T.Resize((IMG_SIZE,IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for lr, batch_size, dropout in GRID:
        run = wandb.init(project=WANDB_PROJECT, reinit=True,
                         config={'lr':lr,'batch_size':batch_size,'dropout':dropout,
                                 'blocks':BLOCKS,'dim':DIM,'heads':HEADS,'seed':SEED})

        best_grid_UF1 = -np.inf
        best_grid_model = None
        fold_metrics = []

        for fold_idx, test_subj in enumerate(subjects):
            fold_start = time.time()
            train_subj = [s for s in subjects if s != test_subj]
            train_ds = Casme2FlowDataset(DATA_ROOT, train_subj, 'train', transform)
            val_ds   = Casme2FlowDataset(DATA_ROOT, [test_subj], 'test', transform)
            dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            dl_val   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

            # ---- skip fold if no validation data (subject had only 'others' frames) ----
            if len(val_ds) == 0:
                print(f"[WARN] Fold {fold_idx} (subject {test_subj}) has 0 validation samples after excluding 'others'. Skipping this fold.")
                continue

            model = HTNet(
                image_size=IMG_SIZE, patch_size=7,
                dim=DIM, heads=HEADS,
                num_hierarchies=len(BLOCKS), block_repeats=BLOCKS,
                mlp_mult=4, dropout=dropout,
                num_classes=N_CLASS
            ).to(device)
            ckpt = torch.load(PRETRAINED_PATH, map_location=device)
            sd = ckpt.get('state_dict', ckpt)
            sd = {k:v for k,v in sd.items() if not k.startswith('mlp_head.2')}
            model.load_state_dict(sd, strict=False)

            for name, param in model.named_parameters():
                if ('layers' in name and '.1' in name) or name.startswith('mlp_head'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            criterion = nn.CrossEntropyLoss()

            best_UF1, best_UAR, patience = -np.inf, -np.inf, 0
            best_weights, best_stats = None, None

            for epoch in range(1, MAX_EPOCHS+1):
                # Train
                model.train()
                tr_loss, y_tr_gt, y_tr_pred = 0.0, [], []
                for xb, yb in dl_train:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item() * xb.size(0)
                    y_tr_gt.extend(yb.cpu().numpy())
                    y_tr_pred.extend(out.argmax(1).cpu().numpy())
                tr_loss /= len(train_ds)
                tr_acc = np.mean(np.array(y_tr_pred) == np.array(y_tr_gt))
                tr_f1m, tr_rem = compute_macro_metrics(y_tr_gt, y_tr_pred)
                tr_f1i = f1_score(y_tr_gt, y_tr_pred, average='micro')
                tr_remi = recall_score(y_tr_gt, y_tr_pred, average='micro')

                # Validate
                model.eval()
                va_loss, y_va_gt, y_va_pred = 0.0, [], []
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb, yb = xb.to(device), yb.to(device)
                        out = model(xb)
                        loss = criterion(out, yb)
                        va_loss += loss.item() * xb.size(0)
                        y_va_gt.extend(yb.cpu().numpy())
                        y_va_pred.extend(out.argmax(1).cpu().numpy())
                va_loss /= len(val_ds)
                va_acc = np.mean(np.array(y_va_pred) == np.array(y_va_gt))
                va_f1m, va_rem = compute_macro_metrics(y_va_gt, y_va_pred)
                va_f1i = f1_score(y_va_gt, y_va_pred, average='micro')
                va_remi = recall_score(y_va_gt, y_va_pred, average='micro')

                wandb.log({
                    'loss/train': tr_loss, 'loss/val': va_loss,
                    'acc/train': tr_acc,   'acc/val': va_acc,
                    'UF1/train': tr_f1m,   'UF1/val': va_f1m,
                    'UAR/train': tr_rem,   'UAR/val': va_rem,
                    'f1_micro/train': tr_f1i, 'f1_micro/val': va_f1i,
                    'recall_micro/train': tr_remi, 'recall_micro/val': va_remi,
                    'epoch': epoch, 'fold': fold_idx
                })

                if (va_f1m > best_UF1) or (va_f1m == best_UF1 and va_rem > best_UAR):
                    best_UF1, best_UAR, patience = va_f1m, va_rem, 0
                    best_weights = copy.deepcopy(model.state_dict())
                    best_stats = (epoch, tr_loss, tr_acc, tr_f1m, tr_rem, tr_f1i, tr_remi,
                                  va_loss, va_acc, va_f1m, va_rem, va_f1i, va_remi)
                else:
                    patience += 1
                    if patience >= PATIENCE:
                        break

            fold_time = time.time() - fold_start

            model.load_state_dict(best_weights)
            cm_plot = wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_va_gt,
                preds=y_va_pred,
                class_names=[LABELS[k] for k in sorted(LABELS.keys())]
            )
            wandb.log({f"confusion_matrix/fold{fold_idx}": cm_plot})

            if best_UF1 > best_grid_UF1:
                best_grid_UF1 = best_UF1
                best_grid_model = copy.deepcopy(best_weights)

            (ep, tr_l, tr_A, tr_f1m, tr_rem, tr_f1i, tr_remi,
             va_l, va_A, va_f1m, va_rem, va_f1i, va_remi) = best_stats
            fold_metrics.append({
                'fold': fold_idx, 'trained_epochs': ep,
                'train_loss': tr_l, 'train_acc': tr_A,
                'train_UF1': tr_f1m, 'train_UAR': tr_rem,
                'train_f1_micro': tr_f1i, 'train_recall_micro': tr_remi,
                'val_loss': va_l,  'val_acc': va_A,
                'val_UF1': va_f1m, 'val_UAR': va_rem,
                'val_f1_micro': va_f1i, 'val_recall_micro': va_remi,
                'train_time_sec': fold_time
            })

        if best_grid_model is not None:
            fname = f"exclude_other_best_lr{lr}_bs{batch_size}_do{dropout}.pth"
            torch.save(best_grid_model, os.path.join(SAVE_DIR, fname))

        df_f = pd.DataFrame(fold_metrics)
        wandb.summary.update({
            'mean_tr_UF1': df_f['train_UF1'].mean(),
            'mean_tr_UAR': df_f['train_UAR'].mean(),
            'mean_tr_ACC': df_f['train_acc'].mean(),
            'mean_tr_loss': df_f['train_loss'].mean(),
            'mean_va_UF1': df_f['val_UF1'].mean(),
            'mean_va_UAR': df_f['val_UAR'].mean(),
            'mean_va_ACC': df_f['val_acc'].mean(),
            'mean_va_loss': df_f['val_loss'].mean(),
        })
        run.finish()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"[CUDA] Cache cleared after grid run: LR={lr}, BS={batch_size}, DO={dropout}")
        print(f"[CUDA] Allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB | Reserved: {torch.cuda.memory_reserved()/1e6:.2f} MB")

        for fm in fold_metrics:
            results.append({'lr': lr, 'batch_size': batch_size, 'dropout': dropout, **fm})

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
