

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset


class InteractionDataset(Dataset):
    def __init__(self, interaction_data, interaction_labels):
        self.interaction_data = interaction_data
        self.interaction_labels = interaction_labels

    def __len__(self):
        return len(self.interaction_labels)

    def __getitem__(self, idx):
        protein_emb, rna_emb = self.interaction_data[idx]
        label = torch.tensor(self.interaction_labels[idx])
        return (protein_emb.squeeze(0), rna_emb[0]), label


class GatedFeatureFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        g = self.gate(torch.cat((x1, x2), dim=1))
        return x1 * g + x2 * (1 - g)


class DualPathNetworkRegression(nn.Module):
    def __init__(self, protein_dim, rna_dim, hidden_dim):
        super().__init__()
        self.protein_path = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        self.rna_path = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        self.fusion = GatedFeatureFusion(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, p, r):
        return self.classifier(
            self.fusion(self.protein_path(p), self.rna_path(r))
        ).squeeze(1)


CODE_DIR = r"biollmnet\Code"
MAX_TGT = 10.0
DATASETS = ["Aptamers", "Repeats", "Ribosomal", "Riboswitch", "Viral_RNA", "miRNA"]


def evaluate_saved_val_split(name, code_dir=CODE_DIR, max_tgt=MAX_TGT, device=None):
    """Load <name>_dataset_v1_{best_model.pth, val_data.pkl} and score them.

    Returns a dict with n, pearson, spearman, mae, r2 (or None if files missing).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_pkl = os.path.join(code_dir, f"{name}_dataset_v1_val_data.pkl")
    ckpt = os.path.join(code_dir, f"{name}_dataset_v1_best_model.pth")
    if not (os.path.exists(val_pkl) and os.path.exists(ckpt)):
        return None

    ds = pickle.load(open(val_pkl, "rb"))
    model = DualPathNetworkRegression(768, 768, 256).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    loader = DataLoader(ds, batch_size=32, shuffle=False)
    preds, tgts = [], []
    with torch.no_grad():
        for (drug, rna), tgt in loader:
            preds.append(model(drug.to(device), rna.to(device)).cpu())
            tgts.append(tgt)
    preds = (torch.cat(preds) * max_tgt).numpy()
    tgts = torch.cat(tgts).numpy()

    pear, _ = pearsonr(tgts, preds)
    spear, _ = spearmanr(tgts, preds)
    return {
        "n": int(len(tgts)),
        "pearson": float(pear),
        "spearman": float(spear),
        "mae": float(mean_absolute_error(tgts, preds)),
        "r2": float(r2_score(tgts, preds)),
        "predictions": preds,
        "targets": tgts,
    }


def main():
    header = f"{'dataset':<14}{'n':>5}   {'Pearson':>9}  {'Spearman':>9}  {'R^2':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for name in DATASETS:
        res = evaluate_saved_val_split(name)
        if res is None:
            print(f"{name:<14}  [missing files]")
            continue
        rows.append((name, res))
        print(
            f"{name:<14}{res['n']:>5}   "
            f"{res['pearson']:>+9.4f}  {res['spearman']:>+9.4f}  "
            f"{res['r2']:>+8.4f}"
        )

    if rows:
        arr = np.array([[r["pearson"], r["spearman"], r["r2"]] for _, r in rows])
        print("-" * len(header))
        print(
            f"{'mean':<14}{'':>5}   "
            f"{arr[:, 0].mean():>+9.4f}  {arr[:, 1].mean():>+9.4f}  "
            f"{arr[:, 2].mean():>+8.4f}"
        )


if __name__ == "__main__":
    main()
