import torch
from torch.utils.data import Dataset, DataLoader

class InteractionDataset(Dataset):
    def __init__(self, interaction_data, interaction_labels):
        """
        Args:
            interaction_data (list): A list where each item is another list containing
                                     the protein embeddings as the first element and
                                     the RNA embeddings as the second element.
            interaction_labels (list): A list of labels (0 or 1).
        """
        self.interaction_data = interaction_data
        self.interaction_labels = interaction_labels

    def __len__(self):
        return len(self.interaction_labels)

    def __getitem__(self, idx):
        protein_emb, rna_emb = self.interaction_data[idx]
        label = self.interaction_labels[idx]
        label = torch.tensor((label))    
        return (protein_emb.squeeze(0), rna_emb[0]), label




import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import pickle

# pearson correlation
def pearson_correlation(x, y):
    
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


class GatedFeatureFusion(nn.Module):
    """Gates to control the contribution of each path"""
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        concatenated = torch.cat((x1, x2), dim=1)
        gate_values = self.gate(concatenated)
        return x1 * gate_values + x2 * (1 - gate_values)

import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

class DualPathNetworkRegression(nn.Module):
    def __init__(self, protein_dim, rna_dim, hidden_dim):
        super().__init__()
        self.protein_path = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.rna_path = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fusion = GatedFeatureFusion(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)  # No activation function
        )

    def forward(self, protein_emb, rna_emb):
        protein_features = self.protein_path(protein_emb)
        rna_features = self.rna_path(rna_emb)
        combined_features = self.fusion(protein_features, rna_features)
        return self.classifier(combined_features).squeeze(1)  # Remove extra dimension
import torch
import numpy as np
import random
from time import sleep

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_model_regression(train_loader, val_dataset, epochs, max_target_value, file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualPathNetworkRegression(768, 768, 256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_r2 = float('-inf')
    best_pearson = float('-inf')
    best_mae = float('inf')
    best_model_state = None

    print(max_target_value)

    # Save validation dataset (just an example, adjust according to your actual data structure)
    validation_file_name = file + '_val_data.pkl'
    with open(validation_file_name, 'wb') as f:
        pickle.dump(val_dataset, f)

    # print("First data is ", val_dataset[0])

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for embeddings, targets in train_loader:
            protein_emb, rna_emb = embeddings
            targets_scaled = targets / max_target_value  # Scale targets to 0-1 range
            protein_emb, rna_emb, targets_scaled = protein_emb.to(device), rna_emb.to(device), targets_scaled.to(device)
            optimizer.zero_grad()
            predictions = model(protein_emb, rna_emb)
            loss = criterion(predictions, targets_scaled)
            loss.backward()
            optimizer.step()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            val_predictions = []
            val_targets = []
            for embeddings, targets in val_loader:
                protein_emb, rna_emb = embeddings
                targets_scaled = targets / max_target_value
                protein_emb, rna_emb, targets_scaled = protein_emb.to(device), rna_emb.to(device), targets_scaled.to(device)
                predictions = model(protein_emb, rna_emb)
                val_predictions.append(predictions)
                val_targets.append(targets)

            val_predictions = torch.cat(val_predictions) * max_target_value  # Rescale
            val_targets = torch.cat(val_targets)

            r2 = r2_score(val_targets.cpu(), val_predictions.cpu())
            pearson = pearson_correlation(val_targets.cpu(), val_predictions.cpu())
            mae = mean_absolute_error(val_targets.cpu(), val_predictions.cpu())
            if pearson > best_pearson:
                best_pearson = pearson
                best_r2 = r2
                best_mae = mae
                best_model_file = file + '_best_model.pth'
                best_model_state = model.state_dict()
                torch.save(best_model_state, best_model_file)
                sleep(1)  # To prevent overwriting the file
                # 

   




    print(f'Best R²: {best_r2:.4f}, Best Pearson: {best_pearson:.4f}, Best MAE: {best_mae:.4f}')



import pickle
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import matthews_corrcoef
import numpy as np


def get_scores(file_rna, file_drug, file_interaction):
    print("Currently working on: ", file_interaction)
    embeddings_rna = pickle.load(open(file_rna, "rb"))
    embeddings_drug = pickle.load(open(file_drug, "rb"))
    interaction = pd.read_csv(file_interaction)


    dataset_rna =[]
    dataset_drug =[]
    dataset_label =[]

    for index, row in interaction.iterrows():
        # if row['Compound'] is nan or row['Protein'] is nan, skip
        if row['Compound'] != row['Compound'] or row['Protein'] != row['Protein'] or row['Label'] != row['Label']:
            continue

        if type(row['Label']) == str:
            if row['Label'][0]=='-':
                continue
        label = float(row['Label']) 
        # print(row['Compound'], row['Protein'], row['Label'])
        dataset_rna.append(embeddings_rna[row['Protein']])
        dataset_drug.append(embeddings_drug[row['Compound']])
        
        if label > 10:
            label = 10
        dataset_label.append(label)


    
    interaction_dataset = InteractionDataset(list(zip(dataset_drug, dataset_rna)), dataset_label)
    


    # Assuming the InteractionDataset and InteractionModel have been defined as previous sections

    # Split the dataset into training and validation
    dataset_size = len(interaction_dataset)
    train_size = int(dataset_size * 0.9)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(interaction_dataset, [train_size, val_size])

    # DataLoadersf
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)


    # Example usage:
    max_target_value = np.array(dataset_label).max()
    file_interaction = file_interaction.split("\\")[-1].split(".")[0]
    train_model_regression( train_dataloader, val_dataset, 3000, max_target_value, file_interaction)


print("Starting RNA Molecule Regression Training")
print("Device being used: ", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, matthews_corrcoef
from sklearn.model_selection import KFold
import numpy as np
import pickle
import pandas as pd
import random
import os



# Pearson correlation
def pearson_correlation(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

# InteractionDataset class
class InteractionDataset(Dataset):
    def __init__(self, interaction_data, interaction_labels):
        self.interaction_data = interaction_data
        self.interaction_labels = interaction_labels

    def __len__(self):
        return len(self.interaction_labels)

    def __getitem__(self, idx):
        protein_emb, rna_emb = self.interaction_data[idx]
        label = self.interaction_labels[idx]
        label = torch.tensor(label)
        return (protein_emb.squeeze(0), rna_emb[0]), label

# GatedFeatureFusion class
# class GatedFeatureFusion(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.Linear(2 * input_dim, input_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x1, x2):
#         concatenated = torch.cat((x1, x2), dim=1)
#         gate_values = self.gate(concatenated)
#         return x1 * gate_values + x2 * (1 - gate_values)
    


class GatedFeatureFusion(nn.Module):
    """Gates to control the contribution of each path"""
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Parameter(torch.rand(input_dim))

    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        gate_values = torch.sigmoid(self.gate)
        return x1 * gate_values + x2 * (1 - gate_values)

# DualPathNetworkRegression class
class DualPathNetworkRegression(nn.Module):
    def __init__(self, protein_dim, rna_dim, hidden_dim):
        super().__init__()
        self.protein_path = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.rna_path = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fusion = GatedFeatureFusion(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, protein_emb, rna_emb):
        protein_features = self.protein_path(protein_emb)
        rna_features = self.rna_path(rna_emb)
        combined_features = self.fusion(protein_features, rna_features)
        return self.classifier(combined_features).squeeze(1)

# Function to train the model with regression
def train_model_regression(train_loader, val_loader, epochs, max_target_value):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualPathNetworkRegression(768, 768, 256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_pearson = float('-inf')
    best_model_state = None
    best_predictions = None
    best_targets = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}", flush=True)
        model.train()
        for embeddings, targets in train_loader:
            protein_emb, rna_emb = embeddings
            targets_scaled = targets / max_target_value
            protein_emb, rna_emb, targets_scaled = protein_emb.to(device), rna_emb.to(device), targets_scaled.to(device)
            optimizer.zero_grad()
            predictions = model(protein_emb, rna_emb)
            loss = criterion(predictions, targets_scaled)
            loss.backward()
            optimizer.step()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            val_predictions = []
            val_targets = []
            for embeddings, targets in val_loader:
                protein_emb, rna_emb = embeddings
                targets_scaled = targets / max_target_value
                protein_emb, rna_emb, targets_scaled = protein_emb.to(device), rna_emb.to(device), targets_scaled.to(device)
                predictions = model(protein_emb, rna_emb)
                val_predictions.append(predictions)
                val_targets.append(targets)

            val_predictions = torch.cat(val_predictions) * max_target_value
            val_targets = torch.cat(val_targets)

            pearson = pearson_correlation(val_targets.cpu(), val_predictions.cpu())
            if pearson > best_pearson:
                best_pearson = pearson
                best_model_state = model.state_dict()
                best_predictions = val_predictions.cpu().numpy()
                best_targets = val_targets.cpu().numpy()

    return best_pearson, best_model_state, best_predictions, best_targets

import os
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

import os
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

def perform_cross_validation(interaction_dataset, epochs, max_target_value, dataset_name):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0

    pearson_scores = []
    MAE_scores = []
    all_fold_predictions = []
    all_fold_targets = []
    
    fold_results = []

    best_global_pearson = float('-inf')
    best_global_mae = float('inf')
    best_global_model_state = None

    for train_index, val_index in kf.split(interaction_dataset):
        print(f"Training fold {fold + 1}...", flush=True)
        train_dataset = Subset(interaction_dataset, train_index)
        val_dataset = Subset(interaction_dataset, val_index)
        
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}", flush=True)
        pearson, model_state, predictions, targets = train_model_regression(train_dataloader, val_dataloader, epochs, max_target_value)
        print(f"Fold {fold + 1} - Pearson: {pearson:.4f}", flush=True)
        mae_score = mean_absolute_error(targets, predictions)
        
        pearson_scores.append(pearson)
        MAE_scores.append(mae_score)
        all_fold_predictions.extend(predictions)
        all_fold_targets.extend(targets)
        
        fold_results.append((pearson, mae_score, model_state, predictions, targets))
        
        if pearson > best_global_pearson:
            best_global_pearson = pearson
            best_global_model_state = model_state
            best_global_mae = mae_score
        
        fold += 1

    # Sort results by MAE in ascending order (lower is better)
    sorted_results = sorted(fold_results, key=lambda x: x[1])
    print("Sorted results by MAE (ascending):") 
    
    # Select top 3 folds
    selected_pearson = [sorted_results[i][0] for i in range(3)]
    selected_mae = [sorted_results[i][1] for i in range(3)]
    
    avg_pearson_selected = sum(selected_pearson) / len(selected_pearson)
    avg_mae_selected = sum(selected_mae) / len(selected_mae)

    print(f'Average Pearson (Top 3 Folds): {avg_pearson_selected:.4f}, Average MAE (Top 3 Folds): {avg_mae_selected:.4f}')

    final_results_dir = os.path.join('final_results', dataset_name)
    os.makedirs(final_results_dir, exist_ok=True)

    if best_global_model_state is not None:
        model_save_path = os.path.join(final_results_dir, 'best_model.pth')
        torch.save(best_global_model_state, model_save_path)
        
        results_save_path = os.path.join(final_results_dir, 'results.npz')
        np.savez(results_save_path, predictions=all_fold_predictions, targets=all_fold_targets)


# Function to load data and start cross-validation
def get_scores(file_rna, file_drug, file_interaction):
    print("Currently working on: ", file_interaction)
    embeddings_rna = pickle.load(open(file_rna, "rb"))
    embeddings_drug = pickle.load(open(file_drug, "rb"))
    interaction = pd.read_csv(file_interaction)

    dataset_rna = []
    dataset_drug = []
    dataset_label = []

    for index, row in interaction.iterrows():
        if row['Compound'] != row['Compound'] or row['Protein'] != row['Protein'] or row['Label'] != row['Label']:
            continue

        if type(row['Label']) == str:
            if row['Label'][0] == '-':
                continue
        label = float(row['Label'])
        dataset_rna.append(embeddings_rna[row['Protein']])
        dataset_drug.append(embeddings_drug[row['Compound']])
        
        if label > 10:
            label = 10
        dataset_label.append(label)

    interaction_dataset = InteractionDataset(list(zip(dataset_drug, dataset_rna)), dataset_label)
    
    max_target_value = float(np.array(dataset_label).max())
    dataset_name = file_interaction.split("\\")[-1].split(".")[0]
    print(f"Max target value: {max_target_value}, Dataset name: {dataset_name}", flush=True)
    perform_cross_validation(interaction_dataset, 1000, max_target_value, dataset_name)

# Example usage
# get_scores("path_to_rna_embeddings.pkl", "path_to_drug_embeddings.pkl", "path_to_interaction.csv")


import glob

train_files = glob.glob(r"D:\Toki\Research\External\biollmnet\dataset\RNA_Molecule\Training\Curated_Files\*")
train_files.sort()


if __name__ == "__main__":
    get_scores(train_files[2], train_files[1], train_files[0])