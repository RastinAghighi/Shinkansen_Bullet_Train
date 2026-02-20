"""
THE NUCLEAR OPTION: DAE + MLP (Porto Seguro Architecture)
Target: >0.96 by learning the latent manifold of survey data.
"""

import pandas as pd
import numpy as np
import os
import sys
import zipfile
import time
import random

# --- INSTALL CHECK ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from scipy.special import erfinv
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("[INSTALL] Installing PyTorch & SciPy...")
    os.system('pip install torch scipy scikit-learn pandas numpy')
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from scipy.special import erfinv
    from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
SUBMISSION_FILE = 'Submission_DAE_Nuclear.csv'
ZIP_PATH = os.path.join('data', 'Olympus', 'archive.zip')

# HYPERPARAMETERS (Tuned for ~90k rows)
CONFIG = {
    'seed': 42,
    'epochs_dae': 30,      # Pre-training epochs
    'epochs_mlp': 20,      # Fine-tuning epochs
    'batch_size': 256,
    'lr_dae': 0.001,
    'lr_mlp': 0.0005,
    'swap_prob': 0.15,     # The "Magic" Noise Level
    'hidden_size': 1024,   # Wide layers
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

FILES = {
    'train_travel': 'Traveldata_train_(1).csv',
    'train_survey': 'Surveydata_train_(1).csv',
    'test_travel': 'Traveldata_test_(1).csv',
    'test_survey': 'Surveydata_test_(1).csv'
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])

# --- RANK GAUSS (The Secret Sauce) ---
class RankGaussScaler:
    def fit_transform(self, X):
        # Convert to Rank, normalize to -1 to 1, apply Inverse Error Function
        # This forces data into a perfect Gaussian shape
        X = np.array(X)
        X_new = np.zeros_like(X)
        for i in range(X.shape[1]):
            col = X[:, i]
            # argsort twice gives ranks
            rank = np.argsort(np.argsort(col))
            N = len(rank)
            # Normalize to (-1, 1) excluding edges to avoid infinity
            rank = (rank / (N + 1)) * 2 - 1
            rank = np.clip(rank, -0.99999, 0.99999)
            X_new[:, i] = erfinv(rank)
        return X_new

# --- DATASET WITH SWAP NOISE ---
class SwapNoiseDataset(Dataset):
    def __init__(self, x_num, x_cat, y=None, swap_prob=0.15, is_train=True):
        self.x_num = torch.FloatTensor(x_num)
        self.x_cat = torch.LongTensor(x_cat)
        self.y = torch.FloatTensor(y).unsqueeze(1) if y is not None else None
        self.swap_prob = swap_prob
        self.is_train = is_train
        
    def __len__(self):
        return len(self.x_num)
    
    def __getitem__(self, idx):
        x_n = self.x_num[idx].clone()
        x_c = self.x_cat[idx].clone()
        
        # Apply Swap Noise (Only during training)
        if self.is_train and self.swap_prob > 0:
            # Numerical Swap
            if torch.rand(1) < self.swap_prob:
                swap_idx = torch.randint(0, len(self.x_num), (1,))
                x_n = self.x_num[swap_idx].squeeze()
            
            # Categorical Swap
            if torch.rand(1) < self.swap_prob:
                swap_idx = torch.randint(0, len(self.x_cat), (1,))
                x_c = self.x_cat[swap_idx].squeeze()
                
        # Return: Corrupted, Clean, Target
        target = self.y[idx] if self.y is not None else torch.tensor(0.0)
        return x_n, x_c, self.x_num[idx], self.x_cat[idx], target

# --- THE MODEL (DAE + MLP) ---
class TabularDAE(nn.Module):
    def __init__(self, n_num, cat_cards, hidden_size=1024):
        super().__init__()
        
        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, min(50, (card + 1) // 2)) for card in cat_cards
        ])
        emb_total = sum(e.embedding_dim for e in self.embeddings)
        input_dim = n_num + emb_total
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(), # Swish activation
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU()
        )
        
        # Decoder (Reconstruction)
        self.decoder_num = nn.Linear(hidden_size, n_num)
        self.decoder_cat = nn.ModuleList([
            nn.Linear(hidden_size, card) for card in cat_cards
        ])
        
        # Classification Head (For Phase 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x_num, x_cat):
        # Embed
        emb_out = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat([x_num] + emb_out, dim=1)
        
        # Encode
        latent = self.encoder(x)
        
        # Decode (Reconstruction)
        rec_num = self.decoder_num(latent)
        rec_cat = [d(latent) for d in self.decoder_cat]
        
        # Classify
        logits = self.head(latent)
        
        return rec_num, rec_cat, logits

def load_data():
    print(f"[INFO] Loading Data from {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        data = {k: pd.read_csv(z.open(v)) for k, v in FILES.items()}
    
    train = pd.merge(data['train_travel'], data['train_survey'], on='ID')
    test = pd.merge(data['test_travel'], data['test_survey'], on='ID')
    
    target = train['Overall_Experience']
    ids = test['ID']
    train.drop(['ID', 'Overall_Experience'], axis=1, inplace=True)
    test.drop(['ID'], axis=1, inplace=True)
    
    # Combined for RankGauss
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Simple Fillna
    df['Total_Delay'] = df['Departure_Delay_in_Mins'].fillna(0) + df['Arrival_Delay_in_Mins'].fillna(0)
    df = df.fillna(0) # DAE handles zeros well
    
    return df, target, ids, len(train)

def main():
    df, y_train, ids, train_len = load_data()
    
    # --- PREPROCESSING ---
    print("[INFO] Applying RankGauss & Encoding...")
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() < 20]
    num_cols = [c for c in df.columns if c not in cat_cols]
    
    # RankGauss for Numericals
    scaler = RankGaussScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols].values)
    
    # Label Encode Categoricals
    cat_cards = []
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        cat_cards.append(len(le.classes_))
        
    X_num = df[num_cols].values
    X_cat = df[cat_cols].values
    y_vals = y_train.values
    
    # Split back
    X_train_num = X_num[:train_len]
    X_train_cat = X_cat[:train_len]
    X_test_num = X_num[train_len:]
    X_test_cat = X_cat[train_len:]
    
    # DAE Dataset (Uses ALL data for pre-training)
    dae_ds = SwapNoiseDataset(X_num, X_cat, swap_prob=CONFIG['swap_prob'], is_train=True)
    dae_loader = DataLoader(dae_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Supervised Dataset (Train only)
    sup_ds = SwapNoiseDataset(X_train_num, X_train_cat, y_vals, swap_prob=0.0, is_train=True)
    sup_loader = DataLoader(sup_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # --- MODEL SETUP ---
    device = CONFIG['device']
    print(f"[INFO] Training on {device}...")
    
    model = TabularDAE(len(num_cols), cat_cards, CONFIG['hidden_size']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr_dae'])
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    
    # --- PHASE 1: UNSUPERVISED PRE-TRAINING ---
    print("\n[PHASE 1] Unsupervised Denoising (Learning the Manifold)...")
    for epoch in range(CONFIG['epochs_dae']):
        model.train()
        avg_loss = 0
        for x_n_corr, x_c_corr, x_n_clean, x_c_clean, _ in dae_loader:
            x_n_corr, x_c_corr = x_n_corr.to(device), x_c_corr.to(device)
            x_n_clean, x_c_clean = x_n_clean.to(device), x_c_clean.to(device)
            
            optimizer.zero_grad()
            rec_num, rec_cat, _ = model(x_n_corr, x_c_corr)
            
            loss = mse(rec_num, x_n_clean)
            for i, r in enumerate(rec_cat):
                loss += ce(r, x_c_clean[:, i])
                
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
        print(f"   Epoch {epoch+1}/{CONFIG['epochs_dae']} - Recon Loss: {avg_loss/len(dae_loader):.4f}")
        
    # --- PHASE 2: SUPERVISED FINE-TUNING ---
    print("\n[PHASE 2] Supervised Classification...")
    # Lower LR for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = CONFIG['lr_mlp']
        
    for epoch in range(CONFIG['epochs_mlp']):
        model.train()
        avg_loss = 0
        for _, _, x_n, x_c, y in sup_loader:
            x_n, x_c, y = x_n.to(device), x_c.to(device), y.to(device)
            
            optimizer.zero_grad()
            _, _, logits = model(x_n, x_c)
            loss = bce(logits, y)
            
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
        print(f"   Epoch {epoch+1}/{CONFIG['epochs_mlp']} - Class Loss: {avg_loss/len(sup_loader):.4f}")
        
    # --- PREDICTION ---
    print("\n[INFO] Generating Predictions...")
    model.eval()
    test_ds = SwapNoiseDataset(X_test_num, X_test_cat, y=None, swap_prob=0.0, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    
    preds = []
    with torch.no_grad():
        for _, _, x_n, x_c, _ in test_loader:
            x_n, x_c = x_n.to(device), x_c.to(device)
            _, _, logits = model(x_n, x_c)
            preds.append(torch.sigmoid(logits).cpu().numpy())
            
    preds = np.concatenate(preds).flatten()
    final_preds = (preds > 0.5).astype(int)
    
    sub = pd.DataFrame({'ID': ids, 'Overall_Experience': final_preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    
    # Also save Probabilities for Blending (CRITICAL)
    sub_prob = pd.DataFrame({'ID': ids, 'Prob_DAE': preds})
    sub_prob.to_csv('DAE_Probs.csv', index=False)
    
    print(f"[VICTORY] Saved {SUBMISSION_FILE}")
    print("[NOTE] Also saved 'DAE_Probs.csv' for blending.")

if __name__ == "__main__":
    main()