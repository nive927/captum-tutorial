# torchrec_captum_visual.py

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from captum.attr import IntegratedGradients, NoiseTunnel
import matplotlib.pyplot as plt
import numpy as np

# Dataset
class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['userId'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movieId'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# Model
class RecSys(nn.Module):
    def __init__(self, n_users, n_movies, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, emb_dim)
        self.movie_emb = nn.Embedding(n_movies + 1, emb_dim)
        self.fc = nn.Linear(emb_dim * 2, 1)

    def forward(self, user_ids, movie_ids):
        user_emb = self.user_emb(user_ids)
        movie_emb = self.movie_emb(movie_ids)
        x = torch.cat([user_emb, movie_emb], dim=1)
        return self.fc(x).squeeze()

# Load & prepare data
df = pd.read_csv("data/ml-latest-small/ratings.csv")
user_map = {u: i for i, u in enumerate(df['userId'].unique())}
movie_map = {m: i for i, m in enumerate(df['movieId'].unique())}
df['userId'] = df['userId'].map(user_map)
df['movieId'] = df['movieId'].map(movie_map)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = MovieLensDataset(train_df)
test_dataset = MovieLensDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# Train model
n_users = len(user_map)
n_movies = len(movie_map)
model = RecSys(n_users, n_movies)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(5):
    total_loss = 0
    for users, movies, ratings in train_loader:
        preds = model(users, movies)
        loss = loss_fn(preds, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    all_preds, all_targets = [], []
    for users, movies, ratings in test_loader:
        pred = model(users, movies)
        all_preds.extend(pred.numpy())
        all_targets.extend(ratings.numpy())
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    print(f"Test MAE: {mae:.4f}")

# ---- Captum ---- #
# Choose a test sample
sample_user_id = test_dataset[0][0].item()
sample_movie_id = test_dataset[0][1].item()

# Forward only on embedding vector
user_embed = model.user_emb(torch.tensor([sample_user_id]))
movie_embed = model.movie_emb(torch.tensor([sample_movie_id]))
input_embed = torch.cat([user_embed, movie_embed], dim=1)
baseline_embed = torch.zeros_like(input_embed)

# Custom forward
def embed_forward(x):
    return model.fc(x)

# Run IG
ig = IntegratedGradients(embed_forward)
attributions, delta = ig.attribute(inputs=input_embed,
                                   baselines=baseline_embed,
                                   return_convergence_delta=True)

# Run Noise Tunnel (optional)
nt = NoiseTunnel(ig)
nt_attr = nt.attribute(input_embed, baselines=baseline_embed, nt_samples=10, nt_type='smoothgrad')

# Attribution Split
user_attr = attributions[:, :model.user_emb.embedding_dim].detach().numpy().flatten()
movie_attr = attributions[:, model.user_emb.embedding_dim:].detach().numpy().flatten()
smoothed_attr = nt_attr.detach().numpy().flatten()

# ---- Visualization ---- #
def plot_attributions(attr, title, color="tab:blue"):
    plt.bar(np.arange(len(attr)), attr, color=color)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Attribution Score")
    plt.title(title)
    plt.grid(True)

plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plot_attributions(user_attr, "User Embedding Attribution")

plt.subplot(1, 3, 2)
plot_attributions(movie_attr, "Movie Embedding Attribution", color="tab:orange")

plt.subplot(1, 3, 3)
plot_attributions(smoothed_attr, "Smoothed IG Attribution (Noise Tunnel)", color="tab:green")

plt.tight_layout()
plt.show()

# Convergence
print(f"\n✅ Convergence Delta: {delta.item():.6f}")
