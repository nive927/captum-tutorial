# torchrec_captum_tutorial.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from captum.attr import IntegratedGradients
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

# Data prep
df = pd.read_csv("data/ml-latest-small/ratings.csv")

# Encode users and movies to 0-based IDs
user_map = {u: i for i, u in enumerate(df['userId'].unique())}
movie_map = {m: i for i, m in enumerate(df['movieId'].unique())}
df['userId'] = df['userId'].map(user_map)
df['movieId'] = df['movieId'].map(movie_map)

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = MovieLensDataset(train_df)
test_dataset = MovieLensDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# Model instantiation
n_users = len(user_map)
n_movies = len(movie_map)
model = RecSys(n_users, n_movies)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training
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

# ✅ New: Captum on Embedding Layer Outputs

# Set model to eval
model.eval()

# Choose a valid test sample
sample_user_id = test_dataset[0][0].item()
sample_movie_id = test_dataset[0][1].item()

# Fetch corresponding embeddings
user_embed = model.user_emb(torch.tensor([sample_user_id]))
movie_embed = model.movie_emb(torch.tensor([sample_movie_id]))

# Concatenate embeddings as input to final FC layer
input_embed = torch.cat([user_embed, movie_embed], dim=1)

# Baseline: zero embedding
baseline_embed = torch.zeros_like(input_embed)

# Define a new custom forward using the fc layer only
def embed_forward(emb_concat):
    return model.fc(emb_concat)

# Run Captum
ig = IntegratedGradients(embed_forward)
attributions = ig.attribute(inputs=input_embed, baselines=baseline_embed)

# Separate attributions for user and movie embeddings
user_attr = attributions[:, :model.user_emb.embedding_dim]
movie_attr = attributions[:, model.user_emb.embedding_dim:]

print("\n🔍 Captum Attribution Results (on embeddings):")
print("User Embedding Attribution:", user_attr)
print("Movie Embedding Attribution:", movie_attr)

