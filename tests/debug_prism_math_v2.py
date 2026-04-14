import torch
import torch.nn.functional as F

torch.manual_seed(42)
# 40 redundant vectors (clustered)
clustered = torch.randn(40, 64) * 0.05 + torch.tensor([1.0] + [0.0]*63)
# 20 diverse vectors (spread out)
diverse = torch.randn(20, 64) * 2.0

embeddings = torch.cat([clustered, diverse], dim=0)

# PRISM Steps
mean_vec = embeddings.mean(dim=0, keepdim=True)
embeddings_centered = embeddings - mean_vec

# Important: Dot product vs Cosine similarity
# PRISM uses raw dot products on centered data. Magnitude matters!
corr = torch.matmul(embeddings_centered, embeddings_centered.T)
row_sums = corr.sum(dim=1)

print(f"Mean magnitude of clustered after centering: {embeddings_centered[:40].norm(dim=1).mean():.4f}")
print(f"Mean magnitude of diverse after centering: {embeddings_centered[40:].norm(dim=1).mean():.4f}")

print(f"\nRow Sums (first 5 clustered): {row_sums[:5].tolist()}")
print(f"Row Sums (first 5 diverse): {row_sums[40:45].tolist()}")

min_val = row_sums.min()
max_val = row_sums.max()
norm_scores = (row_sums - min_val) / (max_val - min_val + 1e-9)

q1 = torch.quantile(norm_scores, 0.333).item()
indices = [i for i, s in enumerate(norm_scores.tolist()) if s <= q1]
print(f"\nQ1: {q1}")
print(f"Selected Indices: {indices}")
