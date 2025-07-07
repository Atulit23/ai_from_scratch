import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())[0]
dropout_rate = 0.2

class GraphDataset:
    def __init__(self, X, y, edge_index, train_mask, val_mask, test_mask, max_dist=100, subset_size=None):
        if subset_size is not None:
            self.X = X[:subset_size]
            self.y = y[:subset_size]
            self.train_mask = train_mask[:subset_size]
            self.val_mask = val_mask[:subset_size]
            self.test_mask = test_mask[:subset_size]
        else:
            self.X = X
            self.y = y
            self.train_mask = train_mask
            self.val_mask = val_mask
            self.test_mask = test_mask

        self.edge_index = edge_index
        self.num_nodes = self.X.shape[0]
        self.max_dist = max_dist
        self.adj = [[] for _ in range(self.num_nodes)]
        self.distance_matrix = [[max_dist] * self.num_nodes for _ in range(self.num_nodes)]
        self.node_degrees = []

    def _create_adj(self):
        for src, dst in self.edge_index.t().tolist():
            if src < self.num_nodes and dst < self.num_nodes:
                self.adj[src].append(dst)
                self.adj[dst].append(src)

    def _floyd_warshall(self):
        for i in range(self.num_nodes):
            self.distance_matrix[i][i] = 0
            for neighbor in self.adj[i]:
                self.distance_matrix[i][neighbor] = 1

        for k in range(self.num_nodes):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if self.distance_matrix[i][j] > self.distance_matrix[i][k] + self.distance_matrix[k][j]:
                        self.distance_matrix[i][j] = self.distance_matrix[i][k] + self.distance_matrix[k][j]

    def _get_degrees(self):
        for neighbors in self.adj:
            self.node_degrees.append(len(neighbors))

    def get_graph_data(self):
        self._create_adj()
        self._floyd_warshall()
        self._get_degrees()
        return {
            "X": self.X,
            "y": self.y,
            "adj": self.adj,
            "distance_matrix": torch.LongTensor(self.distance_matrix),
            "node_degrees": self.node_degrees,
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
            "test_mask": self.test_mask,
        }

class GraphAttentionHead(nn.Module):
    def __init__(self, d_in, d_out, max_dist):
        super().__init__()
        self.w_q = nn.Linear(d_in, d_out, bias=False)
        self.w_k = nn.Linear(d_in, d_out, bias=False)
        self.w_v = nn.Linear(d_in, d_out, bias=False)
        self.distance_embedding = nn.Embedding(max_dist + 2, 1)

    def forward(self, X, dist_matrix):
        distance_bias = self.distance_embedding(dist_matrix).squeeze(-1)
        q = self.w_q(X)
        k = self.w_k(X)
        v = self.w_v(X)
        scores = (q @ k.T) * (k.shape[-1] ** -0.5) + distance_bias
        attn_wts = F.softmax(scores, dim=-1)
        attn_wts = F.dropout(attn_wts, p=dropout_rate, training=self.training)
        return attn_wts @ v

class GraphMultiAttentionHead(nn.Module):
    def __init__(self, num_heads, d_in, d_out, max_dist):
        super().__init__()
        self.heads = nn.ModuleList([
            GraphAttentionHead(d_in, d_out, max_dist)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * d_out, d_out)

    def forward(self, X, dist_matrix):
        out = torch.cat([h(X, dist_matrix) for h in self.heads], dim=-1)
        out = self.proj(out)
        return F.dropout(out, p=dropout_rate, training=self.training)

class FeedForward(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_out, 4 * d_out, bias=False),
            nn.ReLU(),
            nn.Linear(4 * d_out, d_out, bias=False),
            nn.Dropout(dropout_rate)
        )

    def forward(self, X):
        return self.net(X)

class Block(nn.Module):
    def __init__(self, num_heads, d_model, max_dist):
        super().__init__()
        self.attn = GraphMultiAttentionHead(num_heads, d_model, d_model, max_dist)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X, dist_matrix):
        X = self.norm1(self.attn(X, dist_matrix) + X)
        X = self.norm2(self.ffn(X) + X)
        return X

class GraphTransformer(nn.Module):
    def __init__(self, max_dist, num_heads, d_model, num_classes, num_layers, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            Block(num_heads, d_model, max_dist)
            for _ in range(num_layers)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, X, dist_matrix):
        X = self.input_proj(X)
        for block in self.blocks:
            X = block(X, dist_matrix)
        return self.mlp(X)

# Hyperparams and training
subset_size = 3000
max_dist = 100
num_heads = 4
d_model = 256
num_classes = 7
num_layers = 2
lr = 1e-3
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gd = GraphDataset(
    dataset.x, dataset.y, dataset.edge_index,
    dataset.train_mask, dataset.val_mask, dataset.test_mask,
    max_dist=max_dist,
    subset_size=subset_size
)
graph_data = gd.get_graph_data()

X = graph_data["X"].to(device).float()
y = graph_data["y"].to(device).long()
dist_matrix = graph_data["distance_matrix"].to(device).long()
train_mask = graph_data["train_mask"].to(device)

model = GraphTransformer(max_dist, num_heads, d_model, num_classes, num_layers, dataset.num_features).to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    predictions = model(X, dist_matrix)
    loss = F.cross_entropy(predictions[train_mask], y[train_mask])
    optim.zero_grad()
    loss.backward()
    optim.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

torch.save(model.state_dict(), 'model.pth')

model.eval()
with torch.no_grad():
    out = model(X, dist_matrix)
    pred = out.argmax(dim=1)

    val_correct = (pred[graph_data["val_mask"]] == y[graph_data["val_mask"]]).sum().item()
    val_total = graph_data["val_mask"].sum().item()
    val_acc = val_correct / val_total

    test_correct = (pred[graph_data["test_mask"]] == y[graph_data["test_mask"]]).sum().item()
    test_total = graph_data["test_mask"].sum().item()
    test_acc = test_correct / test_total

    print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
