import torch
from torch import Tensor
from torch_geometric.nn.conv import GCNConv
import torch.nn.functional as F


# from torch_geometric.datasets import Planetoid

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        F.log_softmax(x, dim=1)
        return x


def main():
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(dataset.num_node_features, 16, 32).to(device)
    print(model)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()  # モデルを訓練モードにする。
    print("data: ", data)
    print("data x: ", data.x.size())

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
