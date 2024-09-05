# tests

def test1():
    # conda install -y pytorch-geometric -c rusty1s -c conda-forge

    import torch
    from torch_geometric.data import Data

    # [2, number_edges], edge = (node_idx1, node_idx2), e.g. e = (0,1) = (n0, n1) (note this is reflected on the type torch_uu long)

    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)

    # features to each node [num_nodes, D]

    x = torch.tensor([[0.0], [-1.0], [1.0]])

    data = Data(x=x, edge_index=edge_index)
    print(data)

    # https://discuss.pytorch.org/t/pytorch-geometric/44994
    # https://stackoverflow.com/questions/61274847/how-to-visualize-a-torch-geometric-graph-in-python

    import networkx as nx
    from torch_geometric.utils.convert import to_networkx

    g = to_networkx(data)
    nx.draw(g)
    pass

if __name__ == '__main__':
    test1()
    print("Done\a")