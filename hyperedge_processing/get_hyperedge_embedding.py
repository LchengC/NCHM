import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_hyperedge_embedding(node_new_emb, hyperedge_index):
    hyperedge_embedding = torch.tensor([]).to(device)
    if hyperedge_index.numel() > 0:
        num_edges = int(hyperedge_index[1].max()) + 1
        for idx in range(num_edges):
            edge_id = torch.nonzero(hyperedge_index[1] == idx, as_tuple=False)
            num_node = int(edge_id.size()[0])
            hyperedge_embed = torch.zeros_like(node_new_emb[[0]]).to(device)
            for i in edge_id:
                node_id = hyperedge_index[0][i]
                hyperedge_embed += node_new_emb[node_id]
            hyperedge_embed /= num_node
            if idx == 0:
                hyperedge_embedding = hyperedge_embed
            else:
                hyperedge_embedding = torch.cat((hyperedge_embedding, hyperedge_embed),0)
    return hyperedge_embedding





