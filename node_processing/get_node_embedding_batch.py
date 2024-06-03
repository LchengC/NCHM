import torch

def get_node_embedding_batch(self, node_arg, node_mask, n, batch_size, event_idx, embedding_size, device):
    node_embedding = torch.tensor([]).to(device)
    all_indices = torch.arange(n).split(batch_size)
    t = 0
    for k, batch_indices in enumerate(all_indices):
        length = len(batch_indices)
        node_embed = self.roberta_model(node_arg[t:t + length], node_mask[t:t + length])[0].to(device)
        for j in range(length):
            e1_start = event_idx[t + j][0]
            e1_end = event_idx[t + j][1]
            e1_embed = torch.zeros(1, embedding_size).to(device)
            e1_num = e1_end - e1_start
            for i in range(e1_start, e1_end):
                e1_embed += node_embed[j][i]
            event_embed = e1_embed / e1_num
            if k == 0:
                node_embedding = event_embed
            else:
                node_embedding = torch.cat((node_embedding, event_embed))
        t += length
        del node_embed
    return node_embedding