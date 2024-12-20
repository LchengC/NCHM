import torch

# extract the embedding of the event
def extract_node(embed, event_idx, embedding_size, device):
    e1_start = event_idx[0]
    e1_end = event_idx[1]
    e1_embed = torch.zeros(1, embedding_size).to(device)
    e1_num = e1_end - e1_start
    for i in range(e1_start, e1_end):
        e1_embed += embed[i]
    event_embed = e1_embed / e1_num
    return event_embed