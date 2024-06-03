import torch

# extract the embedding of the pair of events
def extract_event_pair(embed, event_idx, embedding_size, device):
    e1_start = event_idx[0]
    e1_end = event_idx[1]
    e2_start = event_idx[2]
    e2_end = event_idx[3]
    e1_embed = torch.zeros(1, embedding_size).to(device)
    e2_embed = torch.zeros(1, embedding_size).to(device)
    e1_num = e1_end - e1_start
    e2_num = e2_end - e2_start
    for i in range(e1_start, e1_end):
        e1_embed += embed[i]
    for j in range(e2_start, e2_end):
        e2_embed += embed[j]
    e1_embed = e1_embed / e1_num
    e2_embed = e2_embed / e2_num
    event_embed = torch.cat((e1_embed, e2_embed), dim=1).to(device)
    return event_embed