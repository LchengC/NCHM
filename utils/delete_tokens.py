import torch

# delete tab
def delete_tokens(arg_idx, arg_mask, index, idx):
    arg_idx = torch.LongTensor(arg_idx)
    temp = torch.nonzero(arg_idx == idx, as_tuple=False)
    indices = temp[index][1]
    arg_i = torch.cat((arg_idx[0][0:indices], arg_idx[0][indices + 1:]))
    arg_i = torch.unsqueeze(arg_i, dim=0)
    arg_m = torch.cat((arg_mask[0][0:indices], arg_mask[0][indices + 1:]))
    arg_m = torch.unsqueeze(arg_m, dim=0)
    return arg_i, arg_m, indices