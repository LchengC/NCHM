import torch
from transformers import RobertaTokenizer
from utils import delete_tokens

def get_node(arg, args):
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    node_idx = []
    node_mask = []
    node_event = []
    event_id_list = []
    event_node_co = arg[0]
    event_node = arg[1:]
    length = len(event_node)
    event_id = list(range(length))
    temp = [None]*length
    z = 0
    if event_node_co[0] != 'NONE':
        for i in range(length):
            if temp[i] == None:
                for j, event_co in enumerate(event_node_co):
                    if event_co.count(i) > 0:
                        for event in event_co:
                            temp[event] = i-z
                            assert i-z >= 0
                temp[i] = i-z
            else:
                z += 1
        for k in temp:
            assert k is not None
        event_id_list.append(event_id)
        event_id_list.append(temp)
    else:
        event_id_list.append(event_id)
        event_id_list.append(event_id)
    for idx, event_node_old in enumerate(event_node):
        e_id = event_node_old[8]
        e_id = e_id.split("_")
        s = event_node_old[6]
        s = s.split()
        s.insert(int(e_id[1]), '<s>')
        s.insert(int(e_id[1]) + len(e_id), '<s>')
        s = " ".join(s)
        s = s.replace(' <s>', '<s>')
        if int(e_id[1]) == 0:
            s = s.replace('<s> ', '<s>', 1)
        encode_dict = tokenizer.encode_plus(
            s,
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg_node,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        arg_1_idx = encode_dict['input_ids']
        arg_1_mask = encode_dict['attention_mask']


        arg_1_idx, arg_1_mask, v1 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
        arg_1_idx, arg_1_mask, v2 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
        arg_e_idx1 = torch.tensor([[v1, v2]])
        if len(node_idx) == 0:
            node_idx = arg_1_idx
            node_mask = arg_1_mask
            node_event = arg_e_idx1
        else:
            node_idx = torch.cat((node_idx, arg_1_idx), dim=0)
            node_mask = torch.cat((node_mask, arg_1_mask), dim=0)
            node_event = torch.cat((node_event, arg_e_idx1), dim=0)
    return node_idx, node_mask, node_event, event_id_list