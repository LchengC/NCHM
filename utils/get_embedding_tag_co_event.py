import torch
from transformers import RobertaTokenizer
from utils.delete_tokens import delete_tokens
import numpy as np

# tokenize sentence and get event idx
def get_batch(arg, node_data, indices, args):
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    new_tokens = ['<A_1>', '<A_2>', '<c1>', '</c1>', '<c2>', '</c2>']  # 50265 50266
    tokenizer.add_tokens(new_tokens)
    state_cau = []
    batch_idx = []
    batch_mask = []
    mask_indices = []
    batch_event = []
    batch_event_id = []
    label_b = []
    clabel_b = []
    for idx in arg[indices]:
        state_w_co = 0
        topic_id = idx[1]
        document_id = idx[2]
        e_1_id = idx[3]
        e_2_id = idx[4]
        m_id = torch.tensor([[e_1_id, e_2_id]])
        label_1 = 1
        clabel_1 = 1
        e1_id = idx[14]
        e2_id = idx[15]
        sentence1_id = idx[11]
        sentence2_id = idx[13]
        s_1 = idx[10]
        s_2 = idx[12]
        s_1 = s_1.split()
        s_2 = s_2.split()
        event_node_co = node_data[topic_id][document_id][0]
        event_node = node_data[topic_id][document_id][1:]
        e1_id = e1_id.split("_")
        e2_id = e2_id.split("_")
        ##########intra##########
        if sentence1_id == sentence2_id:
            # exist a co-reference
            clabel_1 = 0
            sent_dict = {}
            temp = 'In these sentences ,<s> '
            if e1_id[1] < e2_id[1]:
                for kkk in e1_id[1:]:
                    temp += s_1[int(kkk)] + ' '
                temp += '<s> <mask><s> '
                for kkk in e2_id[1:]:
                    temp += s_2[int(kkk)] + ' '
                front = 1
            else:
                for kkk in e2_id[1:]:
                    temp += s_2[int(kkk)] + ' '
                temp += '<s> <mask><s> '
                for kkk in e1_id[1:]:
                    temp += s_1[int(kkk)] + ' '
                front = 2
            temp += '<s> .'
            temp = temp.replace(' <s>', '<s>')
            sent_dict.update({sentence1_id: [s_1]})
            sent_dict[sentence1_id].append([(int(e1_id[1]),'<c1>'),(int(e1_id[-1])+1,'</c1>'),(int(e2_id[1]),'<c2>'),(int(e2_id[-1])+1,'</c2>')])
            if type(event_node_co) == list:
                s_3 = ''
                sentence_list = []
                event_list = []
                for j, event in enumerate(event_node_co):
                    event_t = event.copy()
                    # e1 exist co-reference
                    if event_t.count(e_1_id) > 0:
                        event_t.remove(e_1_id)
                        # e1 and e2 exist co-reference
                        if event_t.count(e_2_id) > 0:
                            del sent_dict[sentence1_id][1]
                            sent_dict[sentence1_id].append([(int(e1_id[1]), '<c2> <c1>'), (int(e1_id[-1]) + 1, '</c1> </c2>'), (int(e2_id[1]), '<c2> <c1>'),(int(e2_id[-1]) + 1, '</c1> </c2>')])
                            state_w_co = 1
                            event_t.remove(e_2_id)
                            if len(event_t) > 0:
                                for e_co in event_t:
                                    sentence_id = event_node[e_co][7]
                                    event_id = event_node[e_co][3]
                                    event_pos = event_node[e_co][8]
                                    event_pos = event_pos.split("_")
                                    if sentence_id is not sentence1_id:
                                        if sentence_id not in sentence_list:
                                            sent = event_node[e_co][6]
                                            sent = sent.split()
                                            sentence_list.append(sentence_id)
                                            event_list.append(event_id)
                                            sent_dict.update({sentence_id: [sent]})
                                            sent_dict[sentence_id].append([(int(event_pos[1]), '<c2> <c1>'), (int(event_pos[-1]) + 1, '</c1> </c2>')])
                                        else:
                                            sent_dict[sentence_id][1].append((int(event_pos[1]), '<c2> <c1>'))
                                            sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c1> </c2>'))
                                    else:
                                        sent_dict[sentence_id][1].append((int(event_pos[1]), '<c2> <c1>'))
                                        sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c1> </c2>'))
                            else:
                                break
                        # e1 and e2 don't exist co-reference
                        else:
                            for e_co in event_t:
                                sentence_id = event_node[e_co][7]
                                event_id = event_node[e_co][3]
                                event_pos = event_node[e_co][8]
                                event_pos = event_pos.split("_")
                                if sentence_id is not sentence1_id:
                                    if sentence_id not in sentence_list:
                                        sent = event_node[e_co][6]
                                        sent = sent.split()
                                        sentence_list.append(sentence_id)
                                        event_list.append(event_id)
                                        sent_dict.update({sentence_id: [sent]})
                                        sent_dict[sentence_id].append([(int(event_pos[1]), '<c1>'), (int(event_pos[-1]) + 1, '</c1>')])
                                    else:
                                        sent_dict[sentence_id][1].append((int(event_pos[1]), '<c1>'))
                                        sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c1>'))
                                else:
                                    sent_dict[sentence_id][1].append((int(event_pos[1]), '<c1>'))
                                    sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c1>'))
                    # e2 exist co-reference
                    if event_t.count(e_2_id) > 0:
                        event_t.remove(e_2_id)
                        for e_co in event_t:
                            sentence_id = event_node[e_co][7]
                            event_id = event_node[e_co][3]
                            event_pos = event_node[e_co][8]
                            event_pos = event_pos.split("_")
                            if sentence_id is not sentence2_id:
                                if sentence_id not in sentence_list:
                                    sent = event_node[e_co][6]
                                    sent = sent.split()
                                    sentence_list.append(sentence_id)
                                    event_list.append(event_id)
                                    sent_dict.update({sentence_id: [sent]})
                                    sent_dict[sentence_id].append([(int(event_pos[1]), '<c2>'), (int(event_pos[-1]) + 1, '</c2>')])
                                else:
                                    sent_dict[sentence_id][1].append((int(event_pos[1]), '<c2>'))
                                    sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c2>'))
                            else:
                                sent_dict[sentence_id][1].append((int(event_pos[1]), '<c2>'))
                                sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c2>'))
                length_event = len(sent_dict)
                new_sent = sorted(sent_dict)
                for i, sent_co in enumerate(new_sent):
                    index = sent_dict[sent_co][1]
                    index = sorted(index, reverse=True)
                    for pos in index:
                        sent_dict[sent_co][0].insert(pos[0], pos[1])
                    temp_sent = " ".join(sent_dict[sent_co][0])
                    if length_event == i + 1:
                        s_3 += temp_sent
                    else:
                        s_3 += temp_sent + ' </s> '
                encode_dict = tokenizer.encode_plus(
                    s_3,
                    text_pair=temp,
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=args.len_arg,
                    truncation=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
            #  don't exist a co-reference
            else:
                if int(e1_id[1]) > int(e2_id[1]):
                    s_1.insert(int(e1_id[1]), '<c1>')
                    s_1.insert(int(e1_id[1]) + len(e1_id), '</c1>')
                    s_1.insert(int(e2_id[1]), '<c2>')
                    s_1.insert(int(e2_id[1]) + len(e2_id), '</c2>')
                else:
                    s_1.insert(int(e2_id[1]), '<c2>')
                    s_1.insert(int(e2_id[1]) + len(e2_id), '</c2>')
                    s_1.insert(int(e1_id[1]), '<c1>')
                    s_1.insert(int(e1_id[1]) + len(e1_id), '</c1>')
                s_1 = " ".join(s_1)
                encode_dict = tokenizer.encode_plus(
                    s_1,
                    text_pair=temp,
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=args.len_arg,
                    truncation=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
        ##########inter##########
        else:
            temp = 'In these sentences ,<s> '
            if sentence1_id < sentence2_id:
                for kkk in e1_id[1:]:
                    temp += s_1[int(kkk)] + ' '
                temp += '<s> <mask><s> '
                for kkk in e2_id[1:]:
                    temp += s_2[int(kkk)] + ' '
                front = 1
            else:
                for kkk in e2_id[1:]:
                    temp += s_2[int(kkk)] + ' '
                temp += '<s> <mask><s> '
                for kkk in e1_id[1:]:
                    temp += s_1[int(kkk)] + ' '
                front = 2
            temp += '<s> .'
            temp = temp.replace(' <s>', '<s>')
            s_3 = ''
            sent_dict = {}
            sent_dict.update({sentence1_id: [s_1]})
            sent_dict[sentence1_id].append([(int(e1_id[1]),'<c1>'),(int(e1_id[-1])+1,'</c1>')])
            sent_dict.update({sentence2_id: [s_2]})
            sent_dict[sentence2_id].append([(int(e2_id[1]),'<c2>'),(int(e2_id[-1]) + 1,'</c2>')])
            # exist a co-reference
            if type(event_node_co) == list:
                sentence_list = []
                event_list = []
                for j, event in enumerate(event_node_co):
                    # whether e1 exist a co-reference
                    if event.count(e_1_id) > 0:
                        # if e1 exist a co-reference
                        event_t = event.copy()
                        event_t.remove(e_1_id)
                        # inter e1, e2 exist a co-reference
                        if event_t.count(e_2_id) > 0:
                            del sent_dict[sentence1_id][1]
                            del sent_dict[sentence2_id][1]
                            sent_dict[sentence1_id].append(
                                [(int(e1_id[1]), '<c2> <c1>'), (int(e1_id[-1]) + 1, '</c1> </c2>')])
                            sent_dict[sentence2_id].append(
                                [(int(e2_id[1]), '<c2> <c1>'), (int(e2_id[-1]) + 1, '</c1> </c2>')])
                            state_w_co = 1
                            event_t.remove(e_2_id)
                            if len(event_t) > 0:
                                for e_co in event_t:
                                    sentence_id = event_node[e_co][7]
                                    event_id = event_node[e_co][3]
                                    event_pos = event_node[e_co][8]
                                    event_pos = event_pos.split("_")
                                    # Determine whether the sentence in which the co-referenced event is located has been recorded
                                    if sentence_id is not sentence1_id and sentence_id is not sentence2_id:
                                        if sentence_id not in sentence_list:
                                            sent = event_node[e_co][6]
                                            sent = sent.split()
                                            sentence_list.append(sentence_id)
                                            event_list.append(event_id)
                                            sent_dict.update({sentence_id: [sent]})
                                            sent_dict[sentence_id].append([(int(event_pos[1]), '<c2> <c1>'),
                                                                           (int(event_pos[-1]) + 1, '</c1> </c2>')])
                                        else:
                                            sent_dict[sentence_id][1].append((int(event_pos[1]), '<c2> <c1>'))
                                            sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c1> </c2>'))
                                    else:
                                        sent_dict[sentence_id][1].append((int(event_pos[1]), '<c2> <c1>'))
                                        sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c1> </c2>'))
                            else:
                                break
                        # e1 exist a co-reference, but e1 do not refer to e2
                        else:
                            for e_co in event_t:
                                sentence_id = event_node[e_co][7]
                                event_id = event_node[e_co][3]
                                event_pos = event_node[e_co][8]
                                event_pos = event_pos.split("_")
                                # Determine whether the sentence in which the co-referenced event is located has been recorded
                                if sentence_id is not sentence1_id and sentence_id is not sentence2_id:
                                    if sentence_id not in sentence_list:
                                        sent = event_node[e_co][6]
                                        sent = sent.split()
                                        sentence_list.append(sentence_id)
                                        event_list.append(event_id)
                                        sent_dict.update({sentence_id: [sent]})
                                        sent_dict[sentence_id].append(
                                            [(int(event_pos[1]), '<c1>'), (int(event_pos[-1]) + 1, '</c1>')])
                                    else:
                                        sent_dict[sentence_id][1].append((int(event_pos[1]), '<c1>'))
                                        sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c1>'))
                                else:
                                    sent_dict[sentence_id][1].append((int(event_pos[1]), '<c1>'))
                                    sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c1>'))
                    else:
                        # e2 exist a co-reference, but e2 do not refer to e1
                        if event.count(e_2_id) > 0:
                            event_t = event.copy()
                            event_t.remove(e_2_id)
                            for e_co in event_t:
                                sentence_id = event_node[e_co][7]
                                event_id = event_node[e_co][3]
                                event_pos = event_node[e_co][8]
                                event_pos = event_pos.split("_")
                                # Determine whether the sentence in which the co-referenced event is located has been recorded
                                if sentence_id is not sentence1_id and sentence_id is not sentence2_id:
                                    if sentence_id not in sentence_list:
                                        sent = event_node[e_co][6]
                                        sent = sent.split()
                                        sentence_list.append(sentence_id)
                                        event_list.append(event_id)
                                        sent_dict.update({sentence_id: [sent]})
                                        sent_dict[sentence_id].append(
                                            [(int(event_pos[1]), '<c2>'), (int(event_pos[-1]) + 1, '</c2>')])
                                    else:
                                        sent_dict[sentence_id][1].append((int(event_pos[1]), '<c2>'))
                                        sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c2>'))
                                else:
                                    sent_dict[sentence_id][1].append((int(event_pos[1]), '<c2>'))
                                    sent_dict[sentence_id][1].append((int(event_pos[-1]) + 1, '</c2>'))

                # e1,e2 neither exists coreferential information in other sentences
                if len(event_list) == 0:
                    s_1.insert(int(e1_id[1]), '<c1>')
                    s_1.insert(int(e1_id[1]) + len(e1_id), '</c1>')
                    s_2.insert(int(e2_id[1]), '<c2>')
                    s_2.insert(int(e2_id[1]) + len(e2_id), '</c2>')
                    s_1 = " ".join(s_1)
                    s_2 = " ".join(s_2)
                    if sentence1_id < sentence2_id:
                        encode_dict = tokenizer.encode_plus(
                            s_1 + ' </s> ' + s_2,
                            text_pair=temp,
                            add_special_tokens=True,
                            padding='max_length',
                            max_length=args.len_arg,
                            truncation=True,
                            pad_to_max_length=True,
                            return_attention_mask=True,
                            return_tensors='pt'
                        )
                    else:
                        encode_dict = tokenizer.encode_plus(
                            s_2 + ' </s> ' + s_1,
                            text_pair=temp,
                            add_special_tokens=True,
                            padding='max_length',
                            max_length=args.len_arg,
                            truncation=True,
                            pad_to_max_length=True,
                            return_attention_mask=True,
                            return_tensors='pt'
                        )
                # Process co-reference
                else:
                    length_event = len(sent_dict)
                    new_sent = sorted(sent_dict)
                    for i, sent_co in enumerate(new_sent):
                        index = sent_dict[sent_co][1]
                        index = sorted(index, reverse=True)
                        for pos in index:
                            sent_dict[sent_co][0].insert(pos[0], pos[1])
                        temp_sent = " ".join(sent_dict[sent_co][0])
                        if length_event == i + 1:
                            s_3 += temp_sent
                        else:
                            s_3 += temp_sent + ' </s> '
                    encode_dict = tokenizer.encode_plus(
                        s_3,
                        text_pair=temp,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=args.len_arg,
                        truncation=True,
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )
            else:
                s_1.insert(int(e1_id[1]), '<c1>')
                s_1.insert(int(e1_id[1]) + len(e1_id), '</c1>')
                s_2.insert(int(e2_id[1]), '<c2>')
                s_2.insert(int(e2_id[1]) + len(e2_id), '</c2>')
                s_1 = " ".join(s_1)
                s_2 = " ".join(s_2)
                if sentence1_id < sentence2_id:
                    encode_dict = tokenizer.encode_plus(
                        s_1 + ' </s> ' + s_2,
                        text_pair=temp,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=args.len_arg,
                        truncation=True,
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )
                else:
                    encode_dict = tokenizer.encode_plus(
                        s_2 + ' </s> ' + s_1,
                        text_pair=temp,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=args.len_arg,
                        truncation=True,
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )
        arg_1_idx = encode_dict['input_ids']
        arg_1_mask = encode_dict['attention_mask']
        if front == 1:
            arg_1_idx, arg_1_mask, v1 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
            arg_1_idx, arg_1_mask, v2 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
            arg_1_idx, arg_1_mask, v3 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
            arg_1_idx, arg_1_mask, v4 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
        else:
            arg_1_idx, arg_1_mask, v3 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
            arg_1_idx, arg_1_mask, v4 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
            arg_1_idx, arg_1_mask, v1 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
            arg_1_idx, arg_1_mask, v2 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
        arg_e_idx1 = torch.tensor([[v1, v2, v3, v4]])
        if idx[9] == 'NONE':
            label_1 = 0
        label_b.append(label_1)
        clabel_b.append(clabel_1)
        state_cau.append(state_w_co)
        if len(batch_idx) == 0:
            batch_idx = arg_1_idx
            batch_mask = arg_1_mask
            batch_event = arg_e_idx1
            batch_event_id = m_id
            try:
                mask_indices = torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1]
                mask_indices = torch.unsqueeze(mask_indices, 0)
            except IndexError:
                print(encode_dict['input_ids'])
        else:
            batch_idx = torch.cat((batch_idx, arg_1_idx), dim=0)
            batch_mask = torch.cat((batch_mask, arg_1_mask), dim=0)
            batch_event = torch.cat((batch_event, arg_e_idx1), dim=0)
            batch_event_id = torch.cat((batch_event_id, m_id), dim=0)
            try:
                mask_indices = torch.cat(
                    (mask_indices, torch.unsqueeze(torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1], 0)), dim=0)
            except IndexError:
                print(encode_dict['input_ids'])
                print(np.argwhere(np.array(encode_dict['input_ids']) == 50264))
    return batch_idx, batch_mask, batch_event, mask_indices, batch_event_id, label_b, clabel_b, state_cau