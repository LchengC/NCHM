import torch
from transformers import RobertaTokenizer
import numpy as np

# tokenize sentence and get event idx
def pes_get_batch(arg, node_data, indices, args):
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    new_tokens = ['<A_1>', '<A_2>', '<c1>', '</c1>', '<c2>', '</c2>']  # 50265 50266
    tokenizer.add_tokens(new_tokens)
    state_cau = []
    batch_idx = []
    batch_mask = []
    mask_indices = []
    label_b = []
    clabel_b = []
    for idx in indices:
        state_w_co = 0
        topic_id = arg[idx][1]
        document_id = arg[idx][2]
        e_1_id = arg[idx][3]
        e_2_id = arg[idx][4]
        label_1 = 1
        clabel_1 = 1
        e1_id = arg[idx][14]
        e2_id = arg[idx][15]
        sentence1_id = arg[idx][11]
        sentence2_id = arg[idx][13]
        s_1 = arg[idx][10]
        s_2 = arg[idx][12]
        s_1 = s_1.split()
        s_2 = s_2.split()
        event_node_co = node_data[topic_id][document_id][0]
        event_node = node_data[topic_id][document_id][1:]
        s_3 = ''
        e1_id = e1_id.split("_")
        e2_id = e2_id.split("_")
        ##########句内情况##########
        if sentence1_id == sentence2_id:
            # 文档存在共指
            clabel_1 = 0
            temp = 'In these sentences , '
            sent_dict = {}
            if e1_id[1] < e2_id[1]:
                for kkk in e1_id[1:]:
                    temp += s_1[int(kkk)] + ' '
                temp += '<mask> '
                for kkk in e2_id[1:]:
                    temp += s_2[int(kkk)] + ' '
            else:
                for kkk in e2_id[1:]:
                    temp += s_2[int(kkk)] + ' '
                temp += '<mask> '
                for kkk in e1_id[1:]:
                    temp += s_1[int(kkk)] + ' '
            temp += '.'
            sent_dict.update({sentence1_id: [s_1]})
            sent_dict[sentence1_id].append([(int(e1_id[1]),'<c1>'),(int(e1_id[-1])+1,'</c1>'),(int(e2_id[1]),'<c2>'),(int(e2_id[-1])+1,'</c2>')])
            if type(event_node_co) == list:
                s_3 = ''
                sentence_list = []
                event_list = []
                for j, event in enumerate(event_node_co):
                    event_t = event.copy()
                    # e1有共指
                    if event_t.count(e_1_id) > 0:
                        event_t.remove(e_1_id)
                        # e1与e2共指
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
                        # e1与e2不共指
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
                    # e2有共指
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
            #  文档内不存在共指关系
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
        ##########句间情况##########
        else:
            temp = 'In these sentences , '
            if sentence1_id < sentence2_id:
                for kkk in e1_id[1:]:
                    temp += s_1[int(kkk)] + ' '
                temp += '<mask> '
                for kkk in e2_id[1:]:
                    temp += s_2[int(kkk)] + ' '
            else:
                for kkk in e2_id[1:]:
                    temp += s_2[int(kkk)] + ' '
                temp += '<mask> '
                for kkk in e1_id[1:]:
                    temp += s_1[int(kkk)] + ' '
            temp += '.'
            s_3 = ''
            sent_dict = {}
            sent_dict.update({sentence1_id: [s_1]})
            sent_dict[sentence1_id].append([(int(e1_id[1]),'<c1>'),(int(e1_id[-1])+1,'</c1>')])
            sent_dict.update({sentence2_id: [s_2]})
            sent_dict[sentence2_id].append([(int(e2_id[1]),'<c2>'),(int(e2_id[-1]) + 1,'</c2>')])
            # 文档存在共指关系
            if type(event_node_co) == list:
                sentence_list = []
                event_list = []
                for j, event in enumerate(event_node_co):
                    # 判别e1是否存在共指
                    if event.count(e_1_id) > 0:
                        # 如果e1存在共指
                        event_t = event.copy()
                        event_t.remove(e_1_id)
                        # 句间e1,e2共指
                        if event_t.count(e_2_id) > 0:
                            del sent_dict[sentence1_id][1]
                            del sent_dict[sentence2_id][1]
                            sent_dict[sentence1_id].append([(int(e1_id[1]), '<c2> <c1>'), (int(e1_id[-1]) + 1, '</c1> </c2>')])
                            sent_dict[sentence2_id].append([(int(e2_id[1]), '<c2> <c1>'), (int(e2_id[-1]) + 1, '</c1> </c2>')])
                            state_w_co = 1
                            event_t.remove(e_2_id)
                            if len(event_t) > 0:
                                for e_co in event_t:
                                    sentence_id = event_node[e_co][7]
                                    event_id = event_node[e_co][3]
                                    event_pos = event_node[e_co][8]
                                    event_pos = event_pos.split("_")
                                    # 判别共指的事件所在句子是否已经被记录
                                    if sentence_id is not sentence1_id and sentence_id is not sentence2_id:
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
                        # e1存在共指,e1与e2不共指
                        else:
                            for e_co in event_t:
                                sentence_id = event_node[e_co][7]
                                event_id = event_node[e_co][3]
                                event_pos = event_node[e_co][8]
                                event_pos = event_pos.split("_")
                                # 判别共指的事件所在句子是否已经被记录
                                if sentence_id is not sentence1_id and sentence_id is not sentence2_id:
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
                    else:
                        # e2存在共指,e1与e2不共指
                        if event.count(e_2_id) > 0:
                            event_t = event.copy()
                            event_t.remove(e_2_id)
                            for e_co in event_t:
                                sentence_id = event_node[e_co][7]
                                event_id = event_node[e_co][3]
                                event_pos = event_node[e_co][8]
                                event_pos = event_pos.split("_")
                                # 判别共指事件所在句子是否已经被记录
                                if sentence_id is not sentence1_id and sentence_id is not sentence2_id:
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
                # e1,e2都不存在共指信息在其他句子
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
                # 处理共指信息
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
        if arg[idx][9] == 'NONE':
            label_1 = 0
        label_b.append(label_1)
        clabel_b.append(clabel_1)
        state_cau.append(state_w_co)
        if len(batch_idx) == 0:
            batch_idx = arg_1_idx
            batch_mask = arg_1_mask
            try:
                mask_indices = torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1]
                mask_indices = torch.unsqueeze(mask_indices, 0)
            except IndexError:
                print(encode_dict['input_ids'])
        else:
            batch_idx = torch.cat((batch_idx, arg_1_idx), dim=0)
            batch_mask = torch.cat((batch_mask, arg_1_mask), dim=0)
            try:
                mask_indices = torch.cat((mask_indices, torch.unsqueeze(torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1], 0)), dim=0)
            except IndexError:
                print(encode_dict['input_ids'])
                print(np.argwhere(np.array(encode_dict['input_ids']) == 50264))
    return batch_idx, batch_mask, mask_indices, label_b, clabel_b, state_cau