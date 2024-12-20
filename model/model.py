# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForMaskedLM, RobertaModel
from hgnn import HypergraphConv
from hyperedge_processing import get_hyperedge_embedding
from node_processing import extract_node
from event_processing import extract_event_pair, extract_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size = 768

class PES(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.roberta_model = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)
        self.roberta_model.resize_token_embeddings(args.vocab_size)
        for param in self.roberta_model.parameters():
            param.requires_grad = True

    def forward(self, batch_arg, arg_mask, mask_indices, batch_size):
        sent_emb = self.roberta_model(batch_arg, arg_mask)[0].to(device)
        event_pair_embed = torch.tensor([]).to(device)
        for i in range(batch_size):
            e_emb = self.extract_event(sent_emb[i], mask_indices[i]).to(device)
            if i == 0:
                event_pair_embed = e_emb
            else:
                event_pair_embed = torch.cat((event_pair_embed, e_emb))
            del e_emb
        token_id = [50265, 50266]
        prediction = event_pair_embed[:, token_id]
        return prediction

    def extract_event(self, embed, mask_idx):
        mask_embed = torch.zeros(1, embedding_size).to(device)
        mask_embed = embed[mask_idx]
        mask_embed = torch.unsqueeze(mask_embed, 0)
        return mask_embed

class NCHM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.roberta_model = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)
        self.roberta_model.resize_token_embeddings(args.vocab_size)
        for param in self.roberta_model.parameters():
            param.requires_grad = False

        self.roberta_model1 = RobertaModel.from_pretrained(args.model_name).to(device)
        for param in self.roberta_model1.parameters():
            param.requires_grad = True

        self.HGNN1 = HypergraphConv(in_channels=embedding_size,
                                    out_channels=args.n_hid,
                                    use_attention=args.attention,
                                    heads=args.n_head,
                                    concat=args.concat,
                                    negative_slope=0.2,
                                    dropout=args.drop_out
                                    )
        self.HGNN2 = HypergraphConv(in_channels=args.head_number * args.n_hid,
                                    out_channels=args.n_last,
                                    use_attention=args.attention,
                                    heads=args.n_head,
                                    concat=args.concat,
                                    negative_slope=0.2,
                                    dropout=args.drop_out
                                    )
        self.mlp = nn.Sequential(nn.Linear(2 * embedding_size + 2 * args.head_number * args.n_last, args.mlp_size),
                                  nn.ReLU(), nn.Dropout(args.drop_out),
                                  nn.Linear(args.mlp_size, 2)).to(device)

    def forward(self, event_arg, event_mask, event_e, mask_indices, event_id, node_arg, node_mask, node_e, document_size, label, event_co_id):
        # Gets the representation vector of the event for the sentence in which it resides
        event_pair = torch.tensor([[]]).to(device)
        mask_embed = torch.tensor([]).to(device)
        # get the embedding of the pair of events
        for j in range(document_size):
            event_arg_t = torch.unsqueeze(event_arg[j], dim=0)
            event_mask_t = torch.unsqueeze(event_mask[j], dim=0)
            sent_emb = self.roberta_model.roberta(event_arg_t, attention_mask=event_mask_t, output_hidden_states=True)[0].to(device)
            sent_emb_mask = self.roberta_model(event_arg_t, event_mask_t)[0].to(device)
            e_emb = extract_event_pair(sent_emb[0], event_e[j], embedding_size, device)
            mask = extract_mask(sent_emb_mask[0], mask_indices[j], embedding_size, device)
            if j == 0:
                event_pair_embed = e_emb
                mask_embed = mask
            else:
                event_pair_embed = torch.cat((event_pair_embed, e_emb))
                mask_embed = torch.cat((mask_embed, mask))
            del sent_emb
            del sent_emb_mask, mask
        token_id = [50265, 50266]
        mask_pre = mask_embed[:, token_id]
        del mask_embed
        n_s_emb = self.roberta_model1(node_arg, node_mask)[0].to(device)
        node_emb = torch.tensor([[]]).to(device)
        node_new_emb = torch.tensor([[]]).to(device)
        # get the embedding of the node
        node_size = len(event_co_id[0])
        for i in range(node_size):
            n_e_emb = extract_node(n_s_emb[i], node_e[i], embedding_size, device)
            if i == 0:
                node_emb = n_e_emb
            else:
                node_emb = torch.cat((node_emb, n_e_emb))
        del n_s_emb
        num_new_nodes = int(event_co_id[1].max()) + 1
        new_node_id = event_co_id[1]
        for j in range(num_new_nodes):
            node_emb_temp = torch.zeros_like(n_e_emb).to(device)
            temp = torch.nonzero(new_node_id == j, as_tuple=False).to(device)
            num_co = temp.size()[0]
            assert num_co != 0
            if num_co > 1:
                for k in temp:
                    node_emb_temp += node_emb[k]
                node_emb_temp = node_emb_temp / num_co
            else:
                node_emb_temp = node_emb[temp[0]]
            if j == 0:
                node_new_emb = node_emb_temp
            else:
                node_new_emb = torch.cat((node_new_emb, node_emb_temp))
        del node_emb
        # get hyperedge_index
        hyperedge_index = self.get_hyperedge_index(mask_pre, event_id, event_co_id[1], label).to(device)
        # get hyperedge embedding
        hyperedge_embedding = get_hyperedge_embedding(node_new_emb, hyperedge_index)
        # HypergraphConv
        node_new_emb = self.HGNN1(node_new_emb, hyperedge_index, None, hyperedge_embedding)
        node_new_emb = F.relu(node_new_emb)
        node_new_emb = F.dropout(node_new_emb, training=self.training)
        hyperedge_embedding = get_hyperedge_embedding(node_new_emb, hyperedge_index)
        node_new_emb = self.HGNN2(node_new_emb, hyperedge_index, None, hyperedge_embedding)

        # concate the event embedding
        for k in range(len(event_id)):
            e_1 = event_id[k][0]
            e_2 = event_id[k][1]
            e_1_co = new_node_id[e_1]
            e_2_co = new_node_id[e_2]
            event_pair_ = torch.cat((node_new_emb[e_1_co] + node_new_emb[e_2_co], node_new_emb[e_1_co] - node_new_emb[e_2_co], event_pair_embed[k]))
            event_pair_ = torch.unsqueeze(event_pair_, dim=0)
            if k == 0:
                event_pair = event_pair_
            else:
                event_pair = torch.cat((event_pair, event_pair_))

        pre = self.mlp(event_pair)
        return pre

    # get hyperedge_index from the prediction
    def get_hyperedge_index(self, mask_pre, event_id, event_co_id, label):
        num_new_nodes = int(event_co_id.max()) + 1
        predt = torch.argmax(mask_pre, dim=1)
        hyperedge_index = torch.tensor([[]]).to(device)
        t = 0
        for i in range(num_new_nodes):
            hyperedge = torch.tensor([]).to(device)
            hyperedge_id = torch.tensor([]).to(device)
            # Get the event id and the number of co-references after the event co-reference resolution
            temp_node_id = torch.nonzero(event_co_id == i, as_tuple=False).to(device)
            assert temp_node_id.size()[1] != 0
            num_co = temp_node_id.size()[0]

            if num_co == 1:
                e1_id = temp_node_id[0][0]
                if not torch.any(event_id == e1_id):
                    continue
                temp = torch.nonzero(event_id == e1_id, as_tuple=False).to(device)
                for j in temp:
                    pos = int(j[0])
                    if not self.training:
                        if predt[pos] > 0:
                            e2_pos = 1 - int(j[1])
                            e2_id = event_id[pos][e2_pos]
                            e2_id = torch.unsqueeze(e2_id, dim=0)
                            e2_co_id = event_co_id[e2_id]
                            if not torch.any(hyperedge == e2_co_id):
                                if len(hyperedge) == 0:
                                    hyperedge = torch.tensor([i]).to(device)
                                    hyperedge = torch.cat((hyperedge, e2_co_id), dim=-1)
                                else:
                                    hyperedge = torch.cat((hyperedge, e2_co_id), dim=-1)
                    else:
                        if label[pos] > 0:
                            e2_pos = 1 - int(j[1])
                            e2_id = event_id[pos][e2_pos]
                            e2_id = torch.unsqueeze(e2_id, dim=0)
                            e2_co_id = event_co_id[e2_id]
                            # e2_co_id = torch.unsqueeze(e2_co_id, dim=0)
                            if not torch.any(hyperedge == e2_co_id):
                                if len(hyperedge) == 0:
                                    hyperedge = torch.tensor([i]).to(device)
                                    hyperedge = torch.cat((hyperedge, e2_co_id), dim=-1)
                                else:
                                    hyperedge = torch.cat((hyperedge, e2_co_id), dim=-1)
            else:
                hyperedge = torch.tensor([i]).to(device)
                for j in temp_node_id:
                    if not torch.any(event_id == j):
                        continue
                    temp = torch.nonzero(event_id == j, as_tuple=False).to(device)
                    for k in temp:
                        pos = int(k[0])
                        if not self.training:
                            if predt[pos] > 0:
                                e2_pos = 1 - int(k[1])
                                e2_id = event_id[pos][e2_pos]
                                e2_id = torch.unsqueeze(e2_id, dim=0)
                                e2_co_id = event_co_id[e2_id]
                                if not torch.any(hyperedge == e2_co_id):
                                    hyperedge = torch.cat((hyperedge, e2_co_id), dim=-1)
                        else:
                            if label[pos] > 0:
                                e2_pos = 1 - int(k[1])
                                e2_id = event_id[pos][e2_pos]
                                e2_id = torch.unsqueeze(e2_id, dim=0)
                                e2_co_id = event_co_id[e2_id]
                                if not torch.any(hyperedge == e2_co_id):
                                    if len(hyperedge) == 0:
                                        hyperedge = e2_co_id
                                    else:
                                        hyperedge = torch.cat((hyperedge, e2_co_id), dim=-1)
            if hyperedge.size()[0] > 0:
                num_id = len(hyperedge)
                hyperedge_id = torch.full([num_id], t).to(device)
                hyperedge = torch.unsqueeze(hyperedge, dim=0)
                hyperedge_id = torch.unsqueeze(hyperedge_id, dim=0)
                node_hyperedge = torch.cat((hyperedge, hyperedge_id), dim=0)
                if t == 0:
                    hyperedge_index = node_hyperedge
                else:
                    hyperedge_index = torch.cat((hyperedge_index, node_hyperedge), dim=-1)
                t += 1
        return hyperedge_index

