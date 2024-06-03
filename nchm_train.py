#coding:utf-8

import time
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import get_node, get_batch, calculate

from model import NCHM


def NCHM_train(args, device, train_data, dev_data, test_data, node_event, printlog):

    train_size = len(train_data)
    dev_size = len(dev_data)
    test_size = len(test_data)

    # ---------- network ----------
    net = NCHM(args).to(device)

    optimizer = torch.optim.AdamW([
        {'params': net.roberta_model1.parameters(), 'lr': args.t_lr},
        {'params': net.HGNN1.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
        {'params': net.HGNN2.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
        {'params': net.mlp.parameters(), 'lr': args.mlp_lr}
    ])

    # model loads initial parameters
    dict_trained = torch.load(args.model)

    net.roberta_model.load_state_dict(dict_trained['roberta_model'])

    cross_entropy = nn.CrossEntropyLoss().to(device)

    # save model and result
    best_intra = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
    best_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
    best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
    dev_best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}

    best_epoch = 0

    printlog('fold: {}'.format(args.fold))
    printlog('num_epoch: {}'.format(args.num_epoch))
    printlog('initial_lr: {}'.format(args.lr))
    printlog('initial_mlp_lr: {}'.format(args.mlp_lr))
    printlog('initial_t_lr: {}'.format(args.t_lr))
    printlog('seed: {}'.format(args.seed))
    printlog('mlp_size: {}'.format(args.mlp_size))
    printlog('n_hid: {}'.format(args.n_hid))
    printlog('n_last: {}'.format(args.n_last))
    printlog('drop_out: {}'.format(args.drop_out))
    printlog('wd: {}'.format(args.wd))
    printlog('len_arg: {}'.format(args.len_arg))
    printlog('len_arg_node: {}'.format(args.len_arg_node))

    printlog('Start training ...')
    breakout = 0

    ##################################  epoch  #################################
    for epoch in range(args.num_epoch):
        print('=' * 20)
        printlog('fold: {}'.format(args.fold))
        printlog('Epoch: {}'.format(epoch))
        torch.cuda.empty_cache()

        all_indices = torch.randperm(train_size).split(1)
        loss_epoch = 0.0
        acc = 0.0
        all_label_ = []
        all_predt_ = []
        all_clabel_ = []
        f1_pred = torch.IntTensor([]).cuda()
        f1_truth = torch.IntTensor([]).cuda()

        start = time.time()

        printlog('RoBERTa_lr:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        printlog('HGNN_lr:{}'.format(optimizer.state_dict()['param_groups'][1]['lr']))
        printlog('PLM_lr:{}'.format(optimizer.state_dict()['param_groups'][3]['lr']))

        ############################################################################
        ##################################  train  #################################
        ############################################################################

        net.train()
        progress = tqdm.tqdm(total=len(train_data), ncols=75,
                             desc='Train {}'.format(epoch))
        total_step = len(train_data)
        step = 0
        n = 0
        for i, batch_indices in enumerate(all_indices, 1):
            progress.update(1)
            # get a batch of wordvecs
            topic = train_data[batch_indices][0][1]
            document_id = train_data[batch_indices][0][2]
            length = len(train_data[batch_indices])
            n += length
            batch_arg, mask_arg, e_idx, mask_indices, event_id, label, clabel, state = get_batch(train_data, node_event, batch_indices, args)
            node_arg, node_mask, node_e, event_co_id = get_node(node_event[topic][document_id], args)


            node_arg = node_arg.to(device)
            node_mask = node_mask.to(device)
            node_e = node_e.to(device)
            batch_arg = batch_arg.to(device)
            mask_arg = mask_arg.to(device)
            e_idx = e_idx.to(device)
            mask_indices = mask_indices.to(device)
            event_id = event_id.to(device)
            event_co_id = torch.tensor(event_co_id).to(device)

            all_label_ += label
            all_clabel_ += clabel
            label = torch.LongTensor(label).to(device)

            # fed data into network
            prediction = net(batch_arg, mask_arg, e_idx, mask_indices, event_id, node_arg, node_mask, node_e, length, label, event_co_id)

            predt = torch.argmax(prediction, dim=1).detach()

            num_correct = (predt == label).sum()
            acc += num_correct.item()
            f1_pred = torch.cat((f1_pred, predt), 0)
            f1_truth = torch.cat((f1_truth, label), 0)

            predt = predt.cpu().tolist()
            all_predt_ += predt

            # loss
            loss = cross_entropy(prediction, label)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # report
            loss_epoch += loss.item()
            if i % 50 == 0:
                printlog('loss={:.4f}, acc={:.4f}, F1_score={:.4f}'.format(
                    loss_epoch / 50, acc / n,
                    f1_score(f1_truth.cpu(), f1_pred.cpu(),
                             average='macro')))
                loss_epoch = 0.0
                acc = 0.0
                n = 0.0
                f1_pred = torch.IntTensor([]).cuda()
                f1_truth = torch.IntTensor([]).cuda()
        end = time.time()
        print('Training Time: {:.2f}s'.format(end - start))

        progress.close()

        ############################################################################
        ##################################  dev  ###################################
        ############################################################################
        all_indices = torch.randperm(dev_size).split(1)
        all_label = []
        all_predt = []
        all_clabel = []

        progress = tqdm.tqdm(total=len(dev_data), ncols=75,
                             desc='Dev {}'.format(epoch))

        net.eval()
        for batch_indices in all_indices:
            progress.update(1)
            # get a batch of wordvecs
            topic = dev_data[batch_indices][0][1]
            document_id = dev_data[batch_indices][0][2]
            length = len(dev_data[batch_indices])
            batch_arg, mask_arg, e_idx, mask_indices, event_id, label, clabel, state = get_batch(dev_data, node_event, batch_indices, args)
            node_arg, node_mask, node_e, event_co_id = get_node(node_event[topic][document_id], args)

            node_arg = node_arg.to(device)
            node_mask = node_mask.to(device)
            node_e = node_e.to(device)
            batch_arg = batch_arg.to(device)
            mask_arg = mask_arg.to(device)
            e_idx = e_idx.to(device)
            event_id = event_id.to(device)
            event_co_id = torch.tensor(event_co_id).to(device)

            all_label += label
            all_clabel += clabel
            label = torch.LongTensor(label).to(device)

            # fed data into network
            prediction = net(batch_arg, mask_arg, e_idx, mask_indices, event_id, node_arg, node_mask, node_e, length, label, event_co_id)

            predt = torch.argmax(prediction, dim=1).detach()

            predt = predt.cpu().tolist()
            all_predt += predt

        progress.close()

        ############################################################################
        ##################################  test  ##################################
        ############################################################################
        all_indices = torch.randperm(test_size).split(1)
        all_label_t = []
        all_predt_t = []
        all_clabel_t = []
        acc = 0.0

        progress = tqdm.tqdm(total=len(test_data), ncols=75,
                             desc='Test {}'.format(epoch))

        net.eval()
        for batch_indices in all_indices:
            progress.update(1)
            # get a batch of wordvecs
            topic = test_data[batch_indices][0][1]
            document_id = test_data[batch_indices][0][2]
            length = len(test_data[batch_indices])
            batch_arg, mask_arg, e_idx, mask_indices, event_id, label, clabel, state = get_batch(test_data, node_event, batch_indices, args)
            node_arg, node_mask, node_e, event_co_id = get_node(node_event[topic][document_id], args)

            node_arg = node_arg.to(device)
            node_mask = node_mask.to(device)
            node_e = node_e.to(device)
            batch_arg = batch_arg.to(device)
            mask_arg = mask_arg.to(device)
            e_idx = e_idx.to(device)
            event_id = event_id.to(device)
            event_co_id = torch.tensor(event_co_id).to(device)

            all_label_t += label
            all_clabel_t += clabel
            label = torch.LongTensor(label).to(device)

            # fed data into network
            prediction = net(batch_arg, mask_arg, e_idx, mask_indices, event_id, node_arg, node_mask, node_e, length, label, event_co_id)

            predt = torch.argmax(prediction, dim=1).detach()

            predt = predt.cpu().tolist()
            all_predt_t += predt

        progress.close()

        ############################################################################
        ##################################  result  ##################################
        ############################################################################
        ######### Train Results Print #########
        printlog('-------------------')
        printlog("TIME: {}".format(time.time() - start))
        printlog('EPOCH : {}'.format(epoch))
        printlog("TRAIN:")
        printlog("\tprecision score: {}".format(precision_score(all_label_, all_predt_, average=None)[1]))
        printlog("\trecall score: {}".format(recall_score(all_label_, all_predt_, average=None)[1]))
        printlog("\tf1 score: {}".format(f1_score(all_label_, all_predt_, average=None)[1]))

        ######### Dev Results Print #########
        printlog("DEV:")
        d_1, d_2, d_3, dev_intra, dev_cross = calculate(all_label, all_predt, all_clabel, epoch, printlog)
        # INTRA + CROSS SENTENCE
        dev_intra_cross = {
            'epoch': epoch,
            'p': precision_score(all_label, all_predt, average=None)[1],
            'r': recall_score(all_label, all_predt, average=None)[1],
            'f1': f1_score(all_label, all_predt, average=None)[1]
        }

        printlog('\tINTRA + CROSS:')
        printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(d_1, d_2, d_3))
        printlog("\t\tprecision score: {}".format(dev_intra_cross['p']))
        printlog("\t\trecall score: {}".format(dev_intra_cross['r']))
        printlog("\t\tf1 score: {}".format(dev_intra_cross['f1']))

        ######### Dev Results Print #########
        printlog("TEST:")
        t_1, t_2, t_3, test_intra, test_cross = calculate(all_label_t, all_predt_t, all_clabel_t, epoch, printlog)

        # INTRA + CROSS SENTENCE
        test_intra_cross = {
            'epoch': epoch,
            'p': precision_score(all_label_t, all_predt_t, average=None)[1],
            'r': recall_score(all_label_t, all_predt_t, average=None)[1],
            'f1': f1_score(all_label_t, all_predt_t, average=None)[1]
        }
        printlog('\tINTRA + CROSS:')
        printlog("\t\tTest Acc={:.4f}".format(acc / test_size))
        printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(t_1, t_2, t_3))
        printlog("\t\tprecision score: {}".format(test_intra_cross['p']))
        printlog("\t\trecall score: {}".format(test_intra_cross['r']))
        printlog("\t\tf1 score: {}".format(test_intra_cross['f1']))

        breakout += 1

        # record the best result
        if dev_intra_cross['f1'] > dev_best_intra_cross['f1']:
            printlog('New best epoch...')
            dev_best_intra_cross = dev_intra_cross
            best_intra_cross = test_intra_cross
            best_intra = test_intra
            best_cross = test_cross
            best_epoch = epoch
            breakout = 0

        printlog('=' * 20)
        printlog('Best result at epoch: {}'.format(best_epoch))
        printlog('Eval intra: {}'.format(best_intra))
        printlog('Eval cross: {}'.format(best_cross))
        printlog('Eval intra cross: {}'.format(best_intra_cross))
        printlog('Breakout: {}'.format(breakout))

        if breakout == 5:
            break