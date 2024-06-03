#coding:utf-8

import time
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import pes_get_batch, calculate

from model import PES

def PES_train(args, device, traindata, devdata, testdata, node_event, printlog):

    train_data, dev_data ,test_data = [], [], []

    for train in traindata:
        train_data.extend(train)

    for dev in devdata:
        dev_data.extend(dev)

    for test in testdata:
        test_data.extend(test)

    train_size = len(train_data)
    dev_size = len(dev_data)
    test_size = len(test_data)

    net = PES(args).to(device)

    optimizer = torch.optim.AdamW([
        {'params': net.roberta_model.parameters(), 'lr': args.t_lr},

    ])

    cross_entropy = nn.CrossEntropyLoss().to(device)

    # save model and result
    best_intra = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
    best_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
    best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
    dev_best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
    state = {}

    best_epoch = 0
    printlog('fold: {}'.format(args.fold))
    printlog('batch_size:{}'.format(args.batch_size))
    printlog('num_epoch: {}'.format(args.num_epoch))
    printlog('initial_RoBERTa_lr: {}'.format(args.t_lr))
    printlog('seed: {}'.format(args.seed))
    printlog('mlp_size: {}'.format(args.mlp_size))
    printlog('drop_out: {}'.format(args.drop_out))
    printlog('len_arg: {}'.format(args.len_arg))
    printlog('Start training PES module...')
    breakout = 0
    ##################################  epoch  #################################
    for epoch in range(args.num_epoch):
        print('=' * 20)
        printlog('Epoch: {}'.format(epoch))
        torch.cuda.empty_cache()

        all_indices = torch.randperm(train_size).split(args.batch_size)
        loss_epoch = 0.0
        acc = 0.0
        all_label_ = []
        all_predt_ = []
        all_clabel_ = []
        f1_pred = torch.IntTensor([]).cuda()
        f1_truth = torch.IntTensor([]).cuda()

        start = time.time()

        printlog('RoBERTa_lr:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        ############################################################################
        ##################################  train  #################################
        ############################################################################

        net.train()
        progress = tqdm.tqdm(total=len(train_data) // args.batch_size + 1, ncols=75,
                             desc='Train {}'.format(epoch))
        total_step = len(train_data) // args.batch_size + 1
        step = 0
        for i, batch_indices in enumerate(all_indices, 1):
            progress.update(1)
            # get a batch of wordvecs
            batch_arg, mask_arg, mask_indices, label, clabel, state_1 = pes_get_batch(train_data, node_event, batch_indices, args)
            batch_arg = batch_arg.to(device)
            mask_arg = mask_arg.to(device)
            mask_indices = mask_indices.to(device)
            length = len(batch_indices)
            # fed data into network
            prediction = net(batch_arg, mask_arg, mask_indices, length)

            all_label_ += label
            all_clabel_ += clabel

            predt = torch.argmax(prediction, dim=1).detach()
            label = torch.LongTensor(label).to(device)

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
            # print(predt)
            loss_epoch += loss.item()
            if i % (3000 // args.batch_size) == 0:
                printlog('loss={:.4f}, acc={:.4f}, F1_score={:.4f}'.format(
                    loss_epoch / (3000 // args.batch_size), acc / 3000,
                    f1_score(f1_truth.cpu(), f1_pred.cpu(),
                             average='macro')))
                loss_epoch = 0.0
                acc = 0.0
                f1_pred = torch.IntTensor([]).cuda()
                f1_truth = torch.IntTensor([]).cuda()
        end = time.time()
        print('Training Time: {:.2f}s'.format(end - start))

        progress.close()

        ############################################################################
        ##################################  dev  ###################################
        ############################################################################
        all_indices = torch.randperm(dev_size).split(args.batch_size)
        all_label = []
        all_predt = []
        all_clabel = []

        progress = tqdm.tqdm(total=len(dev_data) // args.batch_size + 1, ncols=75,
                             desc='Eval {}'.format(epoch))

        net.eval()
        for batch_indices in all_indices:
            progress.update(1)
            # get a batch of dev_data
            batch_arg, mask_arg, mask_indices, label, clabel, state_1 = pes_get_batch(dev_data, node_event, batch_indices, args)
            batch_arg = batch_arg.to(device)
            mask_arg = mask_arg.to(device)
            mask_indices = mask_indices.to(device)
            length = len(batch_indices)

            # fed data into network
            prediction = net(batch_arg, mask_arg, mask_indices, length)
            predt = torch.argmax(prediction, dim=1).detach().cpu().tolist()


            all_label += label
            all_predt += predt
            all_clabel += clabel

        progress.close()

        ############################################################################
        ##################################  test  ##################################
        ############################################################################
        all_indices = torch.randperm(test_size).split(args.batch_size)
        all_label_t = []
        all_predt_t = []
        all_clabel_t = []
        acc = 0.0

        progress = tqdm.tqdm(total=len(test_data) // args.batch_size + 1, ncols=75,
                             desc='Eval {}'.format(epoch))

        net.eval()
        for batch_indices in all_indices:
            progress.update(1)
            # get a batch of dev_data
            batch_arg, mask_arg, mask_indices, label, clabel, state_1 = pes_get_batch(test_data, node_event, batch_indices, args)
            batch_arg = batch_arg.to(device)
            mask_arg = mask_arg.to(device)
            mask_indices = mask_indices.to(device)
            length = len(batch_indices)

            all_clabel_t += clabel
            all_label_t += label
            # fed data into network
            prediction = net(batch_arg, mask_arg, mask_indices, length)
            predt = torch.argmax(prediction, dim=1).detach()

            label = torch.LongTensor(label).to(device)
            num_correct = (predt == label).sum()
            acc += num_correct.item()
            predt = predt.cpu().tolist()
            all_predt_t += predt

        progress.close()

        ############################################################################
        ##################################  result  ##################################
        ############################################################################
        ######### Train Results Print #########
        printlog('-------------------')
        printlog('-------------------')
        printlog('-------------------')
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
            # Early Stop
            breakout = 0
            state = {'roberta_model': net.roberta_model.state_dict()}
            torch.save(state, args.model)
            del state

        printlog('=' * 20)
        printlog('Best result at epoch: {}'.format(best_epoch))
        printlog('Eval intra: {}'.format(best_intra))
        printlog('Eval cross: {}'.format(best_cross))
        printlog('Eval intra cross: {}'.format(best_intra_cross))
        printlog('Breakout: {}'.format(breakout))

        if breakout == 3:
            break