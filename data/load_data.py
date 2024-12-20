# -*- coding: utf-8 -*-
import numpy as np
import random

def load_data(args):
    data_document = np.load('./data/database/train_data.npy', allow_pickle=True).item()
    data_event = np.load('./data/database/train_event_data.npy', allow_pickle=True).item()
    data_dev = []
    data_train = {}
    for topic in data_document:
        if topic != '37' and topic != '41':
            for document in data_document[topic]:
                data = data_document[topic][document].copy()
                data_train[document]= data
        else:
            for document_dev in data_document[topic]:
                data = data_document[topic][document_dev].copy()
                data_dev.append(data)
    doc_name = list(data_train.keys())
    random.shuffle(doc_name)
    doc_num = len(doc_name)
    fold1_doc = doc_name[0: int(doc_num / 5)]
    fold2_doc = doc_name[int(doc_num / 5): int(doc_num / 5) * 2]
    fold3_doc = doc_name[int(doc_num / 5) * 2: int(doc_num / 5) * 3]
    fold4_doc = doc_name[int(doc_num / 5) * 3: int(doc_num / 5) * 4]
    fold5_doc = doc_name[int(doc_num / 5) * 4:]
    fold1 = get_fold_data(data_train, fold1_doc)
    fold2 = get_fold_data(data_train, fold2_doc)
    fold3 = get_fold_data(data_train, fold3_doc)
    fold4 = get_fold_data(data_train, fold4_doc)
    fold5 = get_fold_data(data_train, fold5_doc)

    if args.fold == 1:
        train_data = fold2 + fold3 + fold4 + fold5
        test_data = fold1

    elif args.fold == 2:
        train_data = fold1 + fold3 + fold4 + fold5
        test_data = fold2

    elif args.fold == 3:
        train_data = fold1 + fold2 + fold4 + fold5
        test_data = fold3

    elif args.fold == 4:
        train_data = fold1 + fold2 + fold3 + fold5
        test_data = fold4

    elif args.fold == 5:
        train_data = fold1 + fold2 + fold3 + fold4
        test_data = fold5

    return train_data, data_dev, test_data, data_event

def get_fold_data(trainAndtest_doc_dict, fold_doc):
    fold = []
    for doc in fold_doc:
        fold.append(trainAndtest_doc_dict[doc])
    return fold