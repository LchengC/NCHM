#coding:utf-8

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ECI')

    # dataset
    parser.add_argument('--fold',           default=1,          type=int,       help='Fold number')
    parser.add_argument('--num',            default=2,          type=int,       help='Number of event')

    # # model arguments
    parser.add_argument('--model_name',   default='roberta-base', type=str, help='Log model name')
    parser.add_argument('--vocab_size',     default=50271,      type=int,       help='Size of RoBERTa vocab')
    parser.add_argument('--len_arg',        default=334,        type=int,       help='Sentence length')
    parser.add_argument('--len_arg_node',   default=122,        type=int,       help='Node sentence length')
    parser.add_argument('--mlp_size',       default=200,        type=int,       help='mlp layer_size')
    parser.add_argument('--n_hid',          default=200,        type=int,       help='hgnn hidden layer_size')
    parser.add_argument('--n_last',         default=200,        type=int,       help='hgnn last layer_size')
    parser.add_argument('--n_head',         default=1,          type=int,       help='the number of attention heads')
    parser.add_argument('--head_number',    default=1,          type=int,       help='when the multi-head attentions are concatenated')
    parser.add_argument('--concat',         default=True,       type=bool,      help='If set to False, the multi-head attentions are averaged instead of concatenated')
    parser.add_argument('--attention',      default=False,      type=bool,      help='If set to True, attention will be added to this layer')
    parser.add_argument('--drop_out',       default=0.4,        type=int,       help='dropout layer')

    # # training arguments
    parser.add_argument('--seed',           default=209,        type=int,       help='seed for reproducibility')
    parser.add_argument('--wd',             default=1e-2,       type=float,     help='weight decay')
    parser.add_argument('--num_epoch',      default=20,         type=int,       help='number of total epochs to run')
    parser.add_argument('--lr',             default=1e-3,       type=float,     help='initial hgnn learning rate')
    parser.add_argument('--t_lr',           default=1e-5,       type=float,     help='initial transformer learning rate')
    parser.add_argument('--mlp_lr',         default=2e-5,       type=float,     help='initial mlp learning rate')
    parser.add_argument('--batch_size',     default=16,         type=int,       help='batchsize for optimizer updates')

    parser.add_argument('--log',            default='./out/',   type=str,       help='Log result file name')
    parser.add_argument('--model',          default='./load_model/', type=str,  help='Load model file name')

    args = parser.parse_args()
    return args
