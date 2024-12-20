#coding:utf-8

# This project is for Roberta model.
import os
import time
import numpy as np
import torch
import logging
from datetime import datetime
from data import load_data
from parameter import parse_args
from nchm_train import NCHM_train
from pes_train import PES_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
# load parameters
args = parse_args()

# set seed for random number
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
setup_seed(args.seed)

if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.model):
    os.mkdir(args.model)

args.log = args.log + 'lr' + str(args.lr) + 't_lr' + str(args.t_lr) +'mlp_lr' + str(args.mlp_lr) + '/'
args.model = args.model + 'fold-' + str(args.fold) + '.pth'

if not os.path.exists(args.log):
    os.mkdir(args.log)

t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + 'fold-' + str(args.fold) + '__' + t + '.txt'

# refine
for name in logging.root.manager.loggerDict:
    if 'transformers' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)

logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=args.log,
                    filemode='w')

logger = logging.getLogger(__name__)


def printlog(message: object, printout: object = True) -> object:
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)


# load Roberta model
printlog('Passed args:')
printlog('log path: {}'.format(args.log))
printlog('transformer model: {}'.format(args.model_name))

# load data tsv file
printlog('Loading data')

train_data, dev_data, test_data, node_event = load_data(args)

print('Data loaded')

printlog('\n')
printlog('===========================================================')
printlog('===========================================================')
printlog('======================Train PES module=====================')
printlog('===========================================================')
printlog('===========================================================\n')

PES_train(args, device, train_data, dev_data, test_data, node_event, printlog)

printlog('\n')
printlog('===========================================================')
printlog('===========================================================')
printlog('======================Train NCHM module=====================')
printlog('===========================================================')
printlog('===========================================================\n')

if args.n_head > 1:
    if args.concat == True:
        args.head_number = args.n_head

NCHM_train(args, device, train_data, dev_data, test_data, node_event, printlog)


