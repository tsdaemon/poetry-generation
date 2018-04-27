import os
import numpy as np
import logging
import sys
import torch
import torch.optim as optim
import random
import shutil

from utils.general import init_logging
from model.chargen import CharGen
from config import parser
from trainer import Trainer
from model.utils import device_map_location
from containers.vocab import get_char_vocab


if __name__ == '__main__':
    args = parser.parse_args()

    # arguments validation
    args.cuda = args.cuda and torch.cuda.is_available()

    # random seed
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.benchmark = True

    # prepare dirs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # start logging
    init_logging(os.path.join(args.output_dir, 'parser.log'))
    logging.info('command line: %s', ' '.join(sys.argv))
    logging.info('current config: %s', args)
    logging.info('loading dataset [%s]', args.dataset)

    # load data
    data_dir = args.data_dir
    dataset = args.data_dir
    train_data, dev_data = [torch.load(os.path.join(data_dir, dataset + '.' + t))
                            for t in ['train', 'validation']]
    vocab = get_char_vocab(os.path.join(data_dir, dataset + '.vocab'))

    # load model
    if args.model:
        logging.info('Loading model: {}'.format(args.model))
        # device map location allows to load model trained on GPU on CPU env and vice versa
        model = torch.load(args.model, device_map_location(args.cuda))
    else:
        logging.info('Creating new model'.format(args.model))
        model = CharGen(args, len(vocab))
        if args.cuda:
            model = model.cuda()

    # create learner
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    trainer = Trainer(model, args, optimizer)

    trainer.train_all(train_data, dev_data, args.output_dir)
