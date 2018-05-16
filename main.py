import logging
import os
import random
import sys
import shutil

import torch
import torch.optim as optim

from config import parser
from containers.vocab import get_char_vocab
from model.chargen import CharGen
from model.utils import device_map_location
from training.log_loss_trainer import LogLossTrainer
from utils.general import init_logging
from poetry.softmax_generator import SoftmaxGenerator
import Constants


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

    # prepare out dir
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # start logging
    init_logging(os.path.join(args.output_dir, 'parser.log'))
    logging.info('command line: %s', ' '.join(sys.argv))
    logging.info('current config: %s', args)
    logging.info('loading dataset [%s]', args.dataset)

    # load data
    data_dir = args.data_dir
    dataset = args.dataset
    train_data, dev_data = [torch.load(os.path.join(data_dir, dataset + '.' + t))
                            for t in ['train', 'validation']]
    train_data.prepare_device(args.cuda)
    dev_data.prepare_device(args.cuda)
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
    generator = SoftmaxGenerator(model, Constants.SOP, Constants.EOP, args.decode_max_time_step, len(vocab))
    trainer = LogLossTrainer(model, args, optimizer, generator, vocab)

    trainer.train_all(train_data, dev_data, args.output_dir)
