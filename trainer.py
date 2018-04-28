import torch
from tqdm import tqdm
import logging
import os
import numpy as np
import math
import shutil

from utils.general import get_batches
from utils.io import send_telegram
import Constants
from torch.autograd import Variable as Var


class Trainer(object):
    def __init__(self, model, config, optimizer, generator, vocab):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.generator = generator
        self.vocab = vocab

    def train_all(self, train_data, dev_data, results_dir):
        max_epoch = self.config.max_epoch
        patience_counter = 0
        train_logloss_perf = []
        val_logloss_perf = []

        train_logloss, validation_logloss = 10000, 10000
        for epoch in range(max_epoch):
            train_logloss = self.train(train_data, epoch)
            train_logloss_perf.append(train_logloss)
            logging.info('Epoch {} training finished, loss: {}.'.format(epoch+1, train_logloss))

            epoch_dir = os.path.join(results_dir, str(epoch+1))
            if os.path.exists(epoch_dir):
                shutil.rmtree(epoch_dir)
            os.mkdir(epoch_dir)
            model_path = os.path.join(epoch_dir, 'model.pth')
            logging.info('Saving model at {}.'.format(model_path))
            torch.save(self.model, model_path)

            validation_logloss = self.validate(dev_data, epoch)
            logging.info('Epoch {} validation finished, logloss: {}.'.format(
                epoch + 1, validation_logloss))
            val_logloss_perf.append(validation_logloss)

            self.generate(epoch, epoch_dir)

            if validation_logloss < 10:
                if len(val_logloss_perf) == 0 or validation_logloss > np.array(val_logloss_perf).max():
                    patience_counter = 0
                    logging.info('Found best model on epoch {}'.format(epoch+1))
                else:
                    patience_counter += 1
                    logging.info('Hitting patience_counter: {}'.format(patience_counter))
                    if patience_counter >= self.config.train_patience:
                        logging.info('Early Stop!')
                        break

        report_result = {
            "Train loss": train_logloss,
            "Validation loss": validation_logloss,
            "Last epoch": epoch
        }
        self.report_bot(report_result)

    def train(self, dataset, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        batch_size = self.config.batch_size
        indices = torch.randperm(len(dataset))
        if self.config.cuda:
            indices = indices.cuda()
        total_batches = math.floor(len(indices)/batch_size)+1
        batches = list(get_batches(indices, batch_size))

        for i, batch in tqdm(enumerate(batches), desc='Training epoch '+str(epoch+1)+'',
                             total=total_batches):
            X, y = dataset.get_batch(batch)

            X, y = Var(X, requires_grad=False), Var(y, requires_grad=False)

            loss = self.model.forward_train(X, y)

            total_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            logging.debug('Batch {}, loss {}'.format(i+1, loss[0]))

        return total_loss/len(dataset)

    def validate(self, dataset, epoch):
        self.model.eval()
        total_loss = 0.0

        batch_size = self.config.batch_size
        indices = torch.randperm(len(dataset))
        if self.config.cuda:
            indices = indices.cuda()
        total_batches = math.floor(len(indices) / batch_size) + 1
        batches = list(get_batches(indices, batch_size))

        for i, batch in tqdm(enumerate(batches), desc='Testing epoch ' + str(epoch + 1) + '',
                             total=total_batches):
            X, y = dataset.get_batch(batch)

            X, y = Var(X.unsqueeze(0), requires_grad=False), Var(y.unsqueeze(0),
                                                                 requires_grad=False)
            loss = self.model.forward_train(X, y)
            total_loss += loss.data[0]
            logging.debug('Validation batch {}, loss {}'.format(i, loss[0]))

        total_loss /= len(dataset)

        return total_loss

    def generate(self, epoch, out_dir):
        # generate test words
        generate_file_name = os.path.join(out_dir, 'generated.txt')
        with open(generate_file_name, 'w') as f:
            for i in tqdm(range(10), desc='Generating for epoch {}'.format(epoch+1)):
                vector = self.generator.generate(random_seed=i)
                poem = Constants.postprocess_poem(''.join(self.vocab.convert_to_labels(vector)))
                f.write(poem + "\n\n")

    def report_bot(self, report_dict):
        msg = "Finished experiment with config {}.\n\n".format(self.config)
        msg += "\n".join(["{}: {}.".format(k, v) for k, v in report_dict.items()])
        # send_telegram(msg)
