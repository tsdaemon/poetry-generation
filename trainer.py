import torch
from tqdm import tqdm
import logging
import os
import numpy as np
import math
import shutil

from utils.general import get_batches
from utils.io import send_telegram


class Trainer(object):
    def __init__(self, model, config, optimizer, generator):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.generator = generator

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

            validation_logloss = self.validate(dev_data, epoch, epoch_dir)
            logging.info('Epoch {} validation finished, bleu: {}, accuracy: {}, errors: {}.'.format(
                epoch + 1, validation_logloss))

            val_logloss_perf.append(validation_logloss)

            if validation_logloss < 10:
                if len(val_logloss_perf) == 0 or validation_logloss > np.array(val_logloss_perf).max():
                    patience_counter = 0
                    logging.info('Found best model on epoch {}'.format(epoch+1))
                    best_model_file = model_path
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

        for i, batch in tqdm(enumerate(batches), desc='Training epoch '+str(epoch+1)+'', total=total_batches):
            X, y = dataset.get_batch(batch)

            loss = self.model.forward_train(X, y)
            assert loss > 0, "NLL can not be less than zero"

            total_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            logging.debug('Batch {}, loss {}'.format(i+1, loss[0]))

        return total_loss/len(dataset)

    def validate(self, dataset, epoch, out_dir):
        self.model.eval()
        total_loss = 0.0

        for idx in tqdm(range(len(dataset)), desc='Testing epoch '+str(epoch+1)+''):
            X, y = dataset[idx]

            loss = self.model.forward_train(X, y)
            total_loss += loss.data[0]
            logging.debug('Validation idx {}, loss {}'.format(idx, loss[0]))

        total_loss /= len(dataset)

        return total_loss

    def report_bot(self, report_dict):
        msg = "Finished experiment with config {}.\n\n".format(self.config)
        msg += "\n".join(["{}: {}.".format(k, v) for k, v in report_dict.items()])
        send_telegram(msg)
