import torch
from tqdm import tqdm
import logging
import os
import numpy as np
import shutil

from training.replay_memory import ReplayMemory
from utils.general import get_batches
from utils.io import send_telegram
import Constants
from torch.autograd import Variable as Var


class QTrainer(object):
    def __init__(self, model, config, optimizer, generator, vocab, scorer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.generator = generator
        self.vocab = vocab
        self.scorer = scorer
        self.memory = ReplayMemory(config.replay_capacity)

    def train_all(self, results_dir):
        max_q_epoch = self.config.max_q_epoch
        patience_counter = 0
        play_reward_perf = []
        train_mse_perf = []

        for epoch in range(max_q_epoch):
            # prepare
            epoch_dir = os.path.join(results_dir, 'q' + str(epoch + 1))
            if os.path.exists(epoch_dir):
                shutil.rmtree(epoch_dir)
            os.mkdir(epoch_dir)

            # play
            play_reward = self.play(epoch, epoch_dir)
            play_reward_perf.append(play_reward)
            logging.info('Epoch {} plays finished, average reward: {}.'.format(epoch+1, play_reward))
            model_path = os.path.join(epoch_dir, 'model.pth')
            logging.info('Saving model at {}.'.format(model_path))
            torch.save(self.model, model_path)

            # train
            train_mse = self.train(epoch)
            logging.info('Epoch {} training finished, mse: {}.'.format(
                epoch + 1, train_mse))
            train_mse_perf.append(train_mse)

            if play_reward >= np.array(play_reward_perf).max():
                patience_counter = 0
                logging.info('Found best model on epoch {}'.format(epoch + 1))
            else:
                patience_counter += 1
                logging.info('Hitting patience_counter: {}'.format(patience_counter))
                if patience_counter >= self.config.train_patience:
                    logging.info('Early Stop!')
                    break

    def play(self, epoch, out_dir):
        self.model.eval()
        total_reward = 0

        generate_file_name = os.path.join(out_dir, 'generated.txt')
        with open(generate_file_name, 'w') as f:
            for _ in tqdm(range(self.config.number_of_plays), desc='Generating plays for epoch {}'.format(epoch+1)):
                inputs, outputs, idxs = zip(*self.generator.generate_q)
                poem = Constants.postprocess_poem(''.join(self.vocab.convert_to_labels(idxs)))
                f.write(poem)

                reward = self.scorer.score(poem)
                self.replay.push(reward, inputs, outputs)
                total_reward += reward
        return total_reward / self.config.number_of_plays

    def train(self, epoch):
        self.model.train()
        total_mse = 0

        for _ in tqdm(range(self.config.q_train_batches)):
            play_batch = self.memory.sample(self.config.q_batch_size)




