import torch
from tqdm import tqdm
import logging
import os
import numpy as np
import shutil

from training.replay_memory import ReplayMemory
import Constants
from torch.autograd import Variable as Var
from model.utils import cudafication


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
        train_l1_perf = []

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
            train_l1 = self.train(epoch)
            logging.info('Epoch {} training finished, l1: {}.'.format(
                epoch + 1, train_l1))
            train_l1_perf.append(train_l1)

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
                # decay epsilon
                epsilon = self.config.q_epsilon_start * self.config.q_epsilon_decay ** epoch

                # generate poems
                inputs, outputs = zip(*self.generator.generate_q(epsilon))
                poem = Constants.postprocess_poem(''.join(self.vocab.convert_to_labels(outputs)))

                # save for replay
                reward = self.scorer.score(poem)
                self.memory.push(reward, list(inputs), list(outputs))
                total_reward += reward

                # store history
                f.write('{}\n{}\n\n'.format(poem, reward))

        return total_reward / self.config.number_of_plays

    def train(self, epoch):
        self.model.train()
        total_l1 = 0

        for _ in tqdm(range(self.config.q_train_batches), desc='Training epoch {}'.format(epoch+1)):
            sample = self.memory.sample(self.config.q_batch_size)
            X, idx, q = self.get_train_data(sample)
            l1 = self.model.forward_train_q(X, q, idx)
            total_l1 += l1.item()
            l1.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_l1/self.config.q_train_batches

    def get_train_data(self, sample):
        # sample num
        length = len(sample)

        # longest seq
        longest_length = max([len(s[1]) for s in sample])

        # (batch_size, seq_length)
        qs = np.zeros((length, longest_length), dtype=np.float32)
        inputs = np.zeros((length, longest_length), dtype=np.long)
        outputs = np.zeros((length, longest_length), dtype=np.long)

        for i, sample in enumerate(sample):
            reward, input, output = sample

            # decaying reward
            gamma = np.full((len(input),), self.config.q_gamma)
            gamma[0] = 1
            q = np.flip(gamma.cumprod(), axis=0) * reward

            # save training values
            qs[i, :len(input)] = q
            inputs[i, :len(input)] = input
            inputs[i, len(input):] = Constants.PAD
            outputs[i, :len(output)] = output
            outputs[i, len(output):] = Constants.PAD

        qs = cudafication(torch.from_numpy(qs), self.config.cuda)

        # (batch_size, seq_length)
        X = Var(cudafication(torch.from_numpy(inputs), self.config.cuda),
                             requires_grad=False)
        idx = Var(cudafication(torch.from_numpy(outputs), self.config.cuda),
                               requires_grad=False)
        return X, idx, qs







