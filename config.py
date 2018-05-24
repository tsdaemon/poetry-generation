import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', default='./data/poetry')
parser.add_argument('-dataset', default='nyg')
parser.add_argument('-random_seed', default=181783, type=int)
parser.add_argument('-output_dir', default='./results')

parser.add_argument('-mode', default='log_loss', choices=['log_loss', 'q'])

# neural model's parameters
parser.add_argument('-model', default='', type=str)
parser.add_argument('-char_embed_dim', default=64, type=int)
parser.add_argument('-hidden_dim', default=256, type=int)
parser.add_argument('-layers', default=2, type=int)
parser.add_argument('-dropout', default=0.2, type=float)
parser.add_argument('-freeze_embeds', dest='freeze_embeds', action='store_true')
parser.add_argument('-no_freeze_embeds', dest='freeze_embeds', action='store_false')
parser.set_defaults(freeze_embeds=False)

# training
parser.add_argument('-clip_grad', default=0., type=float)
parser.add_argument('-train_patience', default=10, type=int)
parser.add_argument('-max_epoch', default=200, type=int)
parser.add_argument('-batch_size', default=50, type=int)
parser.add_argument('-lr', default=0.001, type=float)
parser.add_argument('-cuda', dest='cuda', action='store_true')
parser.add_argument('-no_cuda', dest='cuda', action='store_false')

# q learning
parser.add_argument('-max_q_epoch', default=100, type=int)
parser.add_argument('-train_q_patience', default=10, type=int)
parser.add_argument('-number_of_plays', default=100, type=int)
parser.add_argument('-replay_capacity', default=200, type=int)
parser.add_argument('-q_batch_size', default=10, type=int)
parser.add_argument('-q_train_batches', default=10, type=int)
parser.add_argument('-q_gamma', default=0.95, type=int)
parser.add_argument('-q_epsilon_start', default=0.4, type=int)
parser.add_argument('-q_epsilon_decay', default=0.9, type=int)

# decoding
parser.add_argument('-decode_max_time_step', default=1000, type=int)

# input
parser.add_argument('-max_poem_length', default=1000, type=int)
parser.add_argument('-min_char_freq', default=10, type=int)

# scorers
parser.add_argument('-path_to_accents_csv', default='./data/accents.csv')
