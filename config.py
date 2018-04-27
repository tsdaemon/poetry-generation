import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', default='./data/poetry')
parser.add_argument('-dataset', default='nyg')
parser.add_argument('-random_seed', default=181783, type=int)
parser.add_argument('-output_dir', default='./results')

# neural model's parameters
parser.add_argument('-model', default='', type=str)
parser.add_argument('-char_embed_dim', default=64, type=int)
parser.add_argument('-hidden_dim', default=128, type=int)
parser.add_argument('-dropout', default=0.2, type=float)
parser.add_argument('-freeze_embeds', dest='freeze_embeds', action='store_true')
parser.add_argument('-no_freeze_embeds', dest='freeze_embeds', action='store_false')
parser.set_defaults(freeze_embeds=False)

# training
parser.add_argument('-clip_grad', default=0., type=float)
parser.add_argument('-train_patience', default=10, type=int)
parser.add_argument('-max_epoch', default=100, type=int)
parser.add_argument('-batch_size', default=20, type=int)
parser.add_argument('-lr', default=0.001, type=float)
parser.add_argument('-cuda', dest='cuda', action='store_true')
parser.add_argument('-no_cuda', dest='cuda', action='store_false')

# decoding
parser.add_argument('-decode_max_time_step', default=1500, type=int)

# input
parser.add_argument('-max_poem_length', default=1500, type=int)
parser.add_argument('-min_poem_length', default=30, type=int)
parser.add_argument('-min_char_freq', default=10, type=int)
