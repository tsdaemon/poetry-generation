import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from config import parser
from containers.vocab import get_char_vocab
from containers.dataset import Dataset
import Constants


def extract_poems(contents, length):
    return [Constants.preprocess_poem(poem)[:length] for poem in contents.split('\n\n\n')]


def extract_char_vocab(contents, vocabfile, min_freq):
    from collections import defaultdict

    contents = contents.lower()
    counted = defaultdict(int)
    for v in contents:
        counted[v] += 1

    chars = {k for k, f in counted.items() if f >= min_freq and k != '\n'}

    with open(vocabfile, 'w') as f:
        f.write('\n'.join(chars))

    vocab = get_char_vocab(vocabfile)

    return vocab, set(contents) - set(chars)


def generate_out(out_filename, out_char_filename, poems, vocab, max_poem_length):
    with open(out_filename, 'w') as f:
        with open(out_char_filename, 'w') as f2:
            for poem in poems:

                assert len(poem) <= max_poem_length
                poem += ''.join([Constants.PAD_CHAR]*(max_poem_length-len(poem)))
                assert len(poem) == max_poem_length
                f2.write(poem + '\n')

                idx = vocab.convert_to_idx(poem, Constants.UNK_CHAR)
                s = ' '.join(map(str, idx)) + '\n'
                f.write(s)


def remove_punctuation(contents):
    return contents.replace("\"", "").replace(";", "").replace(")", "").replace("(", "")\
        .replace(":", "").replace("-", "").replace("—", "")


if __name__ == '__main__':
    args = parser.parse_args()
    poetry_folder = args.data_dir

    filename = os.path.join(poetry_folder, 'nyg.txt')
    with open(filename) as f:
        contents = f.read()

    contents = remove_punctuation(contents)

    vocabfile = os.path.join(poetry_folder, 'nyg.vocab')
    vocab, excluded_symbols = extract_char_vocab(contents, vocabfile, args.min_char_freq)
    print('Extracted vocabulary, length: {}, excluded symbols: {}'.format(len(vocab), excluded_symbols))

    poems = extract_poems(contents, args.max_poem_length)
    lengths = list(map(len, poems))
    print(
        'Extracted {} examples, min length: {}, max length: {}, median length: {}, quantiles: {}'.format(
            len(poems),
            min(lengths),
            max(lengths),
            np.median(lengths),
            np.percentile(lengths, [10, 20, 30])))

    # split train validation 80:20
    train_length = int(len(poems) * 0.8)
    poems_train = poems[:train_length]
    poems_validation = poems[train_length:]

    to_generate = [(poems_train, 'nyg.train.out', 'nyg.train'),
                   (poems_validation, 'nyg.validation.out', 'nyg.validation')]

    for poems, out_filename, dataset_filename in to_generate:
        out_idx_filename = os.path.join(poetry_folder, out_filename)
        out_char_filename = os.path.join(poetry_folder, out_filename + '.char')
        generate_out(out_idx_filename, out_char_filename, poems, vocab, args.max_poem_length)

        dataset = Dataset(out_idx_filename)
        dataset_filename = os.path.join(poetry_folder, dataset_filename)
        torch.save(dataset, dataset_filename)
