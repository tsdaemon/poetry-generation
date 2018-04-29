import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from config import parser
from containers.vocab import get_char_vocab
from containers.dataset import Dataset
import Constants
from utils.general import get_batches


def extract_by_length(poem, length):
    poem = Constants.preprocess_poem(poem)
    batches = list(map(lambda x: ''.join(x), get_batches(poem, length)))
    return batches


def extract_examples(contents, length):
    return [batch for poem in contents.split('\n\n\n') for batch in extract_by_length(poem, length)]


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


def generate_out(out_filename, out_char_filename, examples, vocab, example_length):
    with open(out_filename, 'w') as f:
        with open(out_char_filename, 'w') as f2:
            for example in examples:

                assert len(example) <= example_length
                example += ''.join([Constants.PAD_CHAR]*(example_length-len(example)))
                assert len(example) == example_length
                f2.write(example + '\n')

                idx = vocab.convert_to_idx(example, Constants.UNK_CHAR)
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

    examples = extract_examples(contents, args.example_length)
    lengths = list(map(len, examples))
    print(
        'Extracted {} examples, min length: {}, max length: {}, median length: {}, quantiles: {}'.format(
            len(examples),
            min(lengths),
            max(lengths),
            np.median(lengths),
            np.percentile(lengths, [10, 20, 30])))

    # split train validation 80:20
    train_length = int(len(examples) * 0.8)
    poems_train = examples[:train_length]
    poems_validation = examples[train_length:]

    to_generate = [(poems_train, 'nyg.train.out', 'nyg.train'),
                   (poems_validation, 'nyg.validation.out', 'nyg.validation')]

    for poems, out_filename, dataset_filename in to_generate:
        out_idx_filename = os.path.join(poetry_folder, out_filename)
        out_char_filename = os.path.join(poetry_folder, out_filename + '.char')
        generate_out(out_idx_filename, out_char_filename, poems, vocab, args.example_length)

        dataset = Dataset(out_idx_filename)
        dataset_filename = os.path.join(poetry_folder, dataset_filename)
        torch.save(dataset, dataset_filename)
