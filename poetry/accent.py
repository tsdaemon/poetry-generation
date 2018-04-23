import re
import pandas as pd

vowel_regex = re.compile('[АЕЄИІЇОУЮЯаеєиіїоуюя]')


def transform_into_rythm_map(word, index):
    return ''.join(map(lambda m: '\'' if index == m.start() else '_', vowel_regex.finditer(word)))


def read_vocab(file_name):
    vocab = {}
    df = pd.read_csv(file_name)

    for _, row in df.iterrows():
        word = row['Word']
        index = row['Accent index']

        rythm_map = transform_into_rythm_map(word, index)

        if word in vocab:
            vocab[word] = vocab[word] + [rythm_map]
        else:
            vocab[word] = [rythm_map]
    return vocab
