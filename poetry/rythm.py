from tokenize_uk import tokenize_sents, tokenize_words

from poetry.accent import transform_into_rythm_map


def transform_lines_into_rythm(lines, accent_vocab):
    rythm = ''
    for line in lines.split('\n'):
        sent_tokens = tokenize_sents(line)
        for sent in sent_tokens:
            for word in filter(lambda w: w in accent_vocab, tokenize_words(sent)):
                accent_options = accent_vocab[word]

                # use only first one, fuck other options
                word, index = accent_options[0]
                rythm_map = transform_into_rythm_map(word, index)
                rythm += rythm_map
        rythm += '\n'
    return rythm

