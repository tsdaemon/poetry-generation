import Constants


# vocab object from harvardnlp/opennmt-py
class Vocab(object):
    def __init__(self, filename=None, data=None, lower=False):
        self.idx_to_label = {}
        self.label_to_idx = {}
        self.lower = lower

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            self.add_specials(data)
        if filename is not None:
            self.load_file(filename)

    def __len__(self):
        return len(self.idx_to_label)

    # Load entries from a file.
    def load_file(self, filename):
        for line in open(filename, encoding='utf-8'):
            token = line.rstrip('\n')
            self.add(token)

    def get_index(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.label_to_idx[key]
        except KeyError:
            return default

    def get_label(self, idx, default=None):
        try:
            return self.idx_to_label[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special
    def add_special(self, label, idx=None):
        idx = self.add(label)
        self.special += [idx]

    # Mark all labels in `labels` as specials
    def add_specials(self, labels):
        for label in labels:
            self.add_special(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label):
        label = label.lower() if self.lower else label
        if label in self.label_to_idx:
            idx = self.label_to_idx[label]
        else:
            idx = len(self.idx_to_label)
            self.idx_to_label[idx] = label
            self.label_to_idx[label] = idx
        return idx

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convert_to_idx(self, labels, unk_word):
        vec = []

        unk = self.get_index(unk_word)
        vec += [self.get_index(label, default=unk) for label in labels]

        return vec

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convert_to_labels(self, idx):
        labels = []

        for i in idx:
            labels += [self.get_label(i)]

        return labels


def get_char_vocab(filename, lower=True):
    vocab = Vocab(filename, data=[Constants.SOP_CHAR,
                                  Constants.EOP_CHAR,
                                  Constants.EOL_CHAR,
                                  Constants.PAD_CHAR,
                                  Constants.UNK_CHAR],
                  lower=lower)
    return vocab