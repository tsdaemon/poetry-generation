SOP = 0
EOP = 1
EOL = 2
PAD = 3
UNK = 4

# start of poem
SOP_CHAR = '~'
# end of poem
EOP_CHAR = '#'
EOL_CHAR = '%'
PAD_CHAR = '\\'
UNK_CHAR = '^'


def preprocess_poem(poem):
    # start char, poem with replaced line breaks, end char, padding
    poem = SOP_CHAR + poem.replace('\n', EOL_CHAR) + EOP_CHAR
    return poem


def postprocess_poem(poem):
    return poem.replace(SOP_CHAR, '') \
        .replace(EOP_CHAR, '') \
        .replace(EOL_CHAR, '\n') \
        .replace(PAD_CHAR, '') \
        .replace(PAD_CHAR, '')
