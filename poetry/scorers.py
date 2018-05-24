class ScorerBase:
    def score(self, text):
        return 0 # max score - 1, it will be weighted in the environment


class RightWordsScorer(ScorerBase):
    def __init__(self, words):
        self.words = set(words)

    def score(self, text):
        tokens = list(map(lambda x: x.strip(), text.split(' ')))
        n_tokens = len(tokens)
        n_right = sum([1 if token in self.words else 0 for token in tokens])
        score = (n_right/n_tokens)-0.5
        return score


class SonetScorer(ScorerBase):
    def score(self, text):
        # follows pattern 4 4 3 3
        lines = text.split('\n')
        score = 0
        if len(lines) > 4 and lines[4].strip() == '':
            score += 1 / 3
        if len(lines) > 9 and lines[9].strip() == '':
            score += 1 / 3
        if len(lines) > 12 and lines[12].strip() == '':
            score += 1 / 3
        return score


class MultyScorer(ScorerBase):
    def __init__(self, words):
        self.scorers = [
            # (SonetScorer(), 1.0),
            (RightWordsScorer(words), 10.0)
        ]

    def score(self, text):
        return sum(map(lambda x: x[0].score(text) * x[1], self.scorers))
