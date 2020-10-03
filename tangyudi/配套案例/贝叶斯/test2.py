import re, collections


def words(text): return re.findall('[a-z]+', text.lower())


def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


print(collections.defaultdict(lambda: 1))
