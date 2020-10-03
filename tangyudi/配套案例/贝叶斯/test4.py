import re, collections


def train(features):
    model = collections.defaultdict(lambda: 0)
    for f in features:
        model[f] += 1
    return model


def words(text): return re.findall('[a-z]+', text.lower())


print(words("tianKongZHIJDS"))
features = ["tian", "kong", "zhi", "cheng", "tian"]

model = train(features)
print(model)
