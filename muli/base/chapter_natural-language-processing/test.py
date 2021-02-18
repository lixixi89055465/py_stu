from mxnet import gluon
from mxnet import nd
from mxnet.contrib import text

text.embedding.get_pretrained_file_names('fasttext')[:10]

text_data=' hello world \n hello nice world \n hi world \n'
counter=text.utils.count_tokens_from_str(text_data)

print(counter)


my_vocab=text.vocab.Vocabulary(counter)
my_embedding=text.embedding.create(
'fasttext',pretrained_file_name='wiki.simple.vec',vocabulary=my_vocab)





