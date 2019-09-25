import torch.nn as nn
import torch.optim as optim
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer, ELMoTokenCharactersIndexer
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import TokenCharactersEncoder
from allennlp.training import Trainer

from allennlp.commands.train import   train_model_from_args
from mtl.dataset_readers.MTLDatasetReader import MTLDatasetReader
from mtl.models.MTLSharedClassifier import MTLSharedClassifier

reader = MTLDatasetReader(token_indexers={
    'tokens':SingleIdTokenIndexer(lowercase_tokens=True),
    'elmo': ELMoTokenCharactersIndexer()
}, max_sequence_len=100)
books_train_dataset = reader.read('./data/mtl-dataset/books.task.train')
books_validation_dataset = reader.read('./data/mtl-dataset/books.task.test')
imdb_train_dataset = reader.read('./data/mtl-dataset/imdb.task.train')
imdb_test_dataset = reader.read('./data/mtl-dataset/imdb.task.test')

vocab = Vocabulary.from_instances(books_train_dataset + books_validation_dataset)
iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens", "num_tokens")])
iterator.index_with(vocab)
print(vocab._index_to_token)
# print(vocab.__getstate__()['_token_to_index']['labels'])
# for batch in itera  tor(books_train_dataset, num_epochs=1, shuffle=True):
#     print(batch['tokens']['tokens'], batch['label'])

print(iterator.get_num_batches(books_train_dataset))

books_iter = iter(iterator._create_batches(books_train_dataset, shuffle=True))
print(len(books_train_dataset))

print(next(books_iter).as_tensor_dict())
'''
EMBEDDING_DIM = 300

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM,
                            pretrained_file='/media/sihui/000970CB000A4CA8/Sentiment-Analysis/embeddings/glove.42B.300d.txt',
                            trainable=False)
# character_embedding = TokenCharactersEncoder(embedding=Embedding(num_embeddings=vocab.get_vocab_size('tokens_characters'), embedding_dim=8),
#                                              encoder=CnnEncoder(embedding_dim=8, num_filters=100, ngram_filter_sizes=[5]), dropout=0.2)
word_embeddings = BasicTextFieldEmbedder({'tokens': token_embedding})

# lstm = PytorchSeq2SeqWrapper(nn.LSTM(input_size=308, hidden_size=100, num_layers=1, bidirectional=True, batch_first=True))
seq2vec = CnnEncoder(embedding_dim=300, num_filters=20, ngram_filter_sizes=[3,4,5], output_dim=100)

model = MTLSharedClassifier(vocab=vocab, text_field_embedder=word_embeddings, seq2vec_encoder=seq2vec, dropout=0.5, num_labels=2)
model.cuda(0)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator, train_dataset=books_train_dataset+imdb_train_dataset,
                  validation_dataset=books_validation_dataset,
                  patience=2,
                  num_epochs=20,
                  cuda_device=0)

trainer.train()
'''


