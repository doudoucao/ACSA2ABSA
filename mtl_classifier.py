from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

@Model.register("mtl_classifier")
class MTLClassifier(Model):
    def __init__(self, vocab, text_field_embedder,
                 seq2vec_encoder, seq2seq_encoder=None, dropout=None,
                 num_labels=None, label_namespace='labels',
                 initializer=InitializerApplicator()):
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        if seq2seq_encoder:
            self.source_seq2seq_encoder = seq2seq_encoder
            self.target_seq2seq_encoder = seq2seq_encoder
            self.shared_seq2seq_encoder = seq2seq_encoder
        else:
            self.source_seq2seq_encoder = None
            self.target_seq2seq_encoder = None
            self.shared_seq2seq_encoder = None

        self.source_seq2vec_encoder = seq2vec_encoder
        self.target_seq2vec_encoder = seq2vec_encoder
        self.shared_seq2vec_encoder = seq2vec_encoder

        self._classifier_input_dim = self.source_seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=label_namespace)

        self.source_classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self.target_classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self, source_tokens, source_label, target_tokens, target_label):
        source_embedded_text = self._text_field_embedder(source_tokens)
        source_mask = get_text_field_mask(source_tokens).float()

        if self.source_seq2seq_encoder:
            source_embedded_text = self._seq2seq_encoder(source_embedded_text, mask=source_mask)

        source_embedded_text = self._seq2vec_encoder(source_embedded_text, mask=source_mask)

        if self._dropout:
            source_embedded_text = self._dropout(source_embedded_text)

        source_logits = self._classification_layer(source_embedded_text)
        source_probs = torch.nn.functional.softmax(source_logits, dim=-1)

        output_dict = {"source_logits": source_logits, "probs": source_probs}

        if source_label is not None:
            source_loss = self._loss(source_logits, source_label.long().view(-1))
            output_dict['source_loss'] = source_loss
            # self._accuracy(source_logits, source_label)

        return output_dict

    def get_metrics(self, reset: bool = False):
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics


