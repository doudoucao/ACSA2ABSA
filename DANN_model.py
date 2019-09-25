import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import torch.nn.init as init

from layers.RNNencoder import RNNEncoder
from layers.SequenceDropout import SequenceDropout
from layers.similarity_functions import LinearSimilarity
from layers.Attention import LegacyMatrixAttention
from layers.utils import *


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()*ctx.constant
        return grad_output, None


class ACSA2ABSA(nn.Module):
    def __init__(self, args, word_embeddings):
        super(ACSA2ABSA, self).__init__()
        self.args = args
        self.vocab_size = len(word_embeddings)
        self.hidden_size = args.hidden_size
        self.embedding_dim = args.embedding_dim
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.seq_len = args.max_seq_len

        self._emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self._emb.from_pretrained(torch.tensor(word_embeddings)).float()
        self._emb.weight.requires_grad = False

        self._rnn_dropout = SequenceDropout(p=0.35)
        self._dropout = nn.Dropout(p=self.dropout)

        self._encoder = RNNEncoder(nn.GRU,
                                   self.args.embedding_dim,
                                   self.args.hidden_size,
                                   bidirectional=True)

        self.linear_similarity = LinearSimilarity(tensor_1_dim=2*self.hidden_size, tensor_2_dim=2*self.hidden_size,
                                                  combination='x,y,x-y,x*y')

        self._coattention = LegacyMatrixAttention(similarity_function=self.linear_similarity)

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size, 3*self.hidden_size), nn.ReLU())

        self._inference_encoder = RNNEncoder(nn.GRU,
                                             3*self.hidden_size,
                                             self.hidden_size,
                                             bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(4*self.hidden_size, 2*self.hidden_size),
                                             #nn.BatchNorm1d(self.hidden_size),
                                             nn.ReLU(),
                                             #nn.Dropout(p=self.dropout),
                                             nn.Linear(2*self.hidden_size, self.num_classes)
                                             # nn.LogSoftmax()
                                             )

        self._domain_classification = nn.Sequential(nn.Linear(4*self.hidden_size, 2*self.hidden_size),
                                                    # nn.BatchNorm1d(self.hidden_size),
                                                    nn.ReLU(),
                                                    nn.Linear(2*self.hidden_size, 2)
                                                    # nn.LogSoftmax()
                                                    )

    def forward(self, text, aspect, alpha):
        # text: [batch_size, text_len]
        # aspect: [batch_size, aspect_len]
        embedded_text = self._emb(text)
        embedded_aspect = self._emb(aspect)

        text_len = torch.sum(text != 0, dim=-1)
        aspect_len = torch.sum(aspect != 0, dim=-1)

        batch_size = embedded_text.size(0)

        text_mask = get_mask_from_sequence_len(text, text_len).to(self.args.device)
        aspect_mask = get_mask_from_sequence_len(aspect, aspect_len).to(self.args.device)

        embedded_text = self._rnn_dropout(embedded_text)
        embedded_aspect = self._rnn_dropout(embedded_aspect)

        encode_text = self._encoder(embedded_text, text_len)
        encode_aspect = self._encoder(embedded_aspect, aspect_len)

        encode_dim = encode_text.size(-1)

        # batch_size, text_len, aspect_len
        attention_matrix = self._coattention(encode_text, encode_aspect)
        # batch_size, text_len, aspect_len
        text_aspect_attention = masked_softmax(attention_matrix, aspect_mask, memory_efficient=True)
        # batch_size, text_len, hidden_size -- > batch, text_len, aspect_len bmm batch, aspect_len, dim
        text_aspect_vectors = weighted_sum(encode_aspect, text_aspect_attention)

        masked_similarity = replace_masked_values(attention_matrix, aspect_mask.unsqueeze(1), -1e7)
        # batch_size, text_length
        aspect_text_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # batch_size, text_length
        aspect_text_attention = masked_softmax(aspect_text_similarity, text_mask, memory_efficient=True)
        # batch_size, encoding_dim
        aspect_text_vector = weighted_sum(encode_text, aspect_text_attention)
        # batch_size, text_len, encoding_dim
        aspect_text_vector = aspect_text_vector.unsqueeze(1).expand(batch_size, encode_text.size(1), encode_dim)

        final_merage_text = torch.cat([encode_text, text_aspect_vectors,
                                       encode_text*text_aspect_vectors,
                                       encode_text*aspect_text_vector], dim=-1)

        modeled_text = self._projection(final_merage_text)

        projected_enhanced_text = self._rnn_dropout(modeled_text)

        # batch_size, text_len, 2*hidden_size
        inference_text = self._inference_encoder(projected_enhanced_text, text_len)

        inference_text_max = replace_masked_values(inference_text, text_mask.unsqueeze(-1), -1e7).max(dim=1)[0]
        inference_text_avg = torch.sum(inference_text*text_mask.unsqueeze(-1), dim=1)/torch.sum(text_mask, 1, keepdim=True)

        # batch_size, text_len, encoding_dim*4
        inference_text_all = torch.cat([inference_text_max, inference_text_avg], dim=-1)

        if self._dropout:
            inference_text_all = self._dropout(inference_text_all)

        class_output = self._classification(inference_text_all)

        reverse_feature = GradReverse.apply(inference_text_all, alpha)
        domain_output = self._domain_classification(reverse_feature)

        return class_output, domain_output