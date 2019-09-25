import os
import random
import pickle
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training import Trainer

from DANN_model import ACSA2ABSA
from mtl.dataset_readers.ACSADatasetReader import ACSADatasetReader
from mtl.dataset_readers.ABSADatasetReader import ABSADatasetReader
from DANN_model import ACSA2ABSA


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname, path):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix: ', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))
        word_vec = _load_word_vec(path, word2idx)
        for word, i in word2idx.items():
            if word == '@@PADDING@@':
                continue
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
            else:
                embedding_matrix[i] = np.random.normal(scale=0.1, size=embed_dim)
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


# load data
def train(args):
    source_reader = ACSADatasetReader(max_sequence_len=args.max_seq_len)
    target_reader = ABSADatasetReader(max_sequence_len=args.max_seq_len)

    source_dataset_train = source_reader.read('./data/MGAN/data/restaurant/train.txt')
    source_dataset_dev = source_reader.read('./data/MGAN/data/restaurant/test.txt')

    target_dataset_train = target_reader.read('/media/sihui/000970CB000A4CA8/Sentiment-Analysis/data/semeval14/Restaurants_Train.xml.seg')
    target_dataset_dev = target_reader.read('/media/sihui/000970CB000A4CA8/Sentiment-Analysis/data/semeval14/Restaurants_Test_Gold.xml.seg')

    vocab = Vocabulary.from_instances(source_dataset_train + source_dataset_dev + target_dataset_train + target_dataset_dev)
    word2idx = vocab.get_token_to_index_vocabulary()
    print(word2idx)
    embedding_matrix = build_embedding_matrix(word2idx, 300, './embedding/embedding_res_res.dat', '/media/sihui/000970CB000A4CA8/Sentiment-Analysis/embeddings/glove.42B.300d.txt')

    iterator = BucketIterator(batch_size=args.batch_size, sorting_keys=[('text', 'num_tokens'), ('aspect', 'num_tokens')])
    iterator.index_with(vocab)

    my_net = ACSA2ABSA(args, word_embeddings=embedding_matrix)

    optimizer = optim.Adam(my_net.parameters(), lr=args.learning_rate)
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    my_net = my_net.to(args.device)
    loss_class = loss_class.to(args.device)
    loss_domain = loss_domain.to(args.device)

    n_epoch = args.epoch

    max_test_acc = 0
    best_epoch = 0

    data_target_iter = iter(iterator(target_dataset_train, shuffle=True))
    # iterator over it forever

    for epoch in range(n_epoch):
        len_target_dataloader = iterator.get_num_batches(target_dataset_train)
        len_source_dataloader = iterator.get_num_batches(source_dataset_train)
        data_source_iter = iter(iterator._create_batches(source_dataset_train, shuffle=True))
        # data_target_iter = iter(iterator._create_batches(target_dataset_train, shuffle=True))
        s_correct, s_total = 0, 0
        i = 0
        while i < len_source_dataloader:
            my_net.train()
            p = float(i + epoch * len_target_dataloader) / n_epoch / len_target_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # train model using source data
            data_source = next(data_source_iter).as_tensor_dict()
            s_text, s_aspect, s_label = data_source['text']['tokens'], data_source['aspect']['tokens'], data_source['label']
            batch_size = len(s_label)

            s_domain_label = torch.zeros(batch_size).long().to(args.device)

            my_net.zero_grad()

            s_text, s_aspect, s_label = s_text.to(args.device), s_aspect.to(args.device), s_label.to(args.device)
            s_class_output, s_domain_output = my_net(s_text, s_aspect, alpha)

            err_s_label = loss_class(s_class_output, s_label)
            # err_s_domain = loss_domain(s_domain_output, s_domain_label)

            # training model using target data
            # data_target = next(data_target_iter).as_tensor_dict()
            '''
            data_target = next(data_target_iter)
            t_text, t_aspect, t_label = data_target['text']['tokens'], data_target['aspect']['tokens'], data_target['label']

            batch_size = len(t_label)
            t_domain_label = torch.ones(batch_size).long().to(args.device)

            t_text, t_aspect, t_label = t_text.to(args.device), t_aspect.to(args.device), t_label.to(args.device)

            t_class_output, t_domain_output = my_net(t_text, t_aspect, alpha)
            # err_t_domain = loss_domain(t_domain_output, t_domain_label)
            '''
            # loss = err_t_domain + err_s_domain + err_s_label
            loss = err_s_label
            loss.backward()

            if args.use_grad_clip:
                clip_grad_norm_(my_net.parameters(), args.grad_clip)

            optimizer.step()

            i += 1

            s_correct += (torch.argmax(s_class_output, -1) == s_label).sum().item()
            s_total += len(s_class_output)
            train_acc = s_correct / s_total

            # evaluate every 50 batch
            if i % 100 == 0:
                my_net.eval()
                # evaluate model on source test data
                s_test_correct, s_test_total = 0, 0
                s_targets_all, s_output_all = None, None
                with torch.no_grad():
                    for i_batch, s_test_batch in enumerate(iterator(source_dataset_dev, num_epochs=1, shuffle=False)):
                        s_test_text = s_test_batch['text']['tokens'].to(args.device)
                        s_test_aspect = s_test_batch['aspect']['tokens'].to(args.device)
                        s_test_label = s_test_batch['label'].to(args.device)

                        s_test_output, _ = my_net(s_test_text, s_test_aspect, alpha)

                        s_test_correct += (torch.argmax(s_test_output, -1) == s_test_label).sum().item()
                        s_test_total += len(s_test_label)

                        if s_targets_all is None:
                            s_targets_all = s_test_label
                            s_output_all = s_test_output
                        else:
                            s_targets_all = torch.cat((s_targets_all, s_test_label), dim=0)
                            s_output_all = torch.cat((s_output_all, s_test_output), dim=0)

                s_test_acc = s_test_correct / s_test_total
                if s_test_acc > max_test_acc:
                    max_test_acc = s_test_acc
                    best_epoch = epoch
                    if not os.path.exists('state_dict'):
                        os.mkdir('state_dict')
                    if s_test_acc > 0.868:
                        path = 'state_dict/source_test_epoch{0}_acc_{1}'.format(epoch, round(s_test_acc, 4))
                        torch.save(my_net.state_dict(), path)

                print('epoch: %d, [iter: %d / all %d], loss_s_label: %f, '
                      's_train_acc: %f, s_test_acc: %f'% (epoch, i, len_source_dataloader,
                                                                             err_s_label.cpu().item(),
                                                                             #err_s_domain.cpu().item(),
                                                                             #err_t_domain.cpu().item(),
                                                                             train_acc,
                                                                             s_test_acc))
    print('max_test_acc: {0} in epoch: {1}'.format(max_test_acc, best_epoch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=float)
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.35, type=float)
    parser.add_argument('--grad_clip', default=10.0, type=float)
    parser.add_argument('--use_grad_clip', default=True, type=bool)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--embedding_dim', default=300, type=int)
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()