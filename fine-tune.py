# how to load model
import os
import torch.nn as nn
import argparse
import torch

from allennlp.training import Trainer
from allennlp.models.model import Model
from allennlp.common.params import Params
from allennlp.commands import fine_tune

from mtl.models.ABSASharedNetwork import ABSASharedNetwork
from mtl.dataset_readers.ABSADatasetReader import ABSADatasetReader
from mtl.dataset_readers.ACSADatasetReader import ACSADatasetReader

config = Params.from_file(os.path.join('checkpoint5', 'config.json'))
# print(config.get('model'))
# config.loading_from_archive = True

weights_path = os.path.join('checkpoint5', 'best.th')

Model.register('absa')
model = ABSASharedNetwork._load(config.duplicate(),
                   weights_file=weights_path,
                   serialization_dir='checkpoint5',
                   cuda_device=0)


def fine_tune(args):
    source_model = model.to(args.device)

    discriminator = nn.Sequential(
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    ).to(args.device)

    source_reader = ACSADatasetReader(max_sequence_len=args.max_seq_len)
    target_reader = ABSADatasetReader(max_sequence_len=args.max_seq_len)

    source_dataset_train = source_reader.read('./data/MGAN/data/restaurant/train.txt')
    source_dataset_dev = source_reader.read('./data/MGAN/data/restaurant/test.txt')

    target_dataset_train = target_reader.read(
        '/media/sihui/000970CB000A4CA8/Sentiment-Analysis/data/semeval14/Restaurants_Train.xml.seg')
    target_dataset_dev = target_reader.read(
        '/media/sihui/000970CB000A4CA8/Sentiment-Analysis/data/semeval14/Restaurants_Test_Gold.xml.seg')

    all_datasets = {'train': target_dataset_train, 'validation': target_dataset_dev}

    vocab = source_model.vocab
    vocab.extend_from_instances(params={}, instances=(instance for key, dataset in all_datasets for instance in dataset))
    print(vocab)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, type=float)
    parser.add_argument('--learning_rate', default=0.0008, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--embedding_dim', default=300, type=int)
    args = parser.parse_args()

    fine_tune(args)

if __name__ == '__main__':
    main()








