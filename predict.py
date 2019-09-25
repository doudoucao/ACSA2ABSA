import torch
import torch.nn
import os

from allennlp.models.archival import load_archive
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.predictors import Predictor
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from mtl.dataset_readers import MTLDatasetReader
from mtl.models.ABSASharedNetwork import ABSASharedNetwork
'''
archive = load_archive('./checkpoint/model.tar.gz')
predictor = Predictor.from_archive(archive=archive, predictor_name='mtl_shared_classifier')
instance = predictor._dataset_reader.read('./data/mtl-dataset/books.task.test')
'''

config = Params.from_file(os.path.join('checkpoint4', 'config.json'))
# print(config.get('model'))
# config.loading_from_archive = True

weights_path = os.path.join('checkpoint4', 'best.th')

Model.register('absa')
model = ABSASharedNetwork._load(config.duplicate(),
                   weights_file=weights_path,
                   serialization_dir='checkpoint4',
                   cuda_device=0)
print(model)

