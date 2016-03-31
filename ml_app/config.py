import tensorflow as tf

from ml_app.utils.io_funs import as_project_path
from ml_app.utils.nlp_funs import default_word_standardizer
from ml_app.utils.app_funs import get_extra_features
from ml_app.ngram_builder.bigram_builder import BigramBuilder
from ml_app.feature_builder.bigram_counter import BigramCounter
from ml_app.models.nn import NeuralNetwork

config = {
    'general':{
        'data_path': as_project_path('data/lett.train'),
        'train_file': as_project_path('data/train.txt'),
        'valid_file': as_project_path('data/valid.txt'),
        'test_file': as_project_path('data/test.txt'),
        'translation_file': as_project_path('data/translations.gz'),
    },
    'ngram_builder':{
        'BigramBuilder':{
            'en_word_standardizer': default_word_standardizer,#lemma extractor
            'fr_word_standardizer': default_word_standardizer, 
            'result_file': as_project_path('data/en_fr_bigram.pkl'), 
        },
    },
    'feature_builder':{
        'BigramCounter':{
            'bigram_builder': BigramBuilder, 
            'en_word_standardizer': default_word_standardizer,#lemma extractor
            'fr_word_standardizer': default_word_standardizer, 
            'add_extra_features': get_extra_features,
            'cache_file': as_project_path('data/en_fr_features.pkl'), 
            'bigram_filter_level': 10,#if change this number remember to delete the feature cache file
        },
    },
    'learning_model':{
        'NeuralNetwork':{
            'layer_description':[
                {   'name': 'input',
                    'unit_size': 784,
                },
                {   'name': 'hidden1',
                    'active_fun': tf.nn.relu,
                    'unit_size': 128,
                },
                {   'name': 'hidden2',
                    'active_fun': tf.nn.relu,
                    'unit_size': 32,
                },
                {   'name': 'output',
                    'active_fun': None, 
                    'unit_size': 10, 
                },
            ],
        },
    },
    'data_provider':{
         'FullDataProvider':{
            'feature_builder': BigramCounter, 
            'train_file': as_project_path('data/train_enriched10.txt'),
            'valid_file': as_project_path('data/valid_enriched10.txt'),
            'test_file': as_project_path('data/test_enriched10.txt'),
         },
    },
    'classifier':{
        'model': NeuralNetwork,
    },
    'logger':{
    },
}
