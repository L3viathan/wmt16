import tensorflow as tf

from ml_app.utils.io_funs import as_project_path
from ml_app.utils.nlp_funs import default_word_standardizer
from ml_app.utils.app_funs import get_extra_features
from ml_app.ngram_builder.bigram_builder import BigramBuilder
from ml_app.feature_builder.bigram_counter import BigramCounter
from ml_app.models.nn import NeuralNetwork
from ml_app.data_provider.full_data_provider import FullDataProvider

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
            'bigram_filter_level': 20,#10#if change this number remember to delete the feature cache file
        },
    },
    'learning_model':{
        #NOTE:
        #bigram over 20: 31986       ; over 10: 72779; over 30:
        #peform over 20: .93:.89:.88 ; over 10:      ; over 30:
        'NeuralNetwork':{
            'layer_description':[
                {   'name': 'input',
                    'unit_size': 31986,#784,#bigram occur over 20
                },
                {   'name': 'hidden1',
                    'active_fun': tf.nn.relu,
                    'unit_size': 2048,#128,
                },
                {   'name': 'output',
                    'active_fun': None,
                    'unit_size': 2,#10
                },
            ],
        },        
        'NeuralNetwork4':{
            'layer_description':[
                {   'name': 'input',
                    'unit_size': 72779,#784,#bigram occur over 10
                },
                {   'name': 'hidden1',
                    'active_fun': tf.nn.relu,
                    'unit_size': 16384,#128,
                },
                {   'name': 'hidden2',
                    'active_fun': tf.nn.relu,
                    'unit_size': 256,#32
                },
                {   'name': 'output',
                    'active_fun': None, 
                    'unit_size': 2,#10 
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
        #NOTE: total train example: 10589
        'model': NeuralNetwork,
        'data_provider': FullDataProvider,
        'learning_rate': 0.0001, #best:0.0001, #0.01,#ok at frst:0.001, 1e-5,#0.01,
        'max_step': 1000,#400,#2000,#400->6 epochs
        'batch_size': 350,
        'step_to_report_loss': 50,
        'step_to_save_eval_model': 100,#100, #200,
        'model_storage_file': as_project_path('data/thanh/models/nn'),
        'load_model_step': 699,
    },
    'logger':{
    },
}
