from ml_app.utils.io_funs import as_project_path, get_extra_features
from ml_app.utils.nlp_funs import default_word_standardizer

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
            'debug': True,
            'en_word_standardizer': default_word_standardizer,#lemma extractor
            'fr_word_standardizer': default_word_standardizer, 
            'result_file': as_project_path('data/en_fr_bigram.pkl'), 
        },
    },
    'feature_builder':{
        'BigramCounter':{
            'en_word_standardizer': default_word_standardizer,#lemma extractor
            'fr_word_standardizer': default_word_standardizer, 
            'add_extra_feature': get_extra_features, 
        },
    },
    'logger':{
    },
}
