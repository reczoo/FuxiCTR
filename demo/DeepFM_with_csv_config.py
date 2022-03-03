import sys
sys.path.append('../')
import os
import logging
from datetime import datetime
from fuxictr import datasets
from fuxictr.datasets.taobao import FeatureEncoder
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.pytorch.models import DeepFM
from fuxictr.pytorch.torch_utils import seed_everything


if __name__ == '__main__':
    # Load params from config files
    config_dir = 'demo_config'
    experiment_id = 'DeepFM_test' # correponds to csv input `taobao_tiny`
    params = load_config(config_dir, experiment_id)

    # set up logger and random seed
    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    # Set feature_encoder that defines how to preprocess data
    feature_encoder = FeatureEncoder(params['feature_cols'], 
                                     params['label_col'], 
                                     dataset_id=params['dataset_id'], 
                                     data_root=params["data_root"])

    # Build dataset from csv to h5
    datasets.build_dataset(feature_encoder, 
                           train_data=params["train_data"], 
                           valid_data=params["valid_data"], 
                           test_data=params["test_data"])
    
    # Get feature_map that defines feature specs
    feature_map = feature_encoder.feature_map

    # Get train and validation data generator from h5
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    train_gen, valid_gen = datasets.h5_generator(feature_map, 
                                                 stage='train', 
                                                 train_data=os.path.join(data_dir, 'train.h5'),
                                                 valid_data=os.path.join(data_dir, 'valid.h5'),
                                                 batch_size=params['batch_size'],
                                                 shuffle=params['shuffle'])
    
    # Model initialization and fitting                                                  
    model = DeepFM(feature_encoder.feature_map, **params)
    model.count_parameters() # print number of parameters used in model
    model.fit_generator(train_gen, 
                        validation_data=valid_gen, 
                        epochs=params['epochs'],
                        verbose=params['verbose'])
    model.load_weights(model.checkpoint) # reload the best checkpoint
    
    logging.info('***** validation results *****')
    model.evaluate_generator(valid_gen)

    logging.info('***** validation results *****')
    test_gen = datasets.h5_generator(feature_map, 
                                     stage='test',
                                     test_data=os.path.join(data_dir, 'test.h5'),
                                     batch_size=params['batch_size'],
                                     shuffle=False)
    model.evaluate_generator(test_gen)


