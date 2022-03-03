import sys
sys.path.append('../')
import os
import logging
from datetime import datetime
from fuxictr import datasets
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.pytorch.models import DeepFM
from fuxictr.pytorch.torch_utils import seed_everything


if __name__ == '__main__':
    # Load params from config files
    config_dir = 'demo_config'
    experiment_id = 'DeepFM_test_h5' # correponds to h5 input `taobao_tiny_h5`
    params = load_config(config_dir, experiment_id)

    # set up logger and random seed
    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    # Load feature_map from json
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"))

    # Get train and validation data generator from h5
    train_gen, valid_gen = datasets.h5_generator(feature_map, 
                                                 stage='train', 
                                                 train_data=os.path.join(data_dir, 'train.h5'),
                                                 valid_data=os.path.join(data_dir, 'valid.h5'),
                                                 batch_size=params['batch_size'],
                                                 shuffle=params['shuffle'])
    
    # Model initialization and fitting                                                  
    model = DeepFM(feature_map, **params)
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


