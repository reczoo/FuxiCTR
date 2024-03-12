import sys
sys.path.append('../')
import os
import logging
from datetime import datetime
from fuxictr import datasets
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
from model_zoo import DeepFM
import pandas as pd 

class CustomFeatureProcessor(FeatureProcessor):
    '''
    In the config/dataset_config.yaml file, the 'extract_country_code' and 'bucketize_age' processors are utilized. 
    Therefore, it is necessary to create a new class that inherits from the 'FeatureProcessor' class and implement 
    the respective functions for these two processors. 
    Each Process Function should accept two arguments: df and col_name.
    '''
    def extract_country_code(self, df, col_name):
        return df[col_name].apply(lambda isrc: isrc[0:2] if not pd.isnull(isrc) else "")

    def bucketize_age(self, df, col_name):
        def _bucketize(age):
            if pd.isnull(age):
                return ""
            else:
                age = float(age)
                if age < 1 or age > 95:
                    return ""
                elif age <= 10:
                    return "1"
                elif age <=20:
                    return "2"
                elif age <=30:
                    return "3"
                elif age <=40:
                    return "4"
                elif age <=50:
                    return "5"
                elif age <=60:
                    return "6"
                else:
                    return "7"
        return df[col_name].apply(_bucketize)
    

if __name__ == '__main__':
    # Load params from config files
    config_dir = './config/example6_config'
    experiment_id = 'DeepFM_test_csv' # corresponds to input `data/tiny_npz`
    params = load_config(config_dir, experiment_id)

    # set up logger and random seed
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # Set feature_encoder that defines how to preprocess data
    use_custom_processor = True 
    if use_custom_processor:
        feature_encoder = CustomFeatureProcessor(feature_cols=params["feature_cols"],
                                        label_col=params["label_col"],
                                        dataset_id=params["dataset_id"], 
                                        data_root=params["data_root"])
    else:
        feature_encoder = FeatureProcessor(feature_cols=params["feature_cols"],
                                        label_col=params["label_col"],
                                        dataset_id=params["dataset_id"], 
                                        data_root=params["data_root"])

    # Build dataset from csv to npz, and remap data paths to npz files
    params["train_data"], params["valid_data"], params["test_data"] = \
        build_dataset(feature_encoder, 
                      train_data=params["train_data"],
                      valid_data=params["valid_data"],
                      test_data=params["test_data"])
    
    # Get feature_map that defines feature specs
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    # Get train and validation data generators
    train_gen, valid_gen = RankDataLoader(feature_map, 
                                          stage='train', 
                                          train_data=params['train_data'],
                                          valid_data=params['valid_data'],
                                          batch_size=params['batch_size'],
                                          shuffle=params['shuffle']).make_iterator()

    # Model initialization and fitting
    model = DeepFM(feature_map, **params)
    model.fit(train_gen, validation_data=valid_gen, epochs=params['epochs'])
    
    logging.info('***** Validation evaluation *****')
    model.evaluate(valid_gen)

    logging.info('***** Test evaluation *****')
    test_gen = RankDataLoader(feature_map, 
                              stage='test',
                              test_data=params['test_data'],
                              batch_size=params['batch_size'],
                              shuffle=False).make_iterator()
    model.evaluate(test_gen)





