import os
import sys

from numpy import vectorize

from src.exception import spamhamException
from src.util.util import load_object, read_yaml_file
from src.config.configuration import Configuartion
from src.constant import *

CONFIG = Configuartion()

config_file = read_yaml_file(CONFIG_FILE_PATH)

import pandas as pd



        
def Get_latest_encoder_object():
    try:
        artifact_dir = CONFIG.get_training_pipeline_config()
        artifact_dir = artifact_dir.artifact_dir
        data_transformation_artifact_dir = DATA_TRANSFORMATION_ARTIFACT_DIR
        
        transformation_path = os.path.join(artifact_dir, data_transformation_artifact_dir)
        latest_file = os.listdir(transformation_path)[-1]
    
        preprocessing_dir = config_file[DATA_TRANSFORMATION_CONFIG_KEY][DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY]
        encoder_file_name = config_file[DATA_TRANSFORMATION_CONFIG_KEY][DATA_TRANSFORMATION_PREPROCESSED_ENCODER_FILE_NAME_KEY]

        encoder_file_path = os.path.join(
            transformation_path ,
            latest_file, 
            preprocessing_dir, 
            encoder_file_name)
        
        print("encoder file path: %s" % encoder_file_path)
        
        encoder_object = load_object(file_path = encoder_file_path)
        print("encoder loaded: %s" % encoder_object)
        return encoder_object
    except Exception as e:
        raise spamhamException(e, sys) from e
        
        
       


class SpamhamPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise spamhamException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise spamhamException(e, sys) from e

    def predict(self, X):
        
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            prediction_value = model.predict([X])
            print("printing int output value", prediction_value)
            
            category_dict = self.get_catagory_dictionary()
            print("printing category dictionary", category_dict)
            
            spam_ham_prediction_output = category_dict[int(prediction_value)]
            print("spam_ham_prediction_output:", spam_ham_prediction_output)
            return spam_ham_prediction_output
        except Exception as e:
            raise spamhamException(e, sys) from e
        
    def get_catagory_dictionary(self) -> dict:
        """
        this function maps encoded categories(e.g. 0,1) into ham or spam.

        Returns:
            dict: _description_
        """
        encoder_object = Get_latest_encoder_object()
        categories = list(encoder_object.categories_[0])
        mapped_categories_dict = dict(zip([num for num in range(len(categories)+1)],categories))
        
        return mapped_categories_dict
        
        # prediction = prediction_dict[int(spam_ham_prediction_output)]