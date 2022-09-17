from cgi import test
from sklearn import preprocessing
from src.exception import spamhamException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig 
from src.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
import sys,os
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from src.constant import *
from src.util.util import read_yaml_file,save_object,save_numpy_array_data,load_data




class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise spamhamException(e,sys) from e

    

    def get_data_vectorizer_object(self):
        try:
            vectorizer = CountVectorizer(stop_words='english')
            return vectorizer

        except Exception as e:
            raise spamhamException(e,sys) from e   
        

           
            

            
        except Exception as e:
            raise spamhamException(e,sys) from e


    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            vectorizer = self.get_data_vectorizer_object()
            encoder = OrdinalEncoder()


            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]
            feature_column_name = schema[FEATURE_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df[feature_column_name]
            target_feature_train_df = train_df[[target_column_name]]

            input_feature_test_df = test_df[feature_column_name]
            target_feature_test_df = test_df[[target_column_name]]
            
            #applying encoding on the target column
            target_feature_train_df = encoder.fit_transform(target_feature_train_df)
            target_feature_test_df =  encoder.transform(target_feature_test_df)
            
            #dump encoder output
            encoder_file_path = self.data_transformation_config.encoder_file_path
            save_object(file_path=encoder_file_path,obj=encoder)
            logging.info("Encoder object saved to: %s" % encoder_file_path)
            
            
            
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=vectorizer.fit_transform(input_feature_train_df)
            input_feature_test_arr = vectorizer.transform(input_feature_test_df)

            #let's transform to numpy array
            
            input_feature_train_arr =  input_feature_train_arr.toarray()
            input_feature_test_arr = input_feature_test_arr.toarray()

            #concatinating input features and targets
            
            print('train feature',input_feature_train_arr)
            print('test feature',input_feature_test_arr)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            print('concatination done')
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            vectorizer_file_path = self.data_transformation_config.preprocessed_object_file_path
            
            logging.info(f"Saving preprocessing object.")
            save_object(file_path=vectorizer_file_path,obj=vectorizer)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=vectorizer_file_path,
            encoder_file_path = encoder_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            print("data transformation successfull")
            return data_transformation_artifact
        except Exception as e:
            raise spamhamException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")
