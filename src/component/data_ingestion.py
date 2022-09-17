from src.entity.config_entity import DataIngestionConfig
import sys, os
from src.exception import spamhamException
from src.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact
import zipfile
import numpy as np
from six.moves import urllib
import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise spamhamException(e, sys)

    def download_housing_data(self) -> str:
        try:
            # extraction remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

            # folder location to download file
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir

            os.makedirs(tgz_download_dir, exist_ok=True)

            housing_file_name = os.path.basename(download_url)

            zip_file_path = os.path.join(tgz_download_dir, housing_file_name)

            logging.info(
                f"Downloading file from :[{download_url}] into :[{zip_file_path}]"
            )
            urllib.request.urlretrieve(download_url, zip_file_path)
            logging.info(f"File :[{zip_file_path}] has been downloaded successfully.")
            return zip_file_path

        except Exception as e:
            raise spamhamException(e, sys) from e

    def extract_zip_file(self, zip_file_path: str):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            os.makedirs(raw_data_dir, exist_ok=True)

            logging.info(
                f"Extracting tgz file: [{zip_file_path}] into dir: [{raw_data_dir}]"
            )
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(raw_data_dir)
            # with tarfile.open(zip_file_path) as spamham_tgz_file_obj:
            #     spamham_tgz_file_obj.extractall(path=raw_data_dir)
                logging.info(f"Extraction completed")

        except Exception as e:
            
            raise spamhamException(e, sys) from e

    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(raw_data_dir)[1]

            spamham_file_path = os.path.join(raw_data_dir, file_name)

            logging.info(f"Reading csv file: [{spamham_file_path}]")
            spamham_dataset = pd.read_csv(spamham_file_path, sep = '\t', names=["class", "text"])

            logging.info(f"Splitting data into train and test")
        
            train_set, test_set = train_test_split(spamham_dataset, test_size=0.2, random_state=42)

           

            train_file_path = os.path.join(
                self.data_ingestion_config.ingested_train_dir, file_name
            )

            test_file_path = os.path.join(
                self.data_ingestion_config.ingested_test_dir, file_name
            )

            if train_set is not None:
                os.makedirs(
                    self.data_ingestion_config.ingested_train_dir, exist_ok=True
                )
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                train_set.to_csv(train_file_path, index=False, sep = '\t')

            if test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                test_set.to_csv(test_file_path, index=False, sep= '\t')

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                is_ingested=True,
                message=f"Data ingestion completed successfully.",
            )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            print("data imhested successfully")
            return data_ingestion_artifact

        except Exception as e:
            raise spamhamException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            zip_file_path = self.download_housing_data()
            self.extract_zip_file(zip_file_path=zip_file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise spamhamException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")
