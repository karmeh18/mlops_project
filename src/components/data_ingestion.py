import os
import sys

from src.exception import Custom_Exception
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split


import pandas as pd


class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    

    def initiate_data_ingestion(self):
        logging.info('Entered in Data Ingestion to Import the file')
        try:
            df=pd.read_csv('notebook\Data\Stud.csv')
            logging.info("Data has been loaded and read as DataFrame")

            logging.info('Training Data Directory has been created and an csv file has been exported')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train-Test split has been initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            logging.info("Splitting has been done and now training data will be exported to the path '{}'".format(self.ingestion_config.train_data_path))
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            logging.info("Splitting has been done and now testing data will be exported to the path '{}'".format(self.ingestion_config.test_data_path))
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion has been Completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        except Exception as e:
            raise Custom_Exception(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr=data_transformation.initiate_data_transformation(train_path=train_data_path,test_path=test_data_path)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
    

