import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import Custom_Exception
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=os.path.join('artifacts','preprocessor.pkl')

    def get_data_transformation_object(self):
        '''This function is responsible for 
        Data Transformation'''
        try:
            num_columns = ["writing score", "reading score"]
            cat_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
                ]
            
            num_pipeline=Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='most_frequent')),
                    ('One-Hot-Encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Pipeline has been initiated for Categorical Columns {}".format(cat_columns))
            logging.info("Pipeline has been initiated for Numerical Columns {}".format(num_columns))

            logging.info("Column Transformer has been initiated for both Numerical and Categorical Columns")
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_columns),
                    ('cat_pipeline',cat_pipeline,cat_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise Custom_Exception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Training Data and Test Data has been ")
            logging.info("Obtaining Preprocessor Object")

            target_column_name='math score'
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying Preprocessing object on training and test data')
            preprocessor_obj=self.get_data_transformation_object()
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.fit_transform(input_feature_test_df)

            logging.info('Applied Preprocessor object to training and test Data')
            logging.info("Combining Preprocessed Input datapoints from training and test data with target feature of training and test data")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info('Preprocessed and Concatenation has been done')

            save_object(file_path=self.data_transformation_config,obj=preprocessor_obj)
            return train_arr,test_arr
        except Exception as e:
            raise Custom_Exception(e,sys)
        




