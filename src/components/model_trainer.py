import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor

from src.exception import Custom_Exception
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=os.path.join('artifacts','model.pkl')

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting Data into train and test data")
            X_train,y_train,X_test,y_test=(train_arr[:,:-1],
                                           train_arr[:,-1],
                                           test_arr[:,:-1],
                                           test_arr[:,-1]
                                           )
            
            models={
                "RandomForestRegressor":RandomForestRegressor(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'LinearRegression':LinearRegression(),
                "GradientBoostRegressor":GradientBoostingRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor()
                }
            model_report=evaluate_models(X_train,y_train,X_test,y_test,models)
            logging.info("Model Training has beend done and now selecting the best model")
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise Custom_Exception("No best Model Found")
            logging.info("Best model has been found and the model is {}".format(best_model_name))
            save_object(file_path=self.model_trainer_config,obj=best_model)
            logging.info("Best Model has been pickled and the path is '{}'".format(self.model_trainer_config))

            predicted=best_model.predict(X_test)
            score=r2_score(y_test,predicted)
            print(best_model_name)
            return score
        
        except Exception as e:
            raise Custom_Exception(e,sys)

