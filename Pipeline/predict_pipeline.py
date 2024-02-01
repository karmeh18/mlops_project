import sys
import os
import pandas as pd
from src.exception import Custom_Exception
from src.logger import logging
from src.utils import load_object

class Predict_Pipeline:
    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            logging.info('Model and Preprocessor object has been loaded')

            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise Custom_Exception(e,sys)
        
class CustomData:
    def __init__(self,
                 gender,
                 race,
                 parent,
                 lunch,
                 test,
                 reading,
                 writing):
         self.gender=gender
         self.race=race
         self.parent=parent
         self.lunch=lunch
         self.test=test
         self.reading=reading
         self.writing=writing

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                'race/ethnicity':[self.race],
                'parental level of education':[self.parent],
                'lunch':[self.lunch],
                'test preparation course':[self.test],
                'reading score':[self.reading],
                'writing score':[self.writing]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise Custom_Exception(e,sys)
