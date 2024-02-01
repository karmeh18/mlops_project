from flask import Flask,request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Pipeline.predict_pipeline import CustomData, Predict_Pipeline

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('predict.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race=request.form.get('ethnicity'),
            parent=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test=request.form.get('test_preparation_course'),
            reading=request.form.get('reading_score'),
            writing=request.form.get('writing_score')
            )
        
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline=Predict_Pipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('predict.html',results=np.round(results[0],3))
    
if __name__=='__main__':
    app.run(host="0.0.0.0",debug=True)