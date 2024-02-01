import os


list_of_files=[
    'notebook/data',
    'src/__init__.py',
    'src/components/__init__.py',
    'src/components/data_ingestion.py',
    'src/components/data_transformation.py',
    'src/components/model_trainer.py',
    'Pipeline/__init__.py',
    'Pipeline/train_pipeline.py',
    'Pipeline/predict_pipeline.py',
    'src/exception.py',
    'src/logger.py',
    'src/utils.py',
    'README.md',
    'requirements.txt',
    'setup.py'
]

for files in list_of_files:
    filedir,filename=os.path.split(files)

    if filedir !="":
        os.makedirs(filedir,exist_ok=True)

    if not os.path.exists(files) or os.path.getsize(files)==0:
        with open(files,"w"):
            pass