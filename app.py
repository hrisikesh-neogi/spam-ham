from ast import Global
from flask import Flask, request
import sys
from flask.helpers import redirect, url_for
from flask.scaffold import F

import pip
from src.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from src.logger import logging
from src.exception import spamhamException
import os, sys
import json
from src.config.configuration import Configuartion
from src.constant import CONFIG_DIR, get_current_time_stamp
from src.pipeline.pipeline import Pipeline
from src.entity.spamham_predictor import  SpamhamPredictor, Get_latest_encoder_object
from flask import send_file, abort, render_template
from src.entity.artifact_entity import ModelTrainerArtifact
import time


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "src"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)




from src.logger import get_log_dataframe

INPUT_TEXT_DATA_KEY = "input_text"
SPAMHAM_PREDICTED_VALUE_KEY = "spamham_prediction_output"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'src'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("src", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)

IS_TRAINING =  False
@app.route('/train', methods=['GET', 'POST'], defaults={'train_model': "False"})
@app.route('/train/<path:train_model>', methods=['GET', 'POST'])
def train(train_model):
    if train_model in request.path.split('/'):
        if "train_model" in request.path:
            message = ""
            
            global TRAINING_PIPELINE_RUNNING_STATUS, IS_TRAINING, pipeline
            
            pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
            TRAINING_PIPELINE_RUNNING_STATUS = pipeline.experiment.running_status
            
            if not TRAINING_PIPELINE_RUNNING_STATUS:
                message = "Training started."
                pipeline.start()
                IS_TRAINING = True
                context = {
                    'model_is_training': True,
                    'message' : "Training started"
                }
                
                
                
                
            else:
                message = "Training is already in progress."
                context = {
                    "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
                    "message": message,
                    'model_is_training': True
                }
            return render_template('train.html', context=context)
            
        
    else:
        print("not doing it")
        model_training_status = False
        message =  "Model is not yet trained"
    
    context={'message': message, 'model_is_training':model_training_status }
    return render_template('train.html',context=context) 


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    
    if not os.path.exists(MODEL_DIR):
        context = {'message': "Could not find model directory: Kindly train your model"}
        return render_template("predict.html", context=context)
        
    
    else:
        
        if IS_TRAINING:
            MODEL_TRAINING_STATUS = pipeline.is_pipeline_running
        else:
            MODEL_TRAINING_STATUS = False
        print("model training status", MODEL_TRAINING_STATUS)
        
        if MODEL_TRAINING_STATUS == True :
            context = {'message': "Training in progress. Please try again later."}
            return render_template("predict.html", context=context)

        else:
            context = {
                'message': None,
                
                INPUT_TEXT_DATA_KEY: None,
                SPAMHAM_PREDICTED_VALUE_KEY: None
            }

            if request.method == 'POST':
                text = request.form['text']
                spamham_predictor = SpamhamPredictor(model_dir=MODEL_DIR)
                spamham_predictor.get_latest_model_path()
                spam_ham_prediction_output = spamham_predictor.predict(X=text)
            
                
                context = {
                    'message': None,
                    INPUT_TEXT_DATA_KEY: text,
                    SPAMHAM_PREDICTED_VALUE_KEY: spam_ham_prediction_output
                }
                
                print("context:", context)
                return render_template('predict.html', context=context)
            
        return render_template("predict.html", context=context)
        
        
        


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
