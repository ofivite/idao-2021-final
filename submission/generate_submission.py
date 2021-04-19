import configparser
import pathlib as path

import numpy as np
import pandas as pd
import joblib

from functions.aux_functions import prepare_data

def main(cfg):
    # parse config
    DATA_FOLDER = path.Path(cfg["DATA"]["DatasetPath"])
    USER_ID = cfg["COLUMNS"]["USER_ID"]
    PREDICTION = cfg["COLUMNS"]["PREDICTION"]
    MODEL_PATH = path.Path(cfg["MODEL"]["FilePath"])
    SUBMISSION_FILE = path.Path(cfg["SUBMISSION"]["FilePath"])
    
    # prepare data (see functions/aux_functions.py for more details)
    test_data = prepare_data(cfg["DATA"]["DatasetPath"])
    
    # idx of samples with education_bool == 0
    unedu_idx = test_data[test_data["education_bool"] == 0].client_id
    
    # drop unecessary columns
    X = test_data.drop(columns=['client_id', 'education_bool'])
    model = joblib.load(MODEL_PATH)
    
    # generate submission
    submission = test_data[[USER_ID]].copy()
    submission[PREDICTION] = model.predict(X)
    
    # manually change prediction for samples with education_bool == 0 to 0
    submission = submission.set_index('client_id')
    submission.loc[unedu_idx] = 0
    submission = submission.reset_index()  
    
    # save submission to a csv file  
    submission.to_csv(SUBMISSION_FILE, index=False)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
