import configparser
import pathlib as path
import logging

import numpy as np
import pandas as pd
import joblib

from functions.aux_functions import prepare_data

from SimpleModel import BoostingModel

logging.basicConfig(format='%(asctime)s %(message)s', filename='training.log', level=logging.DEBUG)


def main(cfg):
    # parse config
    DATA_FOLDER = path.Path(cfg["DATA"]["DatasetPath"])
    MODEL_PATH = path.Path(cfg["MODEL"]["FilePath"])
    
    # get training data (see functions/aux_functions for more details)
    train_data = prepare_data(cfg["DATA"]["DatasetPath"])
    
    # split data to features and labels
    X = train_data.drop(columns=['sale_flg'])
    y = train_data['sale_flg']
    
    # fit the model
    model = BoostingModel()
    model.fit(X, y)
    # save the model
    joblib.dump(model, MODEL_PATH)
    logging.info("model was trained")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
