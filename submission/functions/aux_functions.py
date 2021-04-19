import pandas as pd
import numpy as np

def prepare_data(data_folder):
    
    ''' Input: 
        
        data folder: str (path to the data folder) 
        
        Output:
        
        df_comb_feat: pandas DataFrame (features used to train the lightgbm model) 
        
    '''
        
    # add '/' if data folder is provided without it
    if data_folder[-1] != '/':
        data_folder = data_folder + '/'
    
    # load csv data files
    df_client = pd.read_csv(data_folder + 'client.csv')
    df_funnel = pd.read_csv(data_folder + 'funnel.csv')
    
    # intial cleaning:
    # define education_bool feature that checks if the education column in client.csv is equal to NaN or PRIMARY_PROFESSIONAL; 
    # if it's true, we don't use these samples for training and mark such samples with sale_flg = 0 for test data
    df_client["education_bool"] = (df_client["education"] == 'PRIMARY_PROFESSIONAL') | (pd.isna(df_client["education"])) 
    df_client["education_bool"] = 1 - (df_client["education_bool"]).astype('int64')
    cid_idx = df_client[df_client["education_bool"] == 1].client_id
    
    # funnel.csv was used to train the lightgbm model
    df_comb_feat = df_funnel                          
    df_comb_feat["education_bool"] = df_client["education_bool"]
    
    # for training, remove rows based on education_bool feature, and drop 'education_bool', 'client_id', 'sale_amount' and 'contacts' columns, since they are not used for training
    if 'sale_flg' in list(df_comb_feat.columns):
        df_comb_feat = df_comb_feat.loc[df_comb_feat['client_id'].isin(cid_idx.values)]
        df_comb_feat = df_comb_feat.drop(columns=['education_bool', 'client_id', 'sale_amount', 'contacts'])
    
    return df_comb_feat




