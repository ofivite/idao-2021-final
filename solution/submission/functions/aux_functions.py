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


def fill_from_appl(data_folder, df):
    """
    Function to add more features from appl.csv into inpurt dataframe `df`
    """
    df_appl = pd.read_csv(data_folder + 'appl.csv')
    #
    idx = np.intersect1d(df_appl.query('appl_stts_name_dc == "Issuing a card"').index.unique(), df.index)
    df['is_issuing_card'] = np.nan
    df['is_issuing_card'].loc[idx] = 1
    #
    idx = np.intersect1d(df_appl.query('appl_sale_channel_name == "Telesales" or appl_sale_channel_name == "DRKK"').index.unique(), df.index)
    df['is_telesales_or_drkk'] = np.nan
    df['is_telesales_or_drkk'].loc[idx] = 1

    idx = np.intersect1d(df_appl.query('appl_prod_group_name == "Mortgage" or appl_prod_group_name == "PILS"').index.unique(), df.index)
    df['is_mortgage_or_pils'] = np.nan
    df['is_mortgage_or_pils'].loc[idx] = 1
    return df

def prepare_data_dev(data_folder):
    """
    Dev version testing several other features, e.g. from balance, aum and transaction files
    """
    if data_folder[-1] != '/':
        data_folder = data_folder + '/'

    # load csv data files
    df_aum = pd.read_csv(data_folder + 'aum.csv')
    df_appl = pd.read_csv(data_folder + 'appl.csv')
    df_balance = pd.read_csv(data_folder + 'balance.csv')
    df_client = pd.read_csv(data_folder + 'client.csv')
    df_funnel = pd.read_csv(data_folder + 'funnel.csv')
    df_com = pd.read_csv(data_folder + 'com.csv')
    df_mcc = pd.read_csv(data_folder + 'dict_mcc.csv')
    df_trxn = pd.read_csv(data_folder + 'trxn.csv')


    # intial cleaning
    df_client["education_bool"] = (df_client["education"] == 'PRIMARY_PROFESSIONAL') | (pd.isna(df_client["education"]))
    df_client["education_bool"] = 1 - (df_client["education_bool"]).astype('int64')
    cid_idx = df_client[df_client["education_bool"] == 1].client_id

    # merge by client_id
    df_comb_feat = df_funnel
    df_comb_feat["education_bool"] = df_client["education_bool"]
#
    tmp_balance = df_balance.groupby(['client_id'])['avg_bal_sum_rur'].agg(np.mean)
    tmp_aum = df_aum.groupby(['client_id']).agg(np.mean)
    tmp_com = df_com.groupby(['client_id']).last()['count_comm']
    tmp = df_trxn.merge(df_mcc, on='mcc_cd', how='left')
    tmp = tmp[tmp.brs_mcc_group.isin(['Finance', 'Cash'])]

    tmp_trans = tmp.groupby(['client_id', 'brs_mcc_group'])['tran_amt_rur'].agg(np.mean)
    tmp_trans = tmp_trans.unstack()

    df_comb_feat = df_comb_feat.merge(tmp_balance, on='client_id', how='left')
    df_comb_feat = df_comb_feat.merge(tmp_aum, on='client_id', how='left')
    df_comb_feat = df_comb_feat.merge(tmp_com,  on='client_id', how='left')

    df_comb_feat = df_comb_feat.merge(tmp_trans,  on='client_id', how='left')
    df_comb_feat = df_comb_feat.fillna(0)
    # if train data, drop columns that aren't available in test data except for 'sale_flg'
    if 'sale_flg' in list(df_comb_feat.columns):
        df_comb_feat = df_comb_feat.loc[df_comb_feat['client_id'].isin(cid_idx.values)]
        metric_data = df_comb_feat[['education_bool', 'sale_flg', 'client_id', 'sale_amount', 'contacts']]
        df_comb_feat = df_comb_feat.drop(columns=['education_bool', 'client_id', 'sale_amount',
                                                  'contacts', 'region_cd', 'client_segment'])
    return df_comb_feat, metric_data
