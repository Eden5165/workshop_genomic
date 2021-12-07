import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd)

import pandas as pd
from sklearn.metrics import mean_squared_error
from utils import data_prep_utils

# Testing
def divide_to_folds(dfs):
    """
    param dfs: a list of dfs to divide.
    return: a list of splited dfs.
    """
    division_fp = os.path.join(os.getcwd(), "medical_genomics_2021_data", "folds.txt")
    division_df = pd.read_csv(division_fp, sep='\t', lineterminator='\n', header=None, names=["sampleID", "fold"])
    dfs_folds=[]
    for df in dfs:
        df_folds = []
        for i in range(1, 6):
            samples = division_df[division_df["fold"]==i]["sampleID"]
            df_folds.append(df[df.index.isin(samples)])
        dfs_folds.append(df_folds)
    return dfs_folds


def get_mse(true_drugs, pred_drugs):
    """
    param true_drugs: data frame of original drugs data such taht rows are samples, values are
    log transformed, no NaN values.
    param pred_drugs: data frame of drugs prediction such that rows are samples, values are
    log transformed.
    """
    return mean_squared_error(true_drugs, pred_drugs)


def export_drugs_prediction(drug_pred_df, file_name):
    """
    param pred_df: drug prediction df such that rows are samples.
    param file_name: the file name under which the df will be saved.
    """
    output_fp = os.path.join(os.getcwd(), "prediction_results", file_name)
    drug_pred_df.transpose().to_csv(output_fp, sep="\t", line_terminator='\n', na_rep="NA", index_label=False)

#utils
def merge_by_index(df1, df2):
    """
    get 2 data frames and return merge table by index
    """
    return  pd.merge(df1, df2, left_index=True, right_index=True)

def convert_predict_to_df(prediction, col_names, row_names):
    """
    param prediction: result from model prediction
    param col_names: list of the predicted values names, for examples drugs names list
    param row_names: list of samples indexes/names, for examples sample num
    """
    return pd.DataFrame(prediction,columns=col_names, index=row_names)