import pandas as pd
import os

# Testing
def divide_to_folds(genes_df, drugs_df):
    """
    param genes_df: genes df such that rows are samples.
    param drugs_df: drugs df such that rows are samples.
    return: tuple. array of genes df divided to the 5 folds, same with drugs.
    """
    division_fp = os.path.join(os.getcwd(), "folds.txt")
    division_df = pd.read_csv(division_fp, sep='\t', lineterminator='\n', header=None, names=["sampleID", "fold"])
    genes_folds = []
    drugs_folds = []
    for i in range(1, 6):
        samples = division_df[division_df["fold"]==i]["sampleID"]
        genes_folds.append(genes_df[genes_df.index.isin(samples)])
        drugs_folds.append(drugs_df[drugs_df.index.isin(samples)])
    return genes_folds, drugs_folds


def get_mse():
    pass


def export_drugs_prediction(drug_pred_df, file_name):
    """
    param pred_df: drug prediction df such that rows are samples.
    param file_name: the file name under which the df will be saved.
    """
    output_fp = os.path.join(os.getcwd(), "prediction_results", file_name)
    drug_pred_df.to_csv(output_fp, sep="\t", line_terminator='\n', na_rep="NA", index_label=False)

