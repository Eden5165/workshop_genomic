import pandas as pd
import numpy as np


# Normalization
def sum_gene_exp_to_num(genes_df, num):
    """
    param genes_df: genes df such that rows are samples and first colum is the sample ID
    param num: 1 or 1000000.
    return: if num=1, returns genes_df_sum such that each cell is divided by the sum of it's column, that
    means, each column (gene) is summed to one. if num = 1 mill, same as 1 only multiplied by 1 mill, 
    That menas, each column (gene) is summed to one mill.
    """
    genes_df_sum = genes_df.copy()
    genes_df_sum.iloc[:, 1:] = genes_df_sum.iloc[:, 1:].divide(genes_df_sum.iloc[:, 1:].sum(axis=0).to_numpy(), axis=1)
    if num == 1:
        return genes_df_sum
    genes_df_sum.iloc[:, 1:] = genes_df_sum.iloc[:, 1:].transform(lambda x: x * 1000000)
    return genes_df_sum


def gene_exp_log_trans(genes_df, summed_to):
    """
    param genes_df: genes df such that rows are samples and first colum is the sample ID.
    param summed_to: what number are the column in genes_df summed to. Can be 1 or 1000000.
    return: If summed_to = 1: transform each value x in genes_df to log2(x + 10e-5). If summed
    to = 1000000: x = log2(x + 10)
    """
    addition = 10
    if summed_to == 1:
        addition = pow(10, -5)
    genes_transformed = genes_df.copy()
    genes_transformed.iloc[:, 1:] = genes_transformed.iloc[:, 1:].transform(lambda x: x + addition).apply(np.log2)
    return genes_transformed


def ic50_log_trans(df):
    pass


def norm_per_sample(df):
    pass


def norm_per_col(df):
    pass


def eliminate_extreme_vals(df):
    pass