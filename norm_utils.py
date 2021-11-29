import pandas as pd

# Normalization
def sum_gene_exp_to_one(genes_df):
    """
    param genes_df: genes df such that rows are samples and first colum is the sample ID
    return: genes_df_sum_one: each cell is divided by the sum of it's column, that
    means, each column (gene) is summed to one
    """
    genes_df_sum_one = genes_df.copy()
    genes_df_sum_one.iloc[:, 1:] = genes_df_sum_one.iloc[:, 1:].divide(genes_df_sum_one.iloc[:, 1:].sum(axis=0).to_numpy(), axis=1)
    return genes_df_sum_one



def sum_gene_exp_to_mil(dg):
    pass


def gene_exp_log_trans(df):
    pass


def ic50_log_trans(df):
    pass


def norm_per_sample(df):
    pass


def norm_per_col(df):
    pass


def eliminate_extreme_vals(df):
    pass