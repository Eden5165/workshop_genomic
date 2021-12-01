import pandas as pd
import numpy as np
import math


# Normalization
def sum_gene_exp_to_num_per_sample(genes_df, num):
    pass

def gene_exp_log_trans(genes_df, summed_to=1000000):
    """
    param genes_df: genes df, trasposed or not.
    param summed_to: what number are the column in genes_df summed to. Can be 1 or 1000000.
    return: If summed_to = 1: transform each value x in genes_df to log2(x + 10e-5). If summed
    to = 1000000: x = log2(x + 10)
    """
    addition = 10
    if summed_to == 1:
        addition = pow(10, -5)
    return genes_df.transform(lambda x: x + addition).apply(np.log2)


def ic50_log_trans(drugs_df):
    """
    param genes_df: genes df, trasposed or not.
    Will fill NA values with max value = 10, not recommenced, please fill NA values as wanted in advance.
    return: Transform each value x to log10(x).
    """
    return drugs_df.fillna(10.0).apply(np.log10)


def norm_per_sample(df):
    pass


def norm_per_col(df):
    pass


def eliminate_extreme_vals(df):
    pass