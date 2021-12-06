import os
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer



# Data preperation
def get_df(table_name, sep='\t', lineterminator='\n'):
    """
    param table_name: in ["beat_drug", "beat_rnaseq", "drug_mut_cor", "drug_mut_cor_lables", "tcga_mut", "tcga_rna"]
    param sep: Columns seperator characters. Optional, default='\t'
    param lineterminator: Rows seperator characters. Optional, default='\n'
    return: df for the requested table
    """
    table_fp = os.path.join(os.path.dirname(os.getcwd()), "medical_genomics_2021_data", table_name)
    return pd.read_csv(table_fp, sep=sep, lineterminator=lineterminator)


def filter_beat_and_tcga_by_shared_genes(beat_rna, tcga_rna):
    """
    param beat_rna tcga_rna: original data frames such that rows are genes
    return: beat_rna df  and tcga rna df only with genes that appear in both dfs
    """
    return beat_rna[beat_rna.index.isin(tcga_rna.index)], tcga_rna[tcga_rna.index.isin(beat_rna.index)]


def transpose_df(df):
    """
    param df: df in wich columns are samples
    return: trsanposed df such that first column named SampleID, rows are samples
    """
    return df.transpose()


def get_data_reorgenized(genes_df, other_data_df):
    """
    param genes_df: genes df such that rows are samples
    param other_data_df: drugs or mutations df such that rows are sample
    return: other_data_df reoregenized such that row i represents the sample i that is
    represented by row i in genes_df
    """
    merged = pd.merge(genes_df, other_data_df, left_index=True, right_index=True)
    first_other_field = len(genes_df.columns)
    last_other_field = len(merged.columns)
    data_reorg = merged.iloc[:, list(range(first_other_field, last_other_field))]
    return data_reorg


# Missing values
def missing_vals_reg(drugs_df):
    """

    """
    pass


def missing_vals_knn(drugs_df, k=5):
    """
    param drugs_df: drugs df as returned from data_prep_utils.get_df. That means, rows are drugs.
    param k: number of neighbors with which to run the KNN algorithm.
    return: drugs_full with missing values filled using the KNN algorithm.
    """
    drugs_full = drugs_df.copy()
    imputer = KNNImputer(n_neighbors=k)
    drugs_full.iloc[:, :] = imputer.fit_transform(drugs_full)
    return drugs_full
    

def missing_vals_method(drugs_df, method):
    """
    param drugs_df: drugs df transposed. That means, rows are samples.
    param method: In: ["max", "mean"] A mathematical method to fill the missing values be.
    return: NaN values filled by requested method, per drug.
    """
    values = {
        "max": drugs_df.max,
        "mean": drugs_df.mean
    }
    return drugs_df.fillna(value=values[method]().to_dict())

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


def norm_df(df, col=True):
    """
    Param df: df for normalization
    Param col: bool, if norm by column True, by row False
    return norm_df: same df after normalization
    """
    col_keys, index_keys = df.columns, df.index
    if not col:
        df = transpose_df(df)
        col_keys, index_keys = index_keys, col_keys
    scaler = StandardScaler()
    scaler.fit(df)
    df_norm_ndarray = scaler.transform(df)
    norm_df = pd.DataFrame(df_norm_ndarray,columns=col_keys, index=index_keys)
    if not col:
        norm_df = transpose_df(norm_df)
    return norm_df

def eliminate_extreme_vals(df):
    pass
