# Data preperation
def filter_beat_by_tcga(beat_rna, tcga_rna):
    """
    param beat_rna tcga_rna: original data frames such that rows are genes
    return beat_rna_filtered: beat_rna df only with genes that appear in tcga_rna
    """
    pass


def get_df(table_name, sep='\t', lineterminator='\n'):
    """
    param table_name: in ["beat_drug", "beat_rnaseq", "drug_mut_cor", "drug_mut_cor_lables", "tcga_mut", "tcga_ma"]
    param sep: Columns seperator characters. Optional, default='\t'
    param lineterminator: Rows seperator characters. Optional, default='\n'
    return: df for the requested table
    """
    pass


def transpose_df(df):
    """
    param df: df in wich columns are samples
    return: trsanposed df such that first column named SampleID, rows are samples
    """
    pass

def get_data_reorgenized(genes_df, other_data_df):
    """
    param genes_df: genes df such that rows are samples
    param other_data_df: drugs or mutations df such that rows are sample
    return: other_data_df reoregenized such that row i represents the sample i that is
    represented by row i in genes_df
    """
    pass


"""
All functions starting here will expect to recive genes_df and drugs df
such that both df are in the right orientation (rows are samples) and the
and sample i will appear in row i in both dfs.
That means, genes_df and drugs_df are the results of transpose_df and fit_drugs_to_genes
"""


# Normalization
def sum_gene_exp_to_num(genes_df, num):
    """
    param genes_df: genes df such that rows are samples and first colum is the sample ID
    param num: 1 or 1000000.
    return: if num=1, returns genes_df_sum such that each cell is divided by the sum of it's column, that
    means, each column (gene) is summed to one. if num = 1 mill, same as 1 only multiplied by 1 mill, 
    That menas, each column (gene) is summed to one mill.
    """
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


# Missing values
def missing_vals_reg(df):
    pass


def missing_vals_method(df, method):
    pass


# Feature Selection and Dimension Reduction
def select_high_var_genes(df):
    pass


def regularization(df):
    pass


# Dimension reduction
def pca(df):
    pass

def aml_genes_selection(df):
    # Select only genes associated with AML as features
    pass


def k_means_reduction(df):
    pass


def biological_clusters(df):
    # Use known clustering data that devides gene by function in some way we find fit
    pass


# Regression per drug
def reg_per_drug(df):
    """

    :param df:
    :return: Dictionary as such: {drug_name: {reg_model1: obj, reg_model2: obj...}}
    """
    pass


def linear_reg_per_drug(df):
    pass


def decision_tree_reg_per_drug(df):
    pass


def gradient_boosting_per_drug(df):
    pass


# Chained Multi-output Regression
def chained_multi_reg(df):
    pass


# Drugs Clusters
def get_drug_clusters(df):
    pass


# Task 2


def predict_unlabeled(model1, unlabeled_df):
    pass


def predict_unlabeled_coreg(model1, unlabeled_df):
    pass


# Testing

def get_mse():
    pass









