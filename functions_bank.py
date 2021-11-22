# Data preperation

def transpose_df(df):
    """
    param: 
    """
    pass

def fit_drugs_to_genes(genes_df, drugs_df):
    pass


"""
All functions starting here will expect to recive genes_df and drugs df
such that both df are in the right orientation (rows are samples) and the
and sample i will appear in row i in both dfs.
That means, genes_df and drugs_df are the results of transpose_df and fit_drugs_to_genes
"""


# Normalization
def sum_gene_exp_to_one(df):
    pass


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









