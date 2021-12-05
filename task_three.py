import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier


def build_cor_mat(drugs_df, muts_df):
    """
    param drugs_df: drugs df such that rows are samples and no missing values.
    param muts_df: mutations df such that rows are samples. number of samples
    equals to number of samples in drugs df.
    return: correlation matrix between drugs and mutations suach that drugs are columns.

    Options for inputs:
    1. beat_drug, using all samples or only 174 + beat_mut prediced.
    2. tcga_drug predicted + tcga_mut.
    3. beat_drug + tcga_drug predicted + beat_mut prediced + tcga_mut.
    """
    drugs_ranked = drugs_df.rank()
    return pd.concat([drugs_ranked, muts_df], axis=1).corr()[drugs_ranked.columns].loc[muts_df.columns]


def get_linear_svc_model(train_x, train_y):
    """
    param x_train: genes df such that rows are samples and dimension is reduced to <= num of samples.
    param y_train: mutations df such that rows are samples.
    return: fitted model.
    """
    return MultiOutputClassifier(LinearSVC()).fit(train_x, train_y)

