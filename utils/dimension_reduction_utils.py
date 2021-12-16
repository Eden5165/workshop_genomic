import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils import data_prep_utils


PRESERVE_COMP = 0.95

# Dimension reduction
def pca_full_path(feature_df):
    """
    param feature_df: genes feature df such that rows are samples and normelized
    return: pca model(berfore fiting) and the new features list
    """
    feature_df_norm = data_prep_utils.norm_df(feature_df)

    pca = PCA(n_components=PRESERVE_COMP, random_state = 2020)
    pca.fit(feature_df_norm)
    new_feature_after_pca = pca.transform(feature_df_norm)
    return pca, new_feature_after_pca


def get_pca_model(feature_df):
    return PCA(n_components=PRESERVE_COMP, random_state = 2020)


def aml_genes_selection(df):
    # Select only genes associated with AML as features
    pass


def k_means_reduction(df):
    """
    TODO: with huristic k
    """
    pass

