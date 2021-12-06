import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PRESERVE_COMP = 0.95

# Dimension reduction
def pca(genes_df):
    """
    
    """
    X = genes_df.values
    print(X.shape)

    # Standardize - do we need it? -> check row/column standart
    scaler =  StandardScaler() 
    scaler.fit(X)
    X_scaled= scaler.transform(X)

    pca = PCA(n_components=PRESERVE_COMP, random_state = 2020)

    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    print(X_pca.shape)
    return pca, X_pca



def aml_genes_selection(df):
    # Select only genes associated with AML as features
    pass


def k_means_reduction(df):
    """
    TODO: with huristic k
    """
    pass