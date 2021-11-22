from sklearn.feature_selection import VarianceThreshold, RFE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def select_high_var_genes(gene_df, drug_df):
    """
    https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b

    1. for each drug find uncorllated gene, filter out only genes that not corlated for all the drugs
    2. compare the result with unfiltered result(maybe 2 drug corlation)
    """
    pass


def select_gene_by_corrlation(gene_df, drug_df):
    """
    filtered out all the genes that don't corllate with any of the drugs

    note: run only for genes with low corrlation for all the other drugs(that tested before the tested drug) 

    :param gene_df, drug_df (both df after the preperation)
    :return: 
        genes_to_filter: list of all the genes the function filtered out
        filtered_gene_df: gene df after filter
    """
    genes_to_filter = gene_df.columns[1:]
    first = True
    for drug in drug_df.columns[1:]: #FIXME: validate with yael
        if first:
            my_gene_df = gene_df
            first = False
        else:
            my_gene_df = gene_df.loc[:, (gene_df.columns.isin(genes_to_filter))]

        gene_to_drug_cor, non_relevant_feture = find_corr_by_drug(my_gene_df, drug_df[drug])
        genes_to_filter = non_relevant_feture
    
    filtered_gene_df = gene_df.loc[:, ~(gene_df.columns.isin(genes_to_filter))]

    return genes_to_filter, filtered_gene_df
        
def find_corr_by_drug(gene_df, drug_coulmn, draw = False):
    """
    using Pearson Correlation
    option: create corrlation function -> option linear regression and other corrlation methods
    """
    gene_df['drug_value'] = drug_coulmn
    cor = gene_df.corr() 
    gene_to_drug_cor = abs(cor['drug_value'])
    non_relevant_feture = gene_to_drug_cor[ gene_to_drug_cor < 0.5 ]
    
    if draw:
        plt.figure(figsize=(12,10))
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()
    
    return gene_to_drug_cor, non_relevant_feture

def feature_reduction_by_genes_with_genes_corr(gene_df):
    """
    choose only genes with input, if there are 2 genes that bring the same input choose only one of them.
    this function used corrlation, but we can use different algorithm to group the genes.
    examples: clustering -> using as union-find DS

    :param gene_df, drug_df (both df after the preperation)
    :return: 
    genes_to_filter: list of all the genes the function filtered out
    filtered_gene_df: gene df after filter
    """
    pass

def backward_elimination(model):
    """
    we feed all the possible features to the model at first. 
    We check the performance of the model and then iteratively remove the worst performing features
    one by one till the overall performance of the model comes in acceptable range.
    link: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
    """
    pass

def rfe_elimination(model, gene_df, drug_y):
    """
    works by recursively removing attributes and building a model on those attributes that remain. 
    It uses accuracy metric to rank the feature according to their importance.
    we can choose only the feature that rank the highest
    """
    rfe = RFE(model, 100)
    #Transforming data using RFE
    X_rfe = rfe.fit_transform(gene_df,drug_y)  
    #Fitting the data to model
    model.fit(X_rfe,drug_y)
    genes_ranking_by_place = rfe.ranking_

    genes_list = gene_df.columns[1:]
    genes_to_filter = [genes_list[i]  for i in range(len(genes_list)) if genes_ranking_by_place[i] ]
    filtered_gene_df = gene_df.loc[:, ~(gene_df.columns.isin(genes_to_filter))]

    return genes_to_filter, filtered_gene_df