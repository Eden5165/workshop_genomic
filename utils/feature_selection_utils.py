from sklearn.feature_selection import VarianceThreshold, RFE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/

VT_THRESHOLD = 0.5 # cahnge after normalization



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
    https://machinelearningmastery.com/rfe-feature-selection-in-python/

    works by recursively removing attributes and building a model from the remain attributes. 
    It uses accuracy metrix to rank the feature according to their importance.
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



def find_corr_by_drug(genes_df, drug_coulmn, drug_name):
    """
    find drug corrletion betwwn each drug to all the genes
    param gene_df: gene table df (only gene with small corrleation to all the drugs that tested until now)
    param drug_coulmn: get the drug relevant data
    param draw: bool
    return gene_df_filtered: the gene_df after filtering 
    return low_corr_genes: list of the low corrlation genes
    """
    genes_list = genes_df.keys()
    low_corr_genes = []
    full_gene_n_drug_table = pd.merge(drug_coulmn,  genes_df,left_index=True, right_index=True)
    for gene in genes_list:
        corr_data = full_gene_n_drug_table[[drug_name, gene]].corr()
        corr = abs(corr_data.at[drug_name,gene])
        if corr < 0.3 :
            low_corr_genes.append(gene)

    gene_df_filtered = full_gene_n_drug_table.loc[:, low_corr_genes]

    return gene_df_filtered, low_corr_genes

def select_gene_by_corrlation(genes_df, drug_df):
    """
    filtered out all the genes that don't corllate with any of the drugs

    note: run only for genes with low corrlation for all the other drugs(that tested before the tested drug) 

    :param gene_df, drug_df (both df after the preperation)
    :return: 
        genes_to_filter: list of all the genes the function filtered out
        filtered_gene_df: gene df after filter
    """
    my_genes_df = genes_df
    genes_to_filter = genes_df.keys()[1:]
    counter = 1
    for drug in drug_df.columns: 
        
        my_genes_df = my_genes_df.loc[:, (my_genes_df.columns.isin(genes_to_filter))]

        my_genes_df, genes_to_filter = find_corr_by_drug(my_genes_df, drug_df[drug], drug)

        print(counter, " - ",drug, " -> ", len(genes_to_filter))
        counter += 1
    
    
    new_genes_df = genes_df.loc[:, ~(genes_df.columns.isin(genes_to_filter))]
    return new_genes_df, genes_to_filter

def select_high_var_genes(genes_df): 
    """
    # TODO: DO AFTER NORMALIZATION -> and than change the threshold
    filtered out all the genes with low variance
    param genes_df: genes dataframe *after normalizatiomn*
    return filtered_genes_df
    """
    print(genes_df.shape) # TODO: delete
    vt = VarianceThreshold(VT_THRESHOLD)
    _ = vt.fit(genes_df)
    mask = vt.get_support()
    filtered_genes_df = genes_df.loc[:, mask]
    print(filtered_genes_df.shape) # TODO: delete
    return filtered_genes_df
