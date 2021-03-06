import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from utils import data_prep_utils, general_utils, feature_selection_utils


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


def get_linear_svc_class_model(train_x, train_y):
    """
    param x_train: genes df such that rows are samples and dimension is reduced to <= num of samples.
    param y_train: mutations df such that rows are samples.
    return: fitted model.
    """
    return MultiOutputClassifier(LinearSVC()).fit(train_x, train_y)


def get_gradient_boosting_class_model(train_x, train_y):
    """
    param x_train: genes df such that rows are samples.
    param y_train: mutations df such that rows are samples.
    return: fitted model.
    """
    return MultiOutputClassifier(GradientBoostingClassifier()).fit(train_x, train_y)


def get_knn_model(train_x, train_y, k=5):
    """
    model demands: 
     * features selection + dimentions reductions
     * normlizing
     * k value (neighbors num)
     * p could be 1,2, infinity , each one represent a norm
    """
    return KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2).fit(train_x, train_y)


def load_df(nb = False):
    tcga_genes = data_prep_utils.gene_exp_log_trans(data_prep_utils.get_df("tcga_rna",nb=nb))
    beat_rnaseq = data_prep_utils.gene_exp_log_trans(data_prep_utils.get_df("beat_rnaseq",nb=nb))
    tcga_muts = data_prep_utils.get_df("tcga_mut", nb=nb)
    beat_drug = data_prep_utils.ic50_log_trans(data_prep_utils.get_data_reorgenized(beat_rnaseq, data_prep_utils.missing_vals_knn(data_prep_utils.get_df("beat_drug",nb=nb))))
    beat_rnaseq_shared, tcga_rna_shared = data_prep_utils.filter_beat_and_tcga_by_shared_genes(beat_rnaseq, tcga_genes)
    return tcga_rna_shared, tcga_muts, beat_rnaseq_shared, beat_drug


def filter_samples(beat_rnaseq, drugs):
    samples_path = os.path.join(os.getcwd(), "medical_genomics_2021_data", "drug_mut_cor_labels")
    with open(samples_path, 'r') as fd:
        lines = fd.read().split('\n')
    return beat_rnaseq.loc[beat_rnaseq.index.isin(lines), :], drugs.loc[beat_rnaseq.index.isin(lines), :]


def get_mut_predict(tcga_genes, tcga_muts, beat_rnaseq_shared):
    scaler = StandardScaler()
    scaler.fit(tcga_genes)
    tcga_genes_norm = scaler.transform(tcga_genes)
    beat_rnaseq_norm = scaler.transform(beat_rnaseq_shared)

    vt = VarianceThreshold(0.05)
    vt.fit(tcga_genes_norm)
    filtered_genes_df = vt.transform(tcga_genes_norm)
    filtered_beat_rnaseq_df = vt.transform(beat_rnaseq_norm)

    gbc = get_gradient_boosting_class_model(filtered_genes_df, tcga_muts)
    mut_predict = gbc.predict(filtered_beat_rnaseq_df)


    return mut_predict


def mut_predict_by_one(tcga_genes, tcga_muts, beat_rnaseq_shared, beat_drug):
    
    scaler = StandardScaler()
    scaler.fit(tcga_genes)
    tcga_genes_norm = scaler.transform(tcga_genes)
    beat_rnaseq_norm = scaler.transform(beat_rnaseq_shared)

    vt = VarianceThreshold(0.05)
    vt.fit(tcga_genes_norm)
    filtered_genes_df = vt.transform(tcga_genes_norm)
    filtered_beat_rnaseq_df = vt.transform(beat_rnaseq_norm)

    predict_mut_list = {}

    for mut_num, mut in enumerate(tcga_muts.colunms):
        print(mut)
        model = GradientBoostingClassifier()
        mut_df = tcga_muts.iloc[:, mut_num:mut_num+1]
        model.train(filtered_genes_df,mut_df)
        mut_df_predict = model.predict(filtered_beat_rnaseq_df)
        predict_mut_list[mut] = mut_df_predict
    
    return predict_mut_list
    


if __name__== "__main__" :
    tcga_genes, tcga_muts, beat_rnaseq_shared, beat_drug = load_df()
    beat_rnaseq_shared_filterd, drugs_filtered = filter_samples(beat_rnaseq_shared, beat_drug)
    mut_predict= get_mut_predict(tcga_genes, tcga_muts, beat_rnaseq_shared_filterd)

    print(mut_predict)
    output_fp = os.path.join(os.getcwd(), "mut_predict", "0000")
    mut_predict.transpose().to_csv(output_fp, sep="\t", line_terminator='\n', na_rep="NA", index_label=False)

    mut_drug_df = build_cor_mat(drugs_filtered, general_utils.convert_predict_to_df(mut_predict, tcga_muts.columns, beat_rnaseq_shared_filterd.index))
    print(mut_drug_df)

    output_fp = os.path.join(os.getcwd(), "mut_drug_results", "1")
    mut_drug_df.transpose().to_csv(output_fp, sep="\t", line_terminator='\n', na_rep="NA", index_label=False)







    