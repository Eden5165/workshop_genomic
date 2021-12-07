import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

from utils import data_prep_utils, general_utils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from task_one.task_one_pipelines import PIPLINES

def coreg_algo(beat_rna_df, tcga_rna_df, drug_df):
    """
    """
    pass

def predict_tcga(beat_rna_df, tcga_rna_df, drug_df, pipeline_task_one):
    """
    param beat_rna_df: beat rna df after data prep - tranpose, possibly norm
    param tcga_rna_df: tcga rna df after data prep - tranpose, possibly norm
    param drug_df: drug df after data prep - fill missing + tranpose
    pipeline_task_one: best pipeline from task 1
    return tcga_drug_predict_df
    """
    model_predictor, not_selcted_features, dimention_red_model = pipeline_task_one(beat_rna_df, drug_df)
    tcga_after_filterd_features = tcga_rna_df.loc[:, ~(tcga_rna_df.columns.isin(not_selcted_features))]
    if dimention_red_model is not None:
        tcga_rna_ready_predict =  dimention_red_model.transform(tcga_after_filterd_features)
    else:
        tcga_rna_ready_predict = tcga_after_filterd_features
    tcga_drug_predict = model_predictor.predict(tcga_rna_ready_predict)
    tcga_drug_predict_df = general_utils.convert_predict_to_df(tcga_drug_predict, drug_df.columns, tcga_rna_df.index)

    return tcga_drug_predict_df

def predict_tcga_by_pipline(beat_rna_df, tcga_rna_df, drug_df, pipeline_task_one):
    """
    param beat_rna_df: beat rna df after data prep(full gene list)
    param tcga_rna_df: tcga rna df after data prep (full gene list)
    param drug_df: drug df after data prep - fill missing
    pipeline_task_one: best pipeline from task 1
    return tcga_drug_predict_df
    """
    beat_rna_shared, tcga_rna_shared = data_prep_utils.filter_beat_and_tcga_by_shared_genes(beat_rna_df, tcga_rna_df)
    pipeline_task_one.fit(beat_rna_shared, drug_df)
    tcga_drug_predict = pipeline_task_one.predict(tcga_rna_shared)

    return general_utils.convert_predict_to_df(tcga_drug_predict, drug_df.columns, tcga_rna_df.index)

def get_drug_prediction_df_task_2(pipeline, genes_folds, drugs_folds, pipeline_idx, tcga_rna, tcga_predic_drug):
    """
    param pipeline: a pipeline function object.
    param genes_folds: list of genes df divided by folds, row are samples.
    param drugs_folds: same as genes_folds.
    return: drugs prediction df such that samples are rows.
    """
    predictions = []
    for i in range(5):
        print("In fold number", i)

        train_x_folds = pd.concat(genes_folds[:i] + genes_folds[i+1:])
        train_x = pd.concat([train_x_folds, tcga_rna])

        train_y_folds = pd.concat(drugs_folds[:i] + drugs_folds[i+1:])
        train_y = pd.concat([train_y_folds, tcga_predic_drug])

        test_x, test_y = genes_folds[i], drugs_folds[i]

        pipeline.fit(train_x, train_y)
        prediction = pipeline.predict(test_x)
        predictions.append(general_utils.convert_predict_to_df(prediction, test_y.columns, test_y.index))

    drug_pred_df = pd.concat(predictions)  
    general_utils.export_drugs_prediction(drug_pred_df, "task_two_pipeline_" + str(pipeline_idx))
    return drug_pred_df

def main():
    print("load data\n")
    beat_rnaseq = data_prep_utils.gene_exp_log_trans(data_prep_utils.get_df("beat_rnaseq"))
    tcga_rna =  data_prep_utils.gene_exp_log_trans(data_prep_utils.get_df("tcga_rna"))
    beat_drug_df = data_prep_utils.ic50_log_trans(data_prep_utils.get_data_reorgenized(beat_rnaseq, data_prep_utils.missing_vals_knn(data_prep_utils.get_df("beat_drug"))))
    beat_rnaseq_shared, tcga_rna_shared = data_prep_utils.filter_beat_and_tcga_by_shared_genes(beat_rnaseq, tcga_rna)
    
    print("predict tcga")
    pipeline = PIPLINES["lasso_best_alpha"]
    tcga_predic_drug = predict_tcga_by_pipline(beat_rnaseq_shared, tcga_rna_shared, beat_drug_df, pipeline) 
    
    print("predict drugs")
    genes_folds, drugs_folds = general_utils.divide_to_folds([beat_rnaseq_shared, beat_drug_df])  

    pred = get_drug_prediction_df_task_2(pipeline, genes_folds, drugs_folds, 1, tcga_rna_shared,tcga_predic_drug)

    print(general_utils.get_mse(beat_drug_df, pred))
    

    return pred

if __name__== "__main__" :
    main()