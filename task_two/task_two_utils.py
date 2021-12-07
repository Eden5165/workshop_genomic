import pandas as pd
import numpy as np
from utils import general_utils, data_prep_utils

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
