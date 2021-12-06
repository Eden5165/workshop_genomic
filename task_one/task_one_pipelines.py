import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

from utils import data_prep_utils, norm_utils, missing_vals_utils
import task_one_models

# Piplines.
# All pipelines will return a tuple of three elements.
# First element will always be a fitted model to predict with.
# If a pipeline uses feature selection, second element returned will be the list of features selected, else None.
# If a pipelines uses dimension reduction, third element returned will be the fitted dr model to transform with, else None.

def pipeline_1(beat_rnaseq_train, beat_drug_train):
    return task_one_models.get_lasso_reg_model(beat_rnaseq_train, beat_drug_train, 0.8), None, None

def get_drug_prediction_df(pipeline, genes_folds, drugs_folds):
    predictions = []
    for i in range(5):
        # define 
        pass


def get_drugs(beat_rnaseq, beat_drug):
    """
    param beat_rnaseq: beat_rnaseq such that rows are samples.
    param beat_drug: beat_drug df such that rows are drugs ans no missing values
    return: df ready for pipelines.
    """
    return data_prep_utils.get_data_reorgenized(beat_rnaseq, norm_utils.ic50_log_trans(beat_drug.transpose()))

def main():
    beat_rnaseq = norm_utils.gene_exp_log_trans(data_prep_utils.get_df("beat_rnaseq")).transpose()

    beat_drug_raw = data_prep_utils.get_df("beat_drug")
    beat_drug_k = get_drugs(missing_vals_utils.missing_vals_knn(beat_drug_raw))
    beat_drug_mean = get_drugs(missing_vals_utils.missing_vals_method(beat_drug_raw.transpose(), "mean"))