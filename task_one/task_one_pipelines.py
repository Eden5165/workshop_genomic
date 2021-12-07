import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

from utils import data_prep_utils, general_utils
import task_one_models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Piplines.
# All pipelines will return a tuple of three elements.
# First element will always be a fitted model to predict with.
# If a pipeline uses feature selection, second element returned will be the list of features NOT selected, else empy lise [].
# If a pipelines uses dimension reduction, third element returned will be the fitted dr model to transform with, else None.

def pipeline_1(beat_rnaseq_train, beat_drug_train, alpha=0.8):
    beat_rnaseq_train_norm = data_prep_utils.norm_df(beat_rnaseq_train)
    return task_one_models.get_lasso_reg_model(beat_rnaseq_train_norm, beat_drug_train, alpha), [], None

def get_drug_prediction_df(pipeline, genes_folds, drugs_folds, special_arg=None):
    """
    param pipeline: a pipeline function object.
    param genes_folds: list of genes df divided by folds, row are samples.
    param drugs_folds: same as genes_folds.
    return: drugs prediction df such that drugs are rows.
    """
    predictions = []
    for i in range(5):
        print("In fold number", i)
        train_x = pd.concat(genes_folds[:i] + genes_folds[i+1:])
        train_y = pd.concat(drugs_folds[:i] + drugs_folds[i+1:])
        test_x, test_y = genes_folds[i], drugs_folds[i]
        if special_arg is None:
            prediction = pipeline(train_x, train_y)[0].predict(test_x)
        else:
            prediction = pipeline(train_x, train_y, special_arg)[0].predict(test_x)
        predictions.append(general_utils.convert_predict_to_df(prediction, test_y.columns, test_y.index))
    drug_pred_df = pd.concat(predictions).transpose()        
    general_utils.export_drugs_prediction(drug_pred_df, "task_one_" + pipeline.__name__)
    return drug_pred_df

def test_pipeline_1_alphas(alphas, genes_folds, drugs_folds, beat_drug_k):
    """
    param alphas: a list of alpha values to send to the lasso regressor.
    """
    mses = []
    for alpha in alphas:
        print("Testing alpha:", alpha)
        drug_pred_df = get_drug_prediction_df(pipeline_1, genes_folds, drugs_folds, alpha)
        mses.append(general_utils.get_mse(beat_drug_k, drug_pred_df))
    return mses

def plot_test_pipeline_1_alphas(alphas, mses):
    plot_path = os.path.join(os.getcwd(), "plots", "test_pipeline_1_alphas")
    plt.plot(mses, alphas)
    plt.xlabel('MSE Scores')
    plt.ylabel('Alpha Values')
    plt.title('MSE Scores for Differens Alpha Values - Lasso Regressor')
    plt.savefig(plot_path)
    plt.close()


def get_drugs(beat_rnaseq, beat_drug):
    """
    param beat_rnaseq: beat_rnaseq such that rows are samples.
    param beat_drug: beat_drug df such that rows are samples ans no missing values
    return: df ready for pipelines.
    """
    return data_prep_utils.get_data_reorgenized(beat_rnaseq, data_prep_utils.ic50_log_trans(beat_drug))

def main():
    beat_rnaseq = data_prep_utils.gene_exp_log_trans(data_prep_utils.get_df("beat_rnaseq"))
    beat_rnaseq_t = beat_rnaseq.transpose()

    beat_drug_raw = data_prep_utils.get_df("beat_drug")
    beat_drug_k_t = get_drugs(beat_rnaseq_t, data_prep_utils.missing_vals_knn(beat_drug_raw).transpose())
    beat_drug_mean_t = get_drugs(beat_rnaseq_t, data_prep_utils.missing_vals_method(beat_drug_raw.transpose(), "mean"))

    beat_drug_k, beat_drug_mean = beat_drug_k_t.transpose(), beat_drug_mean_t.transpose()

    
    genes_folds, drugs_folds_k = general_utils.divide_to_folds(beat_rnaseq_t, beat_drug_k_t)
    _, drugs_folds_mean = general_utils.divide_to_folds(beat_rnaseq_t, beat_drug_mean_t)

    # pipeline_1_mse = general_utils.get_mse(beat_drug_k, get_drug_prediction_df(pipeline_1, genes_folds, drugs_folds_k))
    # print(pipeline_1_mse)

    alphas = list(np.arange(0.1, 1.1, 0.1))
    mses = test_pipeline_1_alphas(alphas, genes_folds, drugs_folds_k, beat_drug_k)
    print(mses)
    plot_test_pipeline_1_alphas(alphas, mses)


if __name__== "__main__" :
    main()