import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

from utils import data_prep_utils, general_utils
import task_one_models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor

# Piplines.
# All pipelines will return a tuple of three elements.
# First element will always be a fitted model to predict with.
# If a pipeline uses feature selection, second element returned will be the list of features NOT selected, else empy lise [].
# If a pipelines uses dimension reduction, third element returned will be the fitted dr model to transform with, else None.


PIPLINES = {
    "lasso": [
        Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(random_state=10, max_iter=10000, alpha=0.1))]),
        Pipeline([('scaler', StandardScaler()),('lasso', Lasso(random_state=10, max_iter=10000, alpha=0.2))]),
        Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(random_state=10, max_iter=10000, alpha=0.3))]),
        Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(random_state=10, max_iter=10000, alpha=0.4))]),
        Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(random_state=10, max_iter=10000, alpha=0.5))]),
        Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(random_state=10, max_iter=10000, alpha=0.6))]),
        Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(random_state=10, max_iter=10000, alpha=0.7))]),
        Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(random_state=10, max_iter=10000, alpha=0.8))]),
        Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(random_state=10, max_iter=10000, alpha=0.9))]),
    ],
    "lasso_best_alpha": Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(random_state=10, max_iter=10000, alpha=0.8))])
}


def get_drug_prediction_df(pipeline, genes_folds, drugs_folds, pipeline_idx):
    """
    param pipeline: a pipeline function object.
    param genes_folds: list of genes df divided by folds, row are samples.
    param drugs_folds: same as genes_folds.
    return: drugs prediction df such that samples are rows.
    """
    predictions = []
    for i in range(5):
        print("In fold number", i)
        train_x = pd.concat(genes_folds[:i] + genes_folds[i+1:])
        train_y = pd.concat(drugs_folds[:i] + drugs_folds[i+1:])
        test_x, test_y = genes_folds[i], drugs_folds[i]
        pipeline.fit(train_x, train_y)
        prediction = pipeline.predict(test_x)
        predictions.append(general_utils.convert_predict_to_df(prediction, test_y.columns, test_y.index))
    drug_pred_df = pd.concat(predictions)  
    general_utils.export_drugs_prediction(drug_pred_df, "task_one_pipeline_" + str(pipeline_idx))
    return drug_pred_df

def test_lasso_alphas(genes_folds, drugs_folds, beat_drug, missing_sys):
    """
    """
    mses = []
    for idx, lasso_pipline in enumerate(PIPLINES["lasso"]):
        print("Testing alpha:", "0." + str(idx + 1))
        drug_pred_df = get_drug_prediction_df(lasso_pipline, genes_folds, drugs_folds, missing_sys + "_lasso_" + str(idx))
        # print("+++++++++\n", drug_pred_df.head(), "\n\n", beat_drug_k.head())
        mses.append(general_utils.get_mse(beat_drug, drug_pred_df))
        print("mse:", mses[idx])
    return mses

def plot_test_lasso_pipeline_alphas(alphas, mses):
    plot_path = os.path.join(os.getcwd(), "plots", "test_lasso_alphas")
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

    beat_drug = data_prep_utils.get_df("beat_drug")
    beat_drug_k = get_drugs(beat_rnaseq, data_prep_utils.missing_vals_knn(beat_drug))
    beat_drug_mean = get_drugs(beat_rnaseq, data_prep_utils.missing_vals_method(beat_drug, "mean"))
    
    genes_folds, drugs_folds_k = general_utils.divide_to_folds([beat_rnaseq, beat_drug_k])
    drugs_folds_mean = general_utils.divide_to_folds([beat_drug_mean])[0]

    tcga_rna = data_prep_utils.gene_exp_log_trans(data_prep_utils.get_df("tcga_rna"))
    beat_rnaseq_filt, tcga_rna_filt = data_prep_utils.filter_beat_and_tcga_by_shared_genes(beat_rnaseq, tcga_rna)

    genes_folds_filt = general_utils.divide_to_folds([beat_rnaseq_filt])[0]
    
    # alphas = list(np.arange(0.1, 1, 0.1))
    # mses = test_lasso_alphas(genes_folds, drugs_folds_k, beat_drug_k)
    # print(mses)
    # plot_test_lasso_pipeline_alphas(alphas, mses)

    # print("\n", beat_drug_mean.head())
    
    alphas = list(np.arange(0.1, 1, 0.1))
    mses = test_lasso_alphas(genes_folds_filt, drugs_folds_mean, beat_drug_mean, "mean")
    print(mses)
    plot_test_lasso_pipeline_alphas(alphas, mses)


if __name__== "__main__" :
    main()