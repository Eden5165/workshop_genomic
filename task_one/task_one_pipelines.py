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
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.multioutput import RegressorChain

# Piplines.
# All pipelines will return a tuple of three elements.
# First element will always be a fitted model to predict with.
# If a pipeline uses feature selection, second element returned will be the list of features NOT selected, else empy lise [].
# If a pipelines uses dimension reduction, third element returned will be the fitted dr model to transform with, else None.


PIPLINES = {
    "test_lasso": [
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
    "lasso_best_alpha": Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(random_state=10, max_iter=10000, alpha=0.9))]),
    "test_linear_reg_fs": [
        Pipeline([('lr', LinearRegression())]),
        Pipeline([('scaler', StandardScaler()), ('var_th', VarianceThreshold(threshold=(0))), ('lr', LinearRegression())]),
        Pipeline([('scaler', StandardScaler()), ('var_th', VarianceThreshold(threshold=(0.1))), ('lr', LinearRegression())]),
        Pipeline([('scaler', StandardScaler()), ('var_th', VarianceThreshold(threshold=(0.2))), ('lr', LinearRegression())]),
        Pipeline([('scaler', StandardScaler()), ('var_th', VarianceThreshold(threshold=(0.3))), ('lr', LinearRegression())]),
        Pipeline([('scaler', StandardScaler()), ('var_th', VarianceThreshold(threshold=(0.4))), ('lr', LinearRegression())]),
        Pipeline([('scaler', StandardScaler()), ('var_th', VarianceThreshold(threshold=(0.5))), ('lr', LinearRegression())])
    ],
    "test_regressor_chain": [
        Pipeline([('scaler', StandardScaler()), ('rc_lasso', RegressorChain(base_estimator=Lasso(random_state=10, max_iter=10000, alpha=0.9), order="random"))]),
        Pipeline([('scaler', StandardScaler()), ('rc_lasso', RegressorChain(base_estimator=Lasso(random_state=10, max_iter=10000, alpha=0.9), order=None))])
        ],
    "lr_rc": Pipeline([('scaler', StandardScaler()), ('var_th', VarianceThreshold(threshold=(0.5))), ('rc_lr', RegressorChain(base_estimator=LinearRegression(), order="random"))])

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


def test_pipelines(pipelines, genes_folds, drugs_folds, beat_drug, missing_sys):
    mses = []
    print("Testing", pipelines)
    for idx, pipline in enumerate(PIPLINES[pipelines]):
        print("Testing pipeline:", str(idx + 1))
        drug_pred_df = get_drug_prediction_df(pipline, genes_folds, drugs_folds, f"{missing_sys}_{pipelines}_{str(idx + 1)}")
        mses.append(general_utils.get_mse(beat_drug, drug_pred_df))
        print("mse:", mses[idx])
    return mses

def plot_test_pipelines(param_name, param_vals, mses, plot_name):
    plot_path = os.path.join(os.getcwd(), "plots", plot_name)
    plt.plot(mses, param_vals)
    plt.xlabel('MSE Scores')
    plt.ylabel(f'{param_name} Values')
    plt.title(f'MSE Scores for Differens {param_name} Values - Lasso Regressor')
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
    # alphas_mses = test_pipelines("test_lasso", genes_folds, drugs_folds_k, beat_drug_k, "kmenas")
    # print(alphas_mses)
    # plot_test_pipelines("Alphas", alphas, alphas_mses, "test_lasso_alphas")
    
    # alphas = list(np.arange(0.1, 1, 0.1))
    # mses = test_lasso_alphas(genes_folds_filt, drugs_folds_mean, beat_drug_mean, "mean")
    # print(mses)
    # plot_test_lasso_pipeline_alphas(alphas, mses)

    # thresholds = [-1] + list(np.arange(0.0, 0.6, 0.1))
    # thresholds_mses = test_pipelines("test_linear_reg_fs", genes_folds, drugs_folds_k, beat_drug_k, "kmenas")
    # print(thresholds_mses)
    # plot_test_pipelines("Feature Selection Variance Threshold", thresholds, thresholds_mses, "test_lr_fs_th")

    # orders = ["random", "ordered"]
    # orders_mse = test_pipelines("test_regressor_chain", genes_folds, drugs_folds_k, beat_drug_k, "kmenas")
    # print(orders_mse)
    # plot_test_pipelines("Drugs Order", orders, orders_mse, "test_regressor_chain_orders")

    lr_rc_pred = get_drug_prediction_df(PIPLINES['lr_rc'], genes_folds, drugs_folds_k, 0)
    print(general_utils.get_mse(drugs_folds_k, lr_rc_pred))






if __name__== "__main__" :
    main()