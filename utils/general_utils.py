import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from utils import data_prep_utils

# Testing
def divide_to_folds(dfs, nb =False):
    """
    param dfs: a list of dfs to divide.
    return: a list of splited dfs.
    """
    parent_folder = os.getcwd()
    if nb:
        parent_folder = os.path.dirname(parent_folder)
    division_fp = os.path.join(parent_folder, "medical_genomics_2021_data", "folds.txt")
    division_df = pd.read_csv(division_fp, sep='\t', lineterminator='\n', header=None, names=["sampleID", "fold"])
    dfs_folds=[]
    for df in dfs:
        df_folds = []
        for i in range(1, 6):
            samples = division_df[division_df["fold"]==i]["sampleID"]
            df_folds.append(df[df.index.isin(samples)])
        dfs_folds.append(df_folds)
    return dfs_folds


def get_mse(true_drugs, pred_drugs):
    """
    param true_drugs: data frame of original drugs data such taht rows are samples, values are
    log transformed, no NaN values.
    param pred_drugs: data frame of drugs prediction such that rows are samples, values are
    log transformed.
    """
    return mean_squared_error(true_drugs, pred_drugs)


def export_drugs_prediction(drug_pred_df, file_name):
    """
    param pred_df: drug prediction df such that rows are samples.
    param file_name: the file name under which the df will be saved.
    """
    output_fp = os.path.join(os.getcwd(), "prediction_results", file_name)
    drug_pred_df.transpose().to_csv(output_fp, sep="\t", line_terminator='\n', na_rep="NA", index_label=False)

#utils
def merge_by_index(df1, df2):
    """
    get 2 data frames and return merge table by index
    """
    return  pd.merge(df1, df2, left_index=True, right_index=True)

def convert_predict_to_df(prediction, col_names, row_names):
    """
    param prediction: result from model prediction
    param col_names: list of the predicted values names, for examples drugs names list
    param row_names: list of samples indexes/names, for examples sample num
    """
    df = pd.DataFrame(prediction,columns=col_names, index=row_names)
    print("convert_predict_to_df-> null_count: ", df.isna().sum().sum())
    return df

def classifier_cross_validation(classifier, X, y):
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")
    plt.show()