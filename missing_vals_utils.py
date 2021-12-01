import pandas as pd
from sklearn.impute import KNNImputer


# Missing values
def missing_vals_reg(drugs_df):
    """

    """
    pass


def missing_vals_knn(drugs_df, k=5):
    """
    param drugs_df: drugs df as returned from data_prep_utils.get_df. That means, rows are drugs.
    param k: number of neighbors with which to run the KNN algorithm.
    return: drugs_full with missing values filled using the KNN algorithm.
    """
    drugs_full = drugs_df.copy()
    imputer = KNNImputer(n_neighbors=k)
    drugs_full.iloc[:, :] = imputer.fit_transform(drugs_full)
    return drugs_full
    


def missing_vals_method(drugs_df, method):
    """
    param drugs_df: drugs df transposed. That means, rows are samples.
    param method: In: ["max", "mean"] A mathematical method to fill the missing values be.
    return: NaN values filled by requested method, per drug.
    """
    values = {
        "max": drugs_df.max,
        "mean": drugs_df.mean
    }
    return drugs_df.fillna(value=values[method]().to_dict())