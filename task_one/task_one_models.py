from scipy.sparse.construct import random
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


def get_linear_reg_model(train_x, train_y):
    """
    param tarin_x: genes train df such that rows are samples
    param train_y: drugs train df such that rows are samples, values are log transformed and no
    missing values.
    return: fitted model.
    """
    return LinearRegression().fit(train_x, train_y)


def get_decision_tree_model(train_x, train_y):
    """
    param tarin_x: genes train df such that rows are samples
    param train_y: drugs train df such that rows are samples, values are log transformed and no
    missing values.
    return: fitted model.
    """
    return DecisionTreeRegressor(random_state=0, criterion="friedman_mse", splitter="random").fit(train_x, train_y)



def get_gradient_boosting_tree_model(train_x, train_y):
    """
    param tarin_x: genes train df such that rows are samples
    param train_y: drugs train df such that rows are samples, values are log transformed and no
    missing values.
    return: fitted model.
    """
    # ToDo: Think about which parameternt of GradientBoostingRegressor we want to
    # change from default for testing improvement. For example, n_estimators number.
    return MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(train_x, train_y)


def get_reg_chain_model(train_x, train_y, reg_model='linear', order='random'):
    """
    param tarin_x: genes train df such that rows are samples
    param train_y: drugs train df such that rows are samples, values are log transformed and no
    missing values.
    param reg_model: The model with which the prediction should be made. In: ["linear", "boosting_tree"]
    order: The order in which the drugs should be added to the chain.
    return: fitted model.
    """
    reg_models = {
        "linear": LinearRegression(),
        "boosting_tree": MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    }
    return RegressorChain(base_estimator=reg_models[reg_model], order=order).fit(train_x, train_y)


def get_lasso_reg_model(train_x, train_y, alpha, normalize):
    """
    param tarin_x: genes train df such that rows are samples
    param train_y: drugs train df such that rows are samples, values are log transformed and no
    missing values.
    param alpha: the alpha parameter for the Lasso regressor.
    param normalize: Should be False if the data is alreasy normalized or standartized.
    return: fitted model.
    """
    return Lasso(alpha=alpha, fit_intercept=True, normalize=normalize, copy_X=True, selection="random").fit(train_x, train_y)

def plot_kmeans_elbow(drugs_df, max_k=10):
    """
    param drugs_df: drugs df such that drugs are rows, no missing values and values are log transformed.
    param max_k: int. max values of k to test.
    return: saving the elbow plot under kmeans_elbow
    """
    plot_path = os.path.join(os.path.dirname(os.getcwd()), "plots", ("kmeans_elbow" + str(max_k)))
    distortions = []
    for k in range(1, max_k):
        distortions.append(KMeans(n_clusters=k).fit(drugs_df).inertia_)
    plt.plot(range(1, max_k), distortions)
    plt.xlabel('k')
    plt.ylabel('Distotion')
    plt.title('Elbow Method for K-means algorithm on beat_drug')
    plt.savefig(plot_path)


    