from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def get_linear_reg_model(train_x, train_y):
    """
    param tarin_x: genes train df such that rows are samples
    param train_y: drugs train df such that rows are samples, values are log transformed and no
    missing values.
    """
    return LinearRegression().fit(train_x, train_y)


def get_gradient_boosting_tree_model(train_x, train_y):
    """
    param tarin_x: genes train df such that rows are samples
    param train_y: drugs train df such that rows are samples, values are log transformed and no
    missing values.
    """
    # ToDo: Think about which parameternt of GradientBoostingRegressor we want to
    # change from default for testing improvement. For example, n_estimators number.
    return MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(train_x, train_y)
    


