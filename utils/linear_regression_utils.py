
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model

LR =  LinearRegression()


def liner_regression(x, y):
    print(f"Num of features (genes): {x.shape[0]}")
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    my_LR.fit(x_train, y_train)
    y_pred = my_LR.predict(x_test)
    score = r2_score(y_test, y_pred)
    return(score) 