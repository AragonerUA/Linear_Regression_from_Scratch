import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0.0

    def fit(self, X, y):
        if self.fit_intercept:
            I = pd.Series(1, index=X.index)
            X.insert(loc=0, column="I", value=I)
        X = X.to_numpy()
        y = y.to_numpy()
        Xt = X.T
        beta = np.linalg.inv(Xt @ X) @ Xt @ y
        self.coefficient = beta
        if self.fit_intercept:
            self.intercept = beta[0]
            self.coefficient = beta[1:]

    def predict(self, X):
        X = X.to_numpy()
        y = self.intercept + X @ self.coefficient
        return y

    def r2_score(self, y, yhat):
        numerator = 0
        denominator = 0
        for i in range(len(y)):
            numerator += (y[i] - yhat[i])**2
            denominator += (y[i] - y.mean())**2
        return 1 - (numerator / denominator)

    def rmse(self, y, yhat):
        summ_mse = 0
        for i in range(len(y)):
            summ_mse += (y[i] - yhat[i])**2
        return sqrt((1/len(y)) * summ_mse)


if __name__ == "__main__":
    # First Stage
    '''
    data_dict = {"x": [4.0, 4.5, 5, 5.5, 6.0, 6.5, 7.0], "y": [33, 42, 45, 51, 53, 61, 62]}

    '''
    # data_dict = {"x": [4.0, 7.0], "y": [10.0, 16.0]}
    # data_dict = {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.0, 0.0, 0.0, 0.0, 0.0]}
    # data_dict = {"x": [1.0, 4.5, 14.0, 3.8, 7.0, 19.4], "y": [106.0, 150.7, 200.9, 115.8, 177, 156]}
    '''
    data_df = pd.DataFrame(data_dict)
    model = CustomLinearRegression()
    print(model.fit(np.array(data_df.x), np.array(data_df.y)))
    '''

    # Second Stage
    '''
    data_dict = {"x": [4, 4.5, 5, 5.5, 6, 6.5, 7],
                 "w": [1, -3, 2, 5, 0, 3, 6],
                 "z": [11, 15, 12, 9, 18, 13, 16],
                 "y": [33, 42, 45, 51, 53, 61, 62]}


    # data_dict = {"x": [1, 1, 1, 7],
    #              "w": [27, 2, 2, 7],
    #              "z": [3, 6, 3, 7],
    #              "y": [8, 6, 6, 21]}
                 
    df = pd.DataFrame(data_dict)

    regCustom = CustomLinearRegression(fit_intercept=False)
    regCustom.fit(df[['x', 'w', 'z']], df['y'])
    y_pred = regCustom.predict(df[['x', 'w', 'z']])
    print(y_pred)
    '''

    # Third Stage

    '''
    df = pd.DataFrame({
        'Capacity': [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9],
        'Age': [11, 11, 9, 8, 7, 7, 6, 5, 5, 4],
        'Cost/ton': [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69],
    })

    regCustom = CustomLinearRegression()
    regCustom.fit(df[["Capacity", "Age"]], df["Cost/ton"])
    y_hat = regCustom.predict(df[["Capacity", "Age"]])
    rmserr = regCustom.rmse(df["Cost/ton"], y_hat)
    r2_sc = regCustom.r2_score(df["Cost/ton"], y_hat)
    answer_dict = {'Intercept': regCustom.intercept,
                   'Coefficient': regCustom.coefficient,
                   'R2': r2_sc,
                   'RMSE': rmserr}
    print(answer_dict)
    '''

    # Fourth Stage

    df = pd.DataFrame({"f1": [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87],
                       "f2": [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3],
                       "f3": [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2],
                       "y": [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]})
    model = LinearRegression()
    model.fit(df[["f1", "f2", "f3"]], df["y"])
    pred_sk = model.predict(df[["f1", "f2", "f3"]])
    rmse_sk = sqrt(mean_squared_error(df["y"], pred_sk))
    r2sc_sk = r2_score(df["y"], pred_sk)

    regCustom = CustomLinearRegression()
    regCustom.fit(df[["f1", "f2", "f3"]], df["y"])
    y_hat = regCustom.predict(df[["f1", "f2", "f3"]])
    rmserr = regCustom.rmse(df["y"], y_hat)
    r2_sc = regCustom.r2_score(df["y"], y_hat)
    answer_dict = {'Intercept': model.intercept_ - regCustom.intercept,
                   'Coefficient': model.coef_ - regCustom.coefficient,
                   'R2': r2sc_sk - r2_sc,
                   'RMSE': rmse_sk - rmserr}
    print(answer_dict)
