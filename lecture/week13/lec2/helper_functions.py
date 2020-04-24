import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge

def plot_lasso_models(data, extra_parameters = np.logspace(-3,1,15)):
    lasso_models = dict()
    extra_parameters_lasso = extra_parameters

    for extra_parameter in extra_parameters_lasso:
        lasso_model = Lasso(alpha = extra_parameter, fit_intercept=False, max_iter=10000)
        lasso_model.fit(data.drop(columns = "Response Variable"), data["Response Variable"])

        lasso_models[extra_parameter] = lasso_model

    labels = ['Explanatory Variable Small', 'Explanatory Variable Large']

    coefs = []
    for extra_parameter in extra_parameters_lasso:
        model = lasso_models[extra_parameter]
        coefs.append(model.coef_)

    coefs = zip(*coefs)
    for coef, label in zip(coefs, labels):
        plt.semilogx(extra_parameters_lasso, coef, label = label)

    plt.xlabel('Extra Parameter $\lambda$')
    plt.ylabel('Slopes')
    plt.title('Lasso Regression Slopes')
    plt.legend();
    
def plot_ridge_models(data, extra_parameters = np.logspace(1,5,15)):
    ridge_models = dict()
    extra_parameters_ridge = extra_parameters

    for extra_parameter in extra_parameters_ridge:
        ridge_model = Ridge(alpha = extra_parameter, fit_intercept=False, max_iter=10000)
        ridge_model.fit(data.drop(columns = "Response Variable"), data["Response Variable"])

        ridge_models[extra_parameter] = ridge_model

    labels = ['Explanatory Variable Small', 'Explanatory Variable Large']

    coefs = []
    for extra_parameter in extra_parameters_ridge:
        model = ridge_models[extra_parameter]
        coefs.append(model.coef_)

    coefs = zip(*coefs)
    for coef, label in zip(coefs, labels):
        plt.semilogx(extra_parameters_ridge, coef, label = label)

    plt.xlabel('Extra Parameter $\lambda$')
    plt.ylabel('Slopes')
    plt.title('Ridge Regression Slopes')
    plt.legend();