import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual


def add_patch(ax, coordinates, color = "green"): 
    ax.add_patch(
        matplotlib.patches.Polygon(coordinates,
            color=color,
            fill=False, hatch = '//'
        )
    )
    

def plot_confusion(confusion):
    sns.heatmap(confusion, annot=True, fmt='d',
                cmap="Purples", annot_kws={'fontsize': 24}, square=True,
                xticklabels=[1, 0], yticklabels=[1, 0], cbar=False)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')    
        
def sigmoid(t):
    return 1 / (1 + np.exp(-t))
    
def logistic_loss(y, y_hat):
    return -y * np.log(y_hat + 1e-5) - (1-y) * np.log(1-y_hat + 1e-5)

def average_logistic_loss(w, data, explanatory_variable, response_variable):
    x = data[explanatory_variable]
    y = data[response_variable]
    y_hat = sigmoid(w * x)
    return np.mean(logistic_loss(y, y_hat))

def regularized_average_logistic_loss(w, data, explanatory_variable, response_variable, alpha):
    penalty = alpha * w**2 
    return penalty + average_logistic_loss(w, data, explanatory_variable, response_variable)         

def plotter(w, data):
    xs = np.linspace(-5, 5, 100)
    ys = sigmoid(w * xs)
    yhat = sigmoid(w * data["X"])

    for xvalue, yvalue, yvalue_hat in zip(data["X"],data["Y"], yhat): 
        ymin = min(yvalue, yvalue_hat)
        ymax = max(yvalue, yvalue_hat)
        
        plt.vlines(xvalue, ymin, ymax, color = "red", linestyle = "dashed")
    
    plt.scatter(data['X'], data['Y'])
    plt.plot(xs, ys)
    
    mse = average_logistic_loss(w, data, "X","Y")
    plt.title(f"Average Logistic Loss {mse:0.2f}");
    
def regularizer(data, alpha):
    slope = minimize(regularized_average_logistic_loss, x0 = 1, args=(data, "X", "Y", alpha)).x

    plotter(slope, data)
    
def regression_widget(data): 
    interact(
    plotter,
    data = fixed(data),
    w=widgets.FloatSlider(min=-5, max=10, step=1, value=0, msg_throttle=1))    

    
def regularization_widget(data): 
    interact(
    regularizer,
    data = fixed(data),
    alpha=widgets.FloatSlider(min=0.1, max=3, step=0.1, value=0.1, msg_throttle=1)) 
    
def plot_three_categories(data):
    models = dict()
    for category in data["Categories"].unique():
        categories = np.where(data["Categories"] == category, 1, 0)

        model = LogisticRegression()
        model.fit(data[["Feature 1", "Feature 2"]], categories)
        models[category] = model    

    xvalues = np.linspace(-2,2,50)

    decision_boundaries = dict()
    for category, model in models.items():
        yvalues = (-model.intercept_ - np.squeeze(model.coef_)[0] * xvalues) / np.squeeze(model.coef_)[1]
        decision_boundaries[category] = yvalues

    plt.scatter(data["Feature 1"], data["Feature 2"], c = data["Categories"]);
    plt.plot(xvalues, decision_boundaries[0], color = "purple")
    plt.plot(xvalues, decision_boundaries[1], color = "green")
    plt.plot(xvalues, decision_boundaries[2], color = "yellow")

    plt.ylim([-3,3]);
    
def plot_three_categories_predictions(data):
    models = dict()
    for category in data["Categories"].unique():
        categories = np.where(data["Categories"] == category, 1, 0)

        model = LogisticRegression()
        model.fit(data[["Feature 1", "Feature 2"]], categories)
        models[category] = model  

    predicted_probabilities = dict()
    number_points = 100

    xvalues = np.linspace(-2.2,2.2,number_points)
    yvalues = np.linspace(-2.2,2.2,number_points)
    x, y = np.meshgrid(xvalues, yvalues)

    for category, model in models.items():
        predictions = model.predict_proba(np.vstack((x.flatten(), y.flatten())).T)
        predicted_probabilities[category] = predictions[:,1]

    table_predicted_probabilities = pd.DataFrame(data = predicted_probabilities)

    table_predicted_probabilities["Highest Probability"] = table_predicted_probabilities.apply(np.argmax, axis = 1)
    table_predicted_probabilities["x"] = x.flatten()
    table_predicted_probabilities["y"] = y.flatten()

    plt.scatter(data["Feature 1"], data["Feature 2"], c = data["Categories"]);

    plt.scatter(table_predicted_probabilities["x"], 
                table_predicted_probabilities["y"], 
                c = table_predicted_probabilities["Highest Probability"],
                alpha = 0.05)

    plt.ylim([-2.2,2.2]);    
    


def plot_decision_boundary(data, model):

    x_min = np.min(data['Feature 1']) 
    x_max = np.max(data['Feature 1'])
    y_min = np.min(data['Feature 2']) 
    y_max = np.max(data['Feature 2'])

    xs = np.linspace(x_min, x_max, 100)
    ys = np.linspace(y_min, y_max, 100)

    x, y = np.meshgrid(xs, ys)

    points = pd.DataFrame({
        'xs': x.flatten(),
        'ys': y.flatten(),
    })

    points["pred"] = model.predict(points)
    return points   

def plot_decision_boundary_polynomial(data, model):

    x_min = np.min(data['Feature 1']) 
    x_max = np.max(data['Feature 1'])
    y_min = np.min(data['Feature 2']) 
    y_max = np.max(data['Feature 2'])

    xs = np.linspace(x_min, x_max, 100)
    ys = np.linspace(y_min, y_max, 100)

    x, y = np.meshgrid(xs, ys)

    points = pd.DataFrame({
        'xs': x.flatten(),
        'ys': y.flatten(),
        'xs^2': x.flatten()**2,
        'ys^2': y.flatten()**2,
        'xs x ys': x.flatten() * y.flatten()
    })

    points["pred"] = model.predict(points)
    return points   

       