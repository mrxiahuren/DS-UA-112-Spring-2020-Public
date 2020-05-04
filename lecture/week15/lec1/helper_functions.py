import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize


def add_patch(ax, coordinates, color = "green"): 
    ax.add_patch(
        matplotlib.patches.Polygon(coordinates,
            color=color,
            fill=False, hatch = '//'
        )
    )
    
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
    
def plot_three_categories_predictions(data, show = True):
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

    if show:
        plt.scatter(data["Feature 1"], data["Feature 2"], c = data["Categories"]);

    plt.scatter(table_predicted_probabilities["x"], 
                table_predicted_probabilities["y"], 
                c = table_predicted_probabilities["Highest Probability"],
                alpha = 0.05)

    plt.ylim([-2.2,2.2])
    
    predicted_probabilities = dict()
    for category, model in models.items():
        predictions = model.predict_proba(data[["Feature 1", "Feature 2"]])
        predicted_probabilities[category] = predictions[:,1]

    table_predicted_probabilities = pd.DataFrame(data = predicted_probabilities)

    table_predicted_probabilities["Highest Probability"] = table_predicted_probabilities.apply(np.argmax, axis = 1);    
  
    return table_predicted_probabilities["Highest Probability"].values

def accuracy(observed, predicted):
    return np.mean(observed == predicted)
       
def precision(observed, predicted):    
    tp = sum((observed == predicted) & (observed == 1))
    fp = sum((observed != predicted) & (observed == 0))
   
    return tp / (tp + fp)    

def recall(observed, predicted):
    tp = sum((observed == predicted) & (observed == 1))
    fn = sum((observed != predicted) & (observed == 1))

    return tp / (tp + fn)

def false_positive_rate(observed, predicted):
    tn = sum((observed == predicted) & (observed == 0))
    fp = sum((observed != predicted) & (observed == 0))
    return fp / (tn + fp)

def plot_confusion(confusion):
    sns.heatmap(confusion, annot=True, fmt='d',
                cmap="Purples", annot_kws={'fontsize': 24}, square=True,
                xticklabels=[2, 1, 0], yticklabels=[2, 1, 0], cbar=False)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')  
    
def soft_max(feature1, feature2, params):
    features = np.array([1,feature1,feature2])
    unnormalized = {key : np.exp(np.dot(features,value)) for key, value in params.items()}
    denominator = sum(unnormalized.values())
    normalized = {key : value / denominator for key, value in unnormalized.items()}  
    return normalized 

def make_params(slopes):
    output = {0:np.array([0,0,0])}
    output.update({1:slopes[:3]})
    output.update({2:slopes[3:]})
    return output

def compute_average_logistic_loss(slopes, feature1, feature2, observed):
    params = make_params(slopes)
    output = 0
    length = len(observed)
    for idx in range(length):
        probs = soft_max(feature1[idx], feature2[idx], params)
        predicted = probs[observed[idx]]
        output += -np.log(predicted)
    
    return output / length    
    