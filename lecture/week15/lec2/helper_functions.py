import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import LogisticRegression, LinearRegression 
from scipy.optimize import minimize


def sample_without_replacement(sample_size, data):
    return data.sample(sample_size, replace = False)

def sample_with_replacement(sample_size, table):
    return table.sample(sample_size, replace = True)

def calculate_percentile(data, percentile):
    sorted_data = sorted(data)
    length_data = len(sorted_data)
    percentile_fraction = percentile / 100 
    index = np.ceil(percentile_fraction * length_data) - 1
    return sorted_data[int(index)]

def calculate_pvalue(observed_test_statistic, simulated_test_statistics):
    how_many_less_than = 0

    for value in simulated_test_statistics:
        if value <= observed_test_statistic:
            how_many_less_than = how_many_less_than + 1
         
    trials = len(simulated_test_statistics)
            
    return how_many_less_than / trials
    
def convert_standard_units(array):
    return (array - np.mean(array)) / np.std(array)

def correlation(table, explanatory_variable, response_variable):
    x_standard = convert_standard_units(table[explanatory_variable])
    y_standard = convert_standard_units(table[response_variable])
    
    return np.mean(x_standard * y_standard)

def slope(table, explanatory_variable, response_variable):
    r = correlation(table, explanatory_variable, response_variable)
    y_sd = np.std(table[response_variable])
    x_sd = np.std(table[explanatory_variable])
    return r * y_sd / x_sd

def intercept(table, explanatory_variable, response_variable):
    y_mean = np.mean(table[response_variable])
    x_mean = np.mean(table[explanatory_variable])
    return y_mean - slope(table, explanatory_variable, response_variable) * x_mean

def fitted_values(table, explanatory_variable, response_variable):
    a = slope(table, explanatory_variable, response_variable)
    b = intercept(table, explanatory_variable, response_variable)
    return a * table[explanatory_variable] + b

def generate_blob(mean, covariance, number):
    coordinates = np.random.randn(2, number)
    coordinates = mean.reshape(2,-1) + np.dot(covariance, coordinates)
    return coordinates

def generate_other_sample():
    mean = np.array([4.5, 73])
    covariance = np.array([[0.3,0],[0,4]])
    number = 100
    coordinates1 = generate_blob(mean, covariance, number)    


    mean = np.array([2.2, 50])
    covariance = np.array([[0.3,0],[0,4]])
    number = 100
    coordinates2 = generate_blob(mean, covariance, number)    

    coordinates = np.empty(coordinates1.shape + (2,))
    coordinates[:,:,0] = coordinates1
    coordinates[:,:,1] = coordinates2
    
    return coordinates

def add_patch(ax, coordinates, color = "green"): 
    ax.add_patch(
        matplotlib.patches.Polygon(coordinates,
            color=color,
            fill=False, hatch = '//'
        )
    )
    

def plot_least_squares(data):
    patients = data 
    model = LinearRegression()
    model.fit(patients.drop(columns = ["Class", "Class Color"]), patients["Class"])

    num_points = 30

    uvalues = np.linspace(0, 10, num_points)
    vvalues = np.linspace(0, 10, num_points)
    (u,v) = np.meshgrid(uvalues, vvalues)

    pairs = np.vstack((u.flatten(),v.flatten())).T
    tvalues = model.predict(pairs)

    loss_surface = go.Surface(x=u, y=v, z=tvalues.reshape(u.shape) )

    fig = go.Figure(data=[loss_surface])


    xvalues = patients["Bland Chromatin"].values
    yvalues = patients["Single Epithelial Cell Size"].values
    zvalues = patients["Class"].values
    colors = patients["Class Color"].values

    data_scatter = go.Scatter3d(x=xvalues, 
                        y=yvalues, 
                        z=zvalues, 
                        mode='markers', 
                        marker={"color":colors})

    fig.add_trace(data_scatter)

    fig.update_layout(scene = dict(
        xaxis_title = "Bland Chromatin",
        yaxis_title = "Single Epithelial Cell Size",
        zaxis_title = "Benign or Malignant"))

    fig.show()    
    
def plot_logistic(data):
    patients = data 
    model = LogisticRegression()
    model.fit(patients.drop(columns = ["Class", "Class Color"]), patients["Class"])

    num_points = 30

    uvalues = np.linspace(0, 10, num_points)
    vvalues = np.linspace(0, 10, num_points)
    (u,v) = np.meshgrid(uvalues, vvalues)

    pairs = np.vstack((u.flatten(),v.flatten())).T
    tvalues = model.predict(pairs)

    loss_surface = go.Surface(x=u, y=v, z=tvalues.reshape(u.shape) )

    fig = go.Figure(data=[loss_surface])


    xvalues = patients["Bland Chromatin"].values
    yvalues = patients["Single Epithelial Cell Size"].values
    zvalues = patients["Class"].values
    colors = patients["Class Color"].values

    data_scatter = go.Scatter3d(x=xvalues, 
                        y=yvalues, 
                        z=zvalues, 
                        mode='markers', 
                        marker={"color":colors})

    fig.add_trace(data_scatter)

    fig.update_layout(scene = dict(
        xaxis_title = "Bland Chromatin",
        yaxis_title = "Single Epithelial Cell Size",
        zaxis_title = "Benign or Malignant"))

    fig.show()    
    
def plot_distance(data, point_to_classify, distance):
    number_of_records = len(data)

    np.random.seed(42)
    plt.scatter(data["Bland Chromatin"] + 0.5 * np.random.rand(number_of_records), 
                data["Single Epithelial Cell Size"] + 0.5 * np.random.rand(number_of_records), 
                c = data["Class Color"])

    point_to_classify = np.array(point_to_classify).reshape(2,1)
    plt.scatter(point_to_classify[0,0], point_to_classify[1,0], c="green", marker = "^")

    distance = distance
    radians = np.linspace(0,2* np.pi, 100)
    values = point_to_classify + distance * np.vstack([np.cos(radians), np.sin(radians)])
    plt.plot(values[0,:], values[1,:], "g")

    plt.xlim([0,11])
    plt.ylim([0,11])
    plt.xlabel("Bland Chromatin")
    plt.ylabel("Single Epithelial Cell Size")
    plt.title(f"Distance {distance} from Point ({point_to_classify[0,0]},{point_to_classify[1,0]})")
    
    plt.show();

def plot_distance_widget(data, point_to_classify): 
    interact(
    plot_distance,
    data = fixed(data),
    point_to_classify= fixed(point_to_classify),
    distance=widgets.FloatSlider(min=0.01, max=3, step=0.1, value=1, msg_throttle=1))

def plot_neighbors(data, point_to_classify, distance):
    number_of_records = len(data)

    np.random.seed(42)
    jitter_x = 0.5 * np.random.rand(number_of_records)
    jitter_y = 0.5 * np.random.rand(number_of_records)
    plt.scatter(data["Bland Chromatin"] + jitter_x, 
                data["Single Epithelial Cell Size"] + jitter_y, 
                c = data["Class Color"])

    point_to_classify = np.array(point_to_classify).reshape(2,1)
    plt.scatter(point_to_classify[0,0], point_to_classify[1,0], c="green", marker = "^")

    distance = distance
    radians = np.linspace(0,2* np.pi, 100)
    values = point_to_classify + distance * np.vstack([np.cos(radians), np.sin(radians)])
    plt.plot(values[0,:], values[1,:], "g")

    xvalues, yvalues, value_counts = get_points_within_distance(data, point_to_classify, distance, jitter_x, jitter_y)
    
    if len(xvalues) > 0:
        plt.scatter(xvalues, yvalues, c="green")
    
    if len(value_counts) < 2:
        value_counts = [0,0]
    
    plt.xlim([0,11])
    plt.ylim([0,11])
    plt.xlabel("Bland Chromatin")
    plt.ylabel("Single Epithelial Cell Size")
    plt.title(f"{value_counts[0]} Blue Points and {value_counts[1]} Red Points within Distance {distance} from Point ({point_to_classify[0,0]},{point_to_classify[1,0]})")
    
    plt.show();

def get_points_within_distance(data, point_to_classify, distance, jitter_x, jitter_y):
    patients_copy = data.copy()
    patients_copy["Bland Chromatin"] += jitter_x
    patients_copy["Single Epithelial Cell Size"] += jitter_y
    patients_copy["Distance"] = np.sqrt( (patients_copy["Bland Chromatin"] - point_to_classify[0])**2 + \
                                        (patients_copy["Single Epithelial Cell Size"]  - point_to_classify[1])**2 )
    patients_copy = patients_copy.sort_values("Distance", ascending = True)

    xvalues, yvalues, value_counts = [],[],[0,0]
    
    if len(patients_copy[patients_copy["Distance"] <= distance]) > 0:
        xvalues = patients_copy[patients_copy["Distance"] <= distance]["Bland Chromatin"]
        yvalues = patients_copy[patients_copy["Distance"] <= distance]["Single Epithelial Cell Size"]
        value_counts = patients_copy[patients_copy["Distance"] <= distance]["Class"].value_counts() 
                
    return xvalues, yvalues, value_counts    
    
def plot_neighbors_widget(data, point_to_classify): 
    interact(
    plot_neighbors,
    data = fixed(data),
    point_to_classify= fixed(point_to_classify),
    distance=widgets.FloatSlider(min=0.01, max=3, step=0.1, value=1, msg_throttle=1))
    
def generate_background(original_data, number_of_neighbors, num_points):
    uvalues = np.linspace(0, 11, num_points)
    vvalues = np.linspace(0, 11, num_points)
    (u,v) = np.meshgrid(uvalues, vvalues)

    pairs = np.vstack((u.flatten(),v.flatten())).T
    predicted_category = np.array([classify_point(original_data, point_to_classify, number_of_neighbors) for point_to_classify in pairs])
    predicted_category = ["red" if value == 1 else "blue" for value in predicted_category]
    
    return predicted_category, u, v

def generate_background_plot(original_data, number_of_neighbors, show = True, num_points = 50):
    
    fig,ax = plt.subplots(1,1)
    
    patients = original_data
    number_of_records = len(patients)

    if show:
        ax.scatter(patients["Bland Chromatin"] + 0.5 * np.random.rand(number_of_records), 
                    patients["Single Epithelial Cell Size"] + 0.5 * np.random.rand(number_of_records), 
                    c = patients["Class Color"])

    predicted_category, u, v =  generate_background(original_data, number_of_neighbors, num_points)
    
    ax.scatter(u.flatten(), 
                v.flatten(), 
                c = predicted_category,
                alpha = 0.25)

    ax.set_xlabel("Bland Chromatin")
    ax.set_ylabel("Single Epithelial Cell Size")
    ax.set_title("Scatter-plot of Classifications")
#     plt.show()
    
    return ax  

    
def compute_distance(horizontal_coordinates, vertical_coordinates, point_to_classify):
    distance_squared = (horizontal_coordinates - point_to_classify[0])**2 + (vertical_coordinates - point_to_classify[1])**2
    return np.sqrt(distance_squared)  

def proportion_among_neighbors(original_data, point_to_classify, number_of_neighbors):
    data = original_data.copy()
    
    data["Distance"] = compute_distance(data["Bland Chromatin"],
                                        data["Single Epithelial Cell Size"],
                                        point_to_classify)
    
    data = data.sort_values("Distance", ascending=True)

    smallest_distances = data[:number_of_neighbors]

    number_in_class_1 = np.sum(smallest_distances["Class"])
    number_in_class_0 = number_of_neighbors - number_in_class_1

    return number_in_class_0 / number_of_neighbors, number_in_class_1 / number_of_neighbors

def classify_point(original_data, point_to_classify, number_of_neighbors):
    fraction_blue, fraction_red = proportion_among_neighbors(original_data, point_to_classify, number_of_neighbors)     
    
    if fraction_blue >= fraction_red:
        category = 0
    else:
        category = 1
        
    return category

def compute_accuracy(training_set, testing_set, number_of_neighbors):
    points_to_classify = testing_set[["Bland Chromatin", "Single Epithelial Cell Size"]]

    predicted_categories = [] 
    for point_to_classify in points_to_classify.values: 
        predicted_categories.append(classify_point(training_set, point_to_classify, number_of_neighbors)) 

    observed_categories = testing_set["Class"]    
        
    number_of_records_testing_set = len(testing_set)  
            
    number_correct_classification = 0 
    for row in range(number_of_records_testing_set):
        if predicted_categories[row] == observed_categories.values[row]:
            number_correct_classification = number_correct_classification + 1

    accuracy = number_correct_classification / number_of_records_testing_set
    return accuracy
