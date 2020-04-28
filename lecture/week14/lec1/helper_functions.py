import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter_plot(data, amount_jitter):
    # Get records
    malignant_records = data[data["class"] == 1] 
    benign_records =  data[data["class"] == 0]

    # jitter points
    random_x = amount_jitter * np.random.rand(len(malignant_records))
    random_y = amount_jitter * np.random.rand(len(malignant_records))

    # Create the scatter plot
    plt.plot(random_x + malignant_records["size_unif"],random_y + malignant_records["marg"],'r.', label = "malignant")
    plt.plot(benign_records["size_unif"],benign_records["marg"],'g.', label = "benign")
    plt.xlabel("size_unif")
    plt.ylabel("marg")
    plt.ylim(0,14)
    plt.legend();

def plot_histogram(data, explanatory_variables, response_variables):
    range_of_xvalues = np.sort(data[explanatory_variables[0]].unique())
    range_of_yvalues = np.sort(data[explanatory_variables[1]].unique())

    range_of_xvalues = np.append(range_of_xvalues, np.max(range_of_xvalues) + 1)
    range_of_yvalues = np.append(range_of_yvalues, np.max(range_of_yvalues) + 1)

    for class_value, class_color, label in zip([0,1], ['g','r'], ['benign','malign']):
        table_subset = data[data[response_variables] == class_value]

        histogram_height, bin_edges_xvalues, bin_edges_yvalues = np.histogram2d(table_subset[explanatory_variables[0]], 
                                                                             table_subset[explanatory_variables[1]],
                                                                             [range_of_xvalues, range_of_yvalues])

        grid_xvalues, grid_yvalues = np.meshgrid(bin_edges_xvalues[:-1], bin_edges_yvalues[:-1])
        plt.scatter(grid_xvalues.flatten(), 
                    grid_yvalues.flatten(), 
                    s=4*histogram_height.flatten(),
                    alpha=0.5, 
                    c=class_color,
                    edgecolors='none',
                    label = label)


    plt.ylim([0,14])
    plt.legend()
    plt.xlabel(explanatory_variables[0])
    plt.ylabel(explanatory_variables[1]);
    
def add_patch(ax, coordinates, color = "green"): 
    ax.add_patch(
        matplotlib.patches.Polygon(coordinates,
            color=color,
            fill=False, hatch = '//'
        )
    )
    
def sigmoid(t):
    return 1 / (1 + np.exp(-t))
    
def plot_confusion(confusion):
    sns.heatmap(confusion, annot=True, fmt='d',
                cmap="Purples", annot_kws={'fontsize': 24}, square=True,
                xticklabels=[1, 0], yticklabels=[1, 0], cbar=False)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')    
    
def plot_sigmoid():
    nx = 100
    xm = 10
    slopes = np.array([-0.5,0,0.5,1,2,10])

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize = (16,10))
    ax = np.array(ax).flatten()

    xvalues_scatter  = np.random.uniform(-xm,xm,nx)
    yvalues_scatter = sigmoid(xvalues_scatter)
    yvalues_scatter = (np.random.rand(nx) < yvalues_scatter).astype(int)

    for axis, slope in zip(ax, slopes):
        axis.scatter(xvalues_scatter, 
                    yvalues_scatter,
                    c=yvalues_scatter,
                    edgecolors='none',
                    marker='o')

        xvalues_curve = np.linspace(-xm,xm,100)
        yvalues_curve = sigmoid(slope * xvalues_curve)
        axis.plot(xvalues_curve, yvalues_curve,'b-')

        axis.grid()
        axis.set_title('Slope {0:.1f}'.format(slope))

    plt.subplots_adjust(wspace=0.9, hspace=0.5)