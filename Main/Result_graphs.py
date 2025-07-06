import csv
import numpy as np
from matplotlib import font_manager
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

legends = ['IPSO', 'GLS', 'ACO–DWA', 'LMBSWO', 'MDHO algorithm', 'Proposed ASWO']  # x-axis labels
marker = ['x', '*', ".", ">", "o", 'p', 's']
labels = [2,3,4,5,6]  # metrics

color_code = ['#FFB4F3', '#ef44ff', '#4D0D7A', '#7f0e89', '#390160', '#aa3417']
def plot_graph(file,x_lab,y_lab):
    with open(file + ".csv", 'rt') as f:  #######################file => file --> 1,5,9 -- 2,6,10 -- 3,7,11 -- 4,8,12

        # read & store data in array
        data = []
        content = csv.reader(f)  # content in csv
        for row in content:
            tem = []
            for col in row:
                tem.append(float(col))
            if (len(tem) > 0):  # to remove empty string in csv
                data.append(tem)
        data = np.transpose(data)
    print(len(data))
    methods = legends

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    ax.set_position([0, -0.9, 1, 0.4])  # [left, bottom, width, height]

    # Line plot with markers
    markers = ['x', '*', 'D', 'o', '>', 'P', '.', 's', 'd']
    for i, method in enumerate(data):
        ax.plot(labels, method, marker=markers[i], label=methods[i], linewidth=2, markersize=8, color=color_code[i])
    # ax.scale(1,4)
    # Set axis labels
    ax.set_xlabel(x_lab, fontweight='bold', size=11)
    ax.set_ylabel(y_lab, fontweight='bold', size=11)
    # ax.set_ylim(75, 90)

    # Add legend
    ax.legend()

    # Add table
    cell_text = [[f'{val:.2f}' for val in row] for row in data]
    table = plt.table(cellText=cell_text, rowLabels=methods, colLabels=labels, loc='bottom', cellLoc='center',
                      bbox=[0.2, -0.4, 0.94, 0.28], rowColours=color_code,
                      )
    table.scale(1, 4)  # Adjust the table size

    # Adjust the plot to make space for the table
    plt.subplots_adjust(left=0.1, bottom=0.28, top=0.97)
    # plt.savefig(file + '.jpg')
    # plt.savefig(file + '.svg')
    plt.show()
def plot_graph2(file,x_lab,y_lab):
    plt.figure(figsize=(7, 5), dpi=100)
    with open(file + ".csv", 'rt') as f:  #######################file => file --> 1,5,9 -- 2,6,10 -- 3,7,11 -- 4,8,12

        # read & store data in array
        data = []
        content = csv.reader(f)  # content in csv
        for row in content:
            tem = []
            for col in row:
                tem.append(float(col))
            if (len(tem) > 0):  # to remove empty string in csv
                data.append(tem)
        data = np.transpose(data)
    print(len(data))
    # x-axis labels

    marker = ['x', '*', ".", ">", "o", "d"]
    # labels = ['50', '60','70', '80', '90']  # metrics

    font = font_manager.FontProperties(family='Calibri',  # 'Times new roman',
                                       weight='bold',
                                       style='normal', size=11)
    x = np.array(labels)
    x = x + 1

    linestyles = ['--', '-.', '-', 'dotted', (5, (10, 3)), (0, (5, 10)), (0, (5, 1))]
    markers_list = ['x', '*', 'D', 'o', '>', 'P', '.', 's', 'd']

    colors = ['r', 'g', 'b', 'm', 'c', 'y']
    # new_labels = ['method 1', 'method 2', 'method 3', 'method 4', 'method 5']

    # plot
    for i in range(len(data)):
        y = data[i]
        plt.step(labels, y, label=legends[i], marker=markers_list[i], color=color_code[i], markersize=7,
                 markeredgecolor='k', linewidth=2)

    plt.grid(color='0.9')
    plt.xlabel(x_lab, fontweight='bold', size=11)  ################## Delay
    plt.ylabel(y_lab, fontweight='bold', size=11)
    # plt.xticks(labels, labels)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), fancybox=True, shadow=True, ncol=3, prop=font)

    # plt.savefig(file + '.jpg')  # to show the plot
    # plt.savefig(file + '.svg')  # to show the plot
    plt.show()


file2 = 'graph\\f_m'
file3 = 'graph\\f_s'

file6 = 'graph\\pl_m'
file7 = 'graph\\pl_s'

file10 = 'graph\\ps_m'
file11 = 'graph\\ps_s'


x2_lab = 'Number of Robots'



y1_lab = 'Fitness'
y2_lab = 'Path Length(km)'
y3_lab = 'Path Smoothness(%)'



plot_graph(file2, x2_lab, y1_lab)
plot_graph2(file3, x2_lab, y1_lab)

plot_graph(file6, x2_lab, y2_lab)
plot_graph2(file7, x2_lab, y2_lab)

plot_graph(file10, x2_lab, y3_lab)
plot_graph2(file11, x2_lab, y3_lab)
