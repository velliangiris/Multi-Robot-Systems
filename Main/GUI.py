from Main import Run
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import os, logging, warnings, numpy as np
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True
sg.change_look_and_feel('DarkTeal5')  # for default look and feel

# Designing layout
layout = [[sg.Text("")],
            [sg.Text("\t\t\tSelect nRobots\t\t"), sg.Combo(['2', '3','4','5','6'])],
          [sg.Text("\t\t\tSelect Setup\t\t"), sg.Combo(['Single Target', 'Multiple Target']), sg.Text("\t"), sg.Button("START", size=(10, 2))],

          [sg.Text("\t\t    IPSO\t\t\t   GLS\t ACO–DWA\t\t   LMBSWO \t\tMDHO algorithm  \t Proposed ASWO")],
          # [sg.Text('Energy\t  '), sg.In(key='11', size=(20, 20)), sg.In(key='12', size=(20, 20)),
          #  sg.In(key='13', size=(20, 20)), sg.In(key='14', size=(20, 20)), sg.In(key='15', size=(20, 20)), sg.In(key='16', size=(20, 20)), sg.In(key='17', size=(20, 20))],
          [sg.Text('Path Smothness  '), sg.In(key='21', size=(20, 20)), sg.In(key='22', size=(20, 20)),
           sg.In(key='23', size=(20, 20)), sg.In(key='24', size=(20, 20)), sg.In(key='25', size=(20, 20)), sg.In(key='26', size=(20, 20))],
          [sg.Text('Path Length\t '), sg.In(key='31', size=(20, 20)), sg.In(key='32', size=(20, 20)),
           sg.In(key='33', size=(20, 20)), sg.In(key='34', size=(20, 20)), sg.In(key='35', size=(20, 20)), sg.In(key='36', size=(20, 20))],
          [sg.Text('Fitness\t\t'), sg.In(key='41', size=(20, 20)), sg.In(key='42', size=(20, 20)),
           sg.In(key='43', size=(20, 20)), sg.In(key='44', size=(20, 20)), sg.In(key='45', size=(20, 20)), sg.In(key='46', size=(20, 20))],

          [sg.Text("\t\t\t\t\t\t"), sg.Button('Run graph', size=(10, 1)), sg.Text("\t\t"),
           sg.Button('Close', size=(10, 1))], [sg.Text("")]]


# to plot graph
def plot_graph(result_1, result_2, result_3):
    loc, result = [], []
    result.append(result_1)  # appending the result
    result.append(result_2)
    result.append(result_3)
    # result.append(result_4)
    result = np.transpose(result)

    # labels
    labels = ['IPSO', 'GLS', 'ACO–DWA', 'LMBSWO', 'MDHO algorithm', 'Proposed ASWO']  # x-axis labels
    tick_labels = ['Path Smothness', 'Path Length','Fitness']  # metrics
    bar_width, s = 0.12, 0.0  # bar width, space between bars

    for i in range(len(result)):  # allocating location for bars
        if i is 0:  # initial location - 1st result
            tem = []
            for j in range(len(tick_labels)):
                tem.append(j + 1)
            loc.append(tem)
        else:  # location from 2nd result
            tem = []
            for j in range(len(loc[i - 1])):
                tem.append(loc[i - 1][j] + s + bar_width)
            loc.append(tem)

    # plotting a bar chart
    for i in range(len(result)):
        plt.bar(loc[i], result[i], label=labels[i], tick_label=tick_labels, width=bar_width, edgecolor='black')

    plt.legend()  # show a legend on the plot -- here legends are metrics
    plt.show()  # to show the plot


# Create the Window layout
window = sg.Window('51303_CP4', layout)

# event loop
while True:
    event, value = window.read()  # displays the window

    if event == "START":
        n_r,setup=int(value[0]),value[1]  # reads user input
        Path_Smothness, Path_Length,fitness = Run.callmain(setup, n_r)  # call to main code
        #Path_Smothness.sort(reverse=True),Mi_cost.sort(reverse=True),Ene_Cons.sort(reverse=True)

        # window['11'].Update(Energy[0])
        # window['12'].Update(Energy[1])
        # window['13'].Update(Energy[2])
        # window['14'].Update(Energy[3])
        # window['15'].Update(Energy[4])
        # window['16'].Update(Energy[5])
        # window['17'].Update(Energy[6])

        window['21'].Update(Path_Smothness[0])
        window['22'].Update(Path_Smothness[1])
        window['23'].Update(Path_Smothness[2])
        window['24'].Update(Path_Smothness[3])
        window['25'].Update(Path_Smothness[4])
        window['26'].Update(Path_Smothness[5])
        # window['27'].Update(Path_Smothness[6])

        window['31'].Update(Path_Length[0])
        window['32'].Update(Path_Length[1])
        window['33'].Update(Path_Length[2])
        window['34'].Update(Path_Length[3])
        window['35'].Update(Path_Length[4])
        window['36'].Update(Path_Length[5])
        # window['37'].Update(Path_Length[6])

        window['41'].Update(fitness[0])
        window['42'].Update(fitness[1])
        window['43'].Update(fitness[2])
        window['44'].Update(fitness[3])
        window['45'].Update(fitness[4])
        window['46'].Update(fitness[5])
        # window['47'].Update(fitness[6])



        print("\nDone.!")

    if event == 'Run graph':
        plot_graph(Path_Smothness,Path_Length, fitness)

    if event == 'Close':
        window.close()
        break
