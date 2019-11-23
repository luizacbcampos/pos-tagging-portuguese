import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Receives a results file path and returns it as a dictionary
def return_dict(df):
    index = df.columns.values[0] # indexing using the first column of any giver dataframe
    dict = df.set_index(index).T.to_dict('records') # returns a list of dicts
    dict = dict[0] # so we need to get the only dict that matters, the first and only
    return dict

# This method creates a graph comparing every class's accuracy using it's respective results file
def graph_by_class(file_path,window_size,epochs):
    df = pd.read_csv(file_path)
    class_result = return_dict(df)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    keys = [(k) for k,v in sorted(class_result.items())]
    y_pos = np.arange(len(np.array(keys)))
    ax.barh(y_pos, np.array([(v) for k,v in sorted(class_result.items())]), align='center', color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.array(keys))
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy')
    ax.set_title('Accuracy by Class')
    ax.grid(True)

    plt.xlim(0.92, 1.00005)
    if not os.path.exists('breno_results/graphs'):
        os.makedirs('breno_results/graphs')

    plt.savefig('breno_results/graphs/accuracy_by_class_'+str(window_size)+"-"+str(epochs)+'.png', bbox_inches='tight', pad_inches=1)
    del keys
    plt.close('all')

# Creates one graph representing data stored in the "total_accuracy.csv" file.
def graph_by_window_size(file_path):
    df = pd.read_csv(file_path)
    #df = df.groupby(['epochs'])

    fig,ax = plt.subplots()
    x = np.arange(len(df.window_size.unique()))
    bar_width = 0.1
    b1 = ax.bar(x, df.loc[df['epochs'] == 1, 'accuracy'],
            width=bar_width, label = '1')
    # Same thing, but offset the x by the width of the bar.
    b2 = ax.bar(x + bar_width, df.loc[df['epochs'] == 2, 'accuracy'],
            width=bar_width, label = '2')
    b3 = ax.bar(x + 2*bar_width, df.loc[df['epochs'] == 3, 'accuracy'],
            width=bar_width, label = '3')
    b4 = ax.bar(x + 3*bar_width, df.loc[df['epochs'] == 4, 'accuracy'],
            width=bar_width, label = '4')
    b5 = ax.bar(x + 4*bar_width, df.loc[df['epochs'] == 5, 'accuracy'],
            width=bar_width, label = '5')
    
    ax.set_xticks(x + 2*bar_width)
    ax.set_xticklabels(df.window_size.unique())
    ax.ticklabel_format(axis= 'y',useOffset=False)
    plt.ylim(0.9615444, 0.9615448)
    ax.grid(True)
    ax.legend(title="epochs")
    plt.title("Window Size x Accuracy")
    plt.xlabel("Window Size")
    plt.ylabel("Accuracy")
    plt.show()
    return
    if not os.path.exists('breno_results/graphs'):
        os.makedirs('breno_results/graphs')

    plt.savefig('breno_results/graphs/total_accuracy.png', bbox_inches='tight', pad_inches=1)
    plt.close('all')

#graph_by_window_size("results/total_accuracy.csv")
# graph_by_class("results/7-1.csv",3,1)